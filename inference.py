import os
import sys
import time
import json
import re
import argparse
import random
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    LogitsProcessor,
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# System configurations
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


# XAI utilities
class TTFTProcessor(LogitsProcessor):
    def __init__(self):
        self.first_call = True
        self.ttft_event = torch.cuda.Event(enable_timing=True)
        self.recorded = False

    def __call__(self, input_ids, scores):
        if self.first_call:
            self.ttft_event.record()
            self.first_call = False
            self.recorded = True
        return scores


# Preprocessing utilities
def normalize_c_code_soft(code):
    if not isinstance(code, str):
        return ""
    code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    code = re.sub(r"\n\s*\n", "\n", code)
    code = re.sub(r" +$", "", code, flags=re.MULTILINE)
    return code.strip()


def sanitize_pair(code, text):
    patterns = [
        (r"(?i)bad_", "func_"),
        (r"(?i)_bad", "_func"),
        (r"(?i)good_", "func_"),
        (r"(?i)_good", "_func"),
        (r"(?i)cwe\d+", "vuln_id"),
        (r"(?i)cwe-\d+", "vuln_id"),
        (r"(?i)bad", "func"),
        (r"(?i)good", "func"),
        (r"(?i)safe", "func"),
        (r"(?i)vuln", "proc"),
    ]
    for pat, repl in patterns:
        code = re.sub(pat, repl, code)
    return code, text


def preprocess_data_offline(data_list, tokenizer, config_model, show_progress=True):
    processed_samples = []
    neutral_text = "Analyze code security."
    max_len = config_model["max_code_len"]

    iterator = (
        tqdm(data_list, desc="[INFO] Tokenizing", unit="seq")
        if show_progress
        else data_list
    )

    for item in iterator:
        code = normalize_c_code_soft(str(item.get("text", "")))
        raw_detail = str(item.get("detail", ""))
        label_bin = int(item.get("label", 0))
        code, clean_text = sanitize_pair(code, raw_detail)
        target_text = clean_text if len(clean_text) > 5 else neutral_text

        code_tokens = tokenizer(
            code,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_tokens = tokenizer(
            target_text,
            max_length=config_model["max_desc_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        processed_samples.append(
            {
                "input_ids": code_tokens["input_ids"].flatten(),
                "attention_mask": code_tokens["attention_mask"].flatten(),
                "labels_ids": text_tokens["input_ids"].flatten(),
                "label_bin": torch.tensor(label_bin, dtype=torch.long),
            }
        )
    return processed_samples


# Dataset definition
class FastVulnerabilityDataset(Dataset):
    def __init__(self, processed_data, pad_id):
        self.data = processed_data
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        final_labels = item["labels_ids"].clone()
        final_labels[final_labels == self.pad_id] = -100
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "label_bin": item["label_bin"],
        }


# Model architecture
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_cwe_classes, hidden_size, dropout_rate):
        super().__init__()
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.t5.config.use_cache = True

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2),
        )
        self.cwe_head = nn.Linear(hidden_size, num_cwe_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(
                encoder_outputs.last_hidden_state * mask_expanded, 1
            )
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask

            logits_binary = self.classifier_head(pooled_output)
            logits_cwe = self.cwe_head(pooled_output)

        return logits_binary, logits_cwe


# Evaluation and fusion logic
def semantic_aware_decision_fusion(
    model, loader, device, lambda_fuse, threshold, num_samples, tok_latency_ms
):
    model.eval()
    y_true, y_bin_probs, y_sem_conf = [], [], []

    print("[INFO] Executing GPU warmup...")
    dummy_ids = torch.zeros((1, 512), dtype=torch.long, device=device)
    dummy_mask = torch.ones((1, 512), dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(dummy_ids, dummy_mask)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("[INFO] Executing Semantic-Aware Decision Fusion (SADF) Pipeline...")

    start_event.record()
    with torch.no_grad():
        for batch in tqdm(loader, desc="[INFO] Inference", unit="batch"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits_bin, logits_cwe = model(ids, mask)

            probs_bin = torch.softmax(logits_bin, dim=1)[:, 1]
            probs_cwe = torch.sigmoid(logits_cwe)
            max_conf, _ = torch.max(probs_cwe, dim=1)

            y_true.extend(batch["label_bin"].cpu().numpy())
            y_bin_probs.extend(probs_bin.cpu().numpy())
            y_sem_conf.extend(max_conf.cpu().numpy())

    end_event.record()
    torch.cuda.synchronize()

    total_det_latency_ms = start_event.elapsed_time(end_event)
    avg_det_latency_ms = total_det_latency_ms / num_samples
    end_to_end_latency_ms = tok_latency_ms + avg_det_latency_ms

    y_true = np.array(y_true)
    fused_scores = (lambda_fuse * np.array(y_bin_probs)) + (
        (1 - lambda_fuse) * np.array(y_sem_conf)
    )
    preds = (fused_scores > threshold).astype(int)

    f1 = f1_score(y_true, preds, zero_division=0)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)

    print("\n" + "=" * 60)
    print(f"{'EVALUATION METRICS':^60}")
    print("=" * 60)
    print(f"F1-Score         : {f1:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"Precision        : {prec:.4f}")
    print(f"Accuracy         : {acc:.4f}")
    print("-" * 60)
    print(f"Total Samples    : {num_samples}")
    print(f"Tokenization (CPU)   : {tok_latency_ms:.2f} ms/sample")
    print(f"Detection (Encoder)  : {avg_det_latency_ms:.2f} ms/sample")
    print(f"Total Inference (E2E): {end_to_end_latency_ms:.2f} ms/sample")
    print("=" * 60 + "\n")


# Rationale generation logic
def generate_rationale_samples(
    model, loader, raw_samples, tokenizer, device, config_model, gen_config
):
    model.eval()
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    all_generated_texts = []
    all_preds = []
    all_labels = []
    num_samples = len(raw_samples)

    total_ttft_ms = 0.0
    total_gen_ms = 0.0
    num_batches = 0

    print("[INFO] Executing GPU warmup for generation...")
    dummy_ids = torch.zeros((1, 16), dtype=torch.long, device=device)
    dummy_mask = torch.ones((1, 16), dtype=torch.long, device=device)
    with torch.no_grad():
        _ = base_model.t5.generate(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            max_new_tokens=2,
            num_beams=gen_config.get("num_beams", 4),
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"[INFO] Generating rationales using Beam Search...")

    for batch in tqdm(loader, desc="[INFO] Generation", unit="batch"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label_bin"].to(device)

        ttft_tracker = TTFTProcessor()
        processors = LogitsProcessorList([ttft_tracker])

        try:
            start_event.record()
            gen_ids = base_model.t5.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=config_model["max_desc_len"],
                num_beams=gen_config["num_beams"],
                no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                repetition_penalty=gen_config["repetition_penalty"],
                length_penalty=gen_config["length_penalty"],
                early_stopping=True,
                logits_processor=processors,
            )
            end_event.record()
            torch.cuda.synchronize()

            if ttft_tracker.recorded:
                total_ttft_ms += start_event.elapsed_time(ttft_tracker.ttft_event)
            total_gen_ms += start_event.elapsed_time(end_event)
            num_batches += 1

            with torch.no_grad():
                logits_bin, _ = model(ids, mask)
                preds_bin = torch.argmax(logits_bin, dim=1)

            all_generated_texts.extend(
                tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            )
            all_preds.extend(preds_bin.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[FATAL] CUDA Out Of Memory during generation.")
                print(f"        Action Required: Decrease batch_size or num_beams")
                sys.exit(1)
            raise e

    avg_ttft_ms = total_ttft_ms / num_samples
    avg_gen_ms = total_gen_ms / num_samples

    print("\n" + "=" * 80)
    print(f"{'QUALITATIVE CASE STUDIES':^80}")
    print("=" * 80)

    for i in range(num_samples):
        ground_truth = "VULNERABLE" if all_labels[i] == 1 else "SAFE"
        prediction = "VULNERABLE" if all_preds[i] == 1 else "SAFE"
        status = "MATCH" if all_labels[i] == all_preds[i] else "MISMATCH"

        raw_code = raw_samples[i].get("text", "")
        display_code = raw_code[:400].strip() + (
            "\n[... TRUNCATED ...]" if len(raw_code) > 400 else ""
        )

        print(f"\nSAMPLE ID: {i + 1} | STATUS: {status}")
        print(f"Ground Truth : {ground_truth}")
        print(f"Prediction   : {prediction}")
        print("-" * 80)
        print("[Code Snippet]:")
        print(f"{display_code}")
        print("-" * 80)
        print("[Generated Rationale]:")
        print(f"{all_generated_texts[i]}")
        print("=" * 80)

    print(f"\n[METRICS] XAI Generation Latency Breakdown (Amortized):")
    print(f"          Time To First Token (TTFT) : {avg_ttft_ms:.2f} ms/sample")
    print(f"          Total Generation Time      : {avg_gen_ms:.2f} ms/sample\n")


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOR-Vul Inference Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["reveal", "devign"],
        required=True,
        help="Target dataset config name",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained .bin weights"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to the test JSON file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to pipeline configuration",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Inference batch size"
    )
    parser.add_argument(
        "--show_rationales",
        type=int,
        default=0,
        help="If > 0, skips evaluation and generates N rationales",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config_full = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Configuration file '{args.config}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON syntax in '{args.config}': {str(e)}")
        sys.exit(1)

    try:
        sys_config = config_full["system"]
        model_config = config_full["model"]
        gen_config = config_full["generation"]
        data_config = config_full["datasets"][args.dataset]
    except KeyError as e:
        print(f"[ERROR] Missing required configuration key: {str(e)}")
        sys.exit(1)

    device = torch.device(
        sys_config["device_preference"] if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Initialized System Device: {device}")

    if not args.test_data.endswith(".json"):
        print("[ERROR] Strict Policy: Only JSON format is supported.")
        sys.exit(1)

    print(f"[INFO] Loading {args.dataset.upper()} Dataset...")
    try:
        with open(args.test_data, "r") as f:
            test_raw = json.load(f)
            if isinstance(test_raw, dict) and "val" in test_raw:
                test_raw = list(test_raw["val"].values())
            elif isinstance(test_raw, dict):
                test_raw = list(test_raw.values())
    except FileNotFoundError:
        print(f"[ERROR] Dataset file '{args.test_data}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to parse dataset '{args.test_data}': {str(e)}")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["backbone"], trust_remote_code=True
        )
    except Exception as e:
        print(
            f"[ERROR] Failed to load tokenizer '{model_config['backbone']}': {str(e)}"
        )
        sys.exit(1)

    print("[INFO] Constructing Computational Graph...")
    try:
        model = MultiTaskModel(
            model_name=model_config["backbone"],
            num_cwe_classes=data_config["num_cwe_classes"],
            hidden_size=model_config["hidden_size"],
            dropout_rate=model_config["classifier_dropout"],
        )
    except Exception as e:
        print(f"[ERROR] Failed to construct model: {str(e)}")
        sys.exit(1)

    print("[INFO] Loading Model Weights...")
    try:
        state = torch.load(args.model_path, map_location=device, weights_only=False)
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.to(device)
    except FileNotFoundError:
        print(f"[ERROR] Weights file '{args.model_path}' not found.")
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                "[FATAL] CUDA Out Of Memory during model loading. Check GPU utilization."
            )
            sys.exit(1)
        print(
            f"[ERROR] Failed to map weights to model. Architecture mismatch? {str(e)}"
        )
        sys.exit(1)

    if args.show_rationales > 0:
        print(
            f"[INFO] XAI Mode Activated: Selecting {args.show_rationales} balanced samples. Bypassing global evaluation."
        )
        vuln_samples = [x for x in test_raw if int(x.get("label", 0)) == 1]
        safe_samples = [x for x in test_raw if int(x.get("label", 0)) == 0]

        vuln_count = min(
            args.show_rationales // 2 + (args.show_rationales % 2), len(vuln_samples)
        )
        safe_count = args.show_rationales - vuln_count

        random_samples = random.sample(vuln_samples, vuln_count) + random.sample(
            safe_samples, safe_count
        )
        random.shuffle(random_samples)

        test_proc = preprocess_data_offline(
            random_samples, tokenizer, model_config, show_progress=False
        )
        gen_batch_size = min(8, args.batch_size)
        test_loader = DataLoader(
            FastVulnerabilityDataset(test_proc, tokenizer.pad_token_id),
            batch_size=gen_batch_size,
            shuffle=False,
        )

        generate_rationale_samples(
            model=model,
            loader=test_loader,
            raw_samples=random_samples,
            tokenizer=tokenizer,
            device=device,
            config_model=model_config,
            gen_config=gen_config,
        )
    else:
        print("[INFO] Pre-processing Evaluation Data...")
        t0 = time.perf_counter()
        test_proc = preprocess_data_offline(
            test_raw, tokenizer, model_config, show_progress=True
        )
        t1 = time.perf_counter()
        tok_latency_ms = ((t1 - t0) * 1000) / len(test_raw)

        test_loader = DataLoader(
            FastVulnerabilityDataset(test_proc, tokenizer.pad_token_id),
            batch_size=args.batch_size,
            shuffle=False,
        )

        semantic_aware_decision_fusion(
            model=model,
            loader=test_loader,
            device=device,
            lambda_fuse=data_config["lambda_fuse"],
            threshold=data_config["optimal_threshold"],
            num_samples=len(test_raw),
            tok_latency_ms=tok_latency_ms,
        )
