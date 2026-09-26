# DOR-Vul

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neucom.2026.135164-blue?logo=elsevier)](https://doi.org/10.1016/j.neucom.2026.135164)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/csmuni/DOR-Vul/)
<br>
[![Release](https://img.shields.io/github/v/release/csmuni/DOR-Vul?logo=github&label=Release)](https://github.com/csmuni/DOR-Vul/releases)
[![Kaggle](https://img.shields.io/badge/Kaggle-Models-20BEFF?logo=kaggle)](https://kaggle.com/models/vafaeii/dor-vul/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface)](https://huggingface.co/vafaeim/DOR-Vul)
<br>
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

PyTorch implementation of the DOR-Vul vulnerability detection framework. 

This repository contains the inference code, evaluation datasets, and scripts required to reproduce the quantitative metrics and XAI rationales on the Devign and Reveal benchmarks.

## Quick Start

Automated end-to-end evaluation environments are available for one-click reproducibility:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sMNTGPI5bC33CYapK-LYRhmffr5_RUNO?usp=sharing)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/vafaeii/dor-vul-reproduction-notebook)

## Repository Structure

```text
.
├── config.json                 # Inference configurations
├── data/                       # Test datasets (JSON format)
├── inference.py                # Main evaluation and generation script
├── utils/
│   └── decontaminate_train_set.py # Dataset preprocessing and deduplication script
└── README.md

```

## Pre-trained Weights

Pre-trained model weights are hosted publicly across multiple platforms:

* **Hugging Face:** [vafaeim/DOR-Vul](https://huggingface.co/vafaeim/DOR-Vul)
* **Kaggle Models:** [DOR-Vul](https://www.kaggle.com/models/vafaeii/dor-vul)
* **GitHub Releases:** [v1.0](https://github.com/csmuni/DOR-Vul/releases/tag/v1.0)

## Local Usage

### Requirements

```bash
pip install torch protobuf==3.20.3 sentencepiece accelerate "transformers<4.40" scikit-learn tqdm

```

### 1. Evaluation

To run the standard evaluation on a dataset (e.g., Reveal):

```bash
python inference.py \
    --dataset reveal \
    --model_path <path_to_reveal_weight>.pt \
    --test_data data/reveal_test_split.json \
    --batch_size 32  # Set to 1 for sequential latency profiling

```

### 2. XAI Generation

To bypass global evaluation and generate vulnerability rationales for specific samples:

```bash
python inference.py \
    --dataset devign \
    --model_path <path_to_devign_weight>.pt \
    --test_data data/devign_test_split.json \
    --show_rationales 8

```

## License

Apache 2.0

## Citation

If you find this repository useful in your research, please consider citing our work. The BibTeX entry will be updated with the DOI upon final publication:

```bibtex
@article{vafaei2027dorvul,
  title   = {DOR-vul: Cross-modal reasoning distillation for efficient vulnerability recognition},
  author  = {Amirreza {Vafaei Moghadam} and Davar Giveki},
  journal = {Neurocomputing},
  volume  = {707},
  pages   = {135164},
  year    = {2027},
  issn    = {0925-2312},
  doi     = {10.1016/j.neucom.2026.135164}
}

```
