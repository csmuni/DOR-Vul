# DOR-Vul

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

Pre-trained model weights are hosted publicly:

* **Kaggle Models:** [DOR-Vul](https://www.kaggle.com/models/vafaeii/dor-vul/)
* **GitHub Releases:** [v1.0](https://github.com/vafaeim/DOR-Vul/releases/tag/v1.0)

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
    --batch_size 32

```

### 2. XAI Generation

To generate vulnerability rationales for specific samples:

```bash
python inference.py \
    --dataset devign \
    --model_path <path_to_devign_weight>.pt \
    --test_data data/devign_test_split.json \
    --show_rationales 4

```

## License

Apache 2.0

## Citation

If you find this repository useful in your research, please consider citing our work. The BibTeX entry will be updated with the DOI upon publication:

```bibtex
@article{dorvul2026,
  title={DOR-Vul: Cross-Modal Reasoning Distillation for Efficient Vulnerability Recognition},
  author={Vafaei Moghadam, Amirreza and Giveki, Davar},
  journal={Under Review},
  year={2026}
}

```
