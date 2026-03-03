# TinyH2GNet: Exploring Lightweight Models for Gene Expression Prediction from H&E Images

This repository contains the implementation for predicting spatial gene expression from histopathology images using convolutional and transformer-based models. The focus of this work is to analyze the trade-off between model size, prediction performance, and inference time.

## Overview

Spatial Transcriptomics (ST) enables measurement of gene expression with spatial context but remains costly and resource-intensive. This project investigates whether compact deep learning models can achieve competitive performance for image-based spatial gene expression prediction, enabling deployment in resource-constrained settings.

## Datasets

Experiments are conducted on three public spatial transcriptomics datasets:

- **Hist2ST (Breast Cancer, Visium)**
  https://data.mendeley.com/datasets/29ntw7sh4r/5

- **HER2 (Breast Cancer, Visium)**
  https://github.com/almaan/her2st

- **10x Xenium (Human Breast Cancer)**
  https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast

Please download the datasets from the official sources before running preprocessing.

## Data Preprocessing

All datasets are converted into a unified format:

- **.h5 files**
  Contain extracted image patches corresponding to spatial spots or cells.

- **.h5ad files**
  Contain gene expression matrices, spatial coordinates, and metadata.

Preprocessing includes:

- Patch extraction from whole-slide images
- Gene filtering (HVG selection on VISIUM Dataset)
- Log-normalization
- Scaling using configurable `scale_factor`
- Patient-wise metadata organization

Processed data is required before model training.

## Models

We benchmark the following models:

- STNet
- DeepSpaCE
- HisToGene
- TCGN
- THItoGene
- EfficientNet (full and parameter-scaled variants)
- TinyH2GNet (EfficientNet-based)

EfficientNet backbones are scaled using compound scaling to construct lightweight variants:

- 3%
- 5%
- 10%
- 20%
- 100% (full model)

These variants are designed to study accuracy–efficiency trade-offs.

## Training Configuration

- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 128
- Maximum epochs: 200
- Loss function: Mean Squared Error (MSE)
- Weight decay: 1e-5
- Random seed: 123

### Early Stopping
- Monitored on validation Spearman correlation
- Patience: 30 epochs

### Learning Rate Scheduler
- Patience: 5 epochs
- Decay factor: 0.1
- Minimum learning rate: 1e-8

All experiments use patient-level cross-validation to prevent data leakage.

## Evaluation Metrics

We report:

- L1 error
- L2 error
- Pearson correlation (PCC)
- Spearman correlation
- MAPE (%)
- Parameter count
- Inference time

Both accuracy and runtime are analyzed.

## Hardware

### CPU Experiments

- 4 physical cores (8 logical threads)
- 32 GB RAM
- No GPU acceleration

### GPU Experiments

- NVIDIA A100 (40 GB)

All models are implemented in:

- Python 3.10
- PyTorch

## Running the Code

### Training
```bash
python main.py --config parameters.json
```

### Evaluation
```bash
python evaluation/evaluation_visium.py
```

### CPU-only Inference
```bash
python evaluation/evaluation_visium.py --device cpu
```

## Repository Structure

```
code_model_training/
│
├── models/                 # Model definitions
├── utils/                  # Data utilities
├── evaluation/             # Evaluation scripts
├── parameters.json         # Experiment configuration
└── modelResults/           # Saved models and outputs
```

## Reproducibility

- Fixed random seed (123)
- Identical patient-wise splits across models
- All hyperparameters specified in configuration files

Results reproducible with provided scripts.