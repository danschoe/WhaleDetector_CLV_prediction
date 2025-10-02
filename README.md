# Customer Lifetime Value Prediction using CNN

PyTorch implementation of CNN-based Customer Lifetime Value (LTV) prediction, adapted from "Customer lifetime value in video games using deep learning and parametric models" (Chen et al. 2018) on video game player value prediction for retail transaction data.

## Overview

Predicts customer lifetime value using convolutional neural networks on sequential transaction data from the H&M Personalized Fashion Recommendations dataset from Kaggle. The model learns temporal patterns from purchase histories to forecast future spending without feature engineering.

## Installation
Install Python 3.11+ and then install the dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

Place the H&M dataset files in `./data/`:
- `transactions_polars.parquet` - Transaction data

## Usage
```bash
python train.py
```

The script will:
1. Load and preprocess transaction data
2. Train the CNN model with early stopping
3. Evaluate on test set
4. Save predictions to `./data/predictions.csv`

## Model Architecture
```
Input (1, 360) → Conv1D(32, k=7) → ReLU → MaxPool(2) → 
Conv1D(16, k=3) → ReLU → Conv1D(1, k=1) → ReLU → Flatten → 
FC(300) → ReLU → FC(150) → ReLU → FC(60) → ReLU → Output(1)
```

## Project Structure
```
├── configs/config.yaml          # Configuration
├── data_processing/             # Data loading and preprocessing
├── modules/                     # Model and training logic
└── train.py                     # Main training script
```

## Citation
```bibtex
@article{chen2018customer,
  title={Customer lifetime value in video games using deep learning and parametric models},
  author={Chen, Pei Pei and Guitart, Anna and Fern{\'a}ndez del R{\'\i}o, Ana and Peri{\'a}{\~n}ez, {\'A}frica},
  journal={arXiv preprint arXiv:1811.12799},
  year={2018}
}
```