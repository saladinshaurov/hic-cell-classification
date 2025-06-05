# Hi-C Cell Type Classification

This project uses machine learning to classify single-cell types based on contact maps derived from Hi-C data.

## 📁 Structure

- `src/`: Feature extraction, modeling, and utility scripts
- `notebooks/`: Jupyter notebooks for EDA and testing
- `data/`: Raw or preprocessed Hi-C contact files (not tracked by Git)
- `results/`: Plots, metrics, and exported feature tables

## Features Extracted

- Total/intra/inter contacts
- Diagonal decay
- Distance-contact correlation
- Chromosome-wise contact summaries

## Models

- KNN, SVM
- sklearn-compatible workflow

## 📦 Requirements

Install packages:

```bash
pip install -r requirements.txt