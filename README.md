# Heart Disease Prediction using Machine Learning

[![CI Tests](https://github.com/22054204/Heart-Disease-Prediction-using-AI-ML-/actions/workflows/tests.yml/badge.svg)](https://github.com/22054204/Heart-Disease-Prediction-using-AI-ML-/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Ranjan%20Sharma-brightgreen.svg)](#author)

An end-to-end machine learning project for predicting the presence of coronary heart disease using clinical patient attributes from the UCI Heart Disease (Cleveland) dataset.

This repository features both exploratory research notebooks and a reproducible, production-grade Python package with CLI tools for model training, evaluation, and inference.

---

> [!IMPORTANT]
> **Clinical and Educational Disclaimer**  
> This project is designed strictly for academic, research, and educational purposes. It is **not** a certified medical diagnostic device and must **not** be used as a substitute for professional clinical judgment, diagnosis, or treatment planning.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Performance & Evaluation](#model-performance--evaluation)
- [Dataset Details](#dataset-details)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [CLI Usage](#cli-usage)
  - [1. Model Training](#1-model-training)
  - [2. Model Prediction](#2-model-prediction)
- [Running Automated Tests](#running-automated-tests)
- [Jupyter Notebook Exploration](#jupyter-notebook-exploration)
- [Author](#author)
- [License](#license)

---

## Overview

Cardiovascular diseases (CVDs) remain the leading cause of mortality globally. Early and reliable identification of high-risk patients enables timely medical intervention and personalized lifestyle changes. 

This project implements machine learning models that analyze routine non-invasive clinical diagnostic features (e.g., resting blood pressure, cholesterol, resting ECG, maximum heart rate achieved, ST depression) to assess heart disease probability.

---

## Key Features

- **Leakage-Free Architecture**: Imputation and feature scaling are encapsulated inside Scikit-Learn `Pipeline` objects, fitted strictly on training folds to prevent data leakage.
- **Full Data Preservation**: Retains all 303 patient records from the Cleveland dataset; missing values are handled systematically rather than discarding rows.
- **Dual Model Architectures**: Supports both balanced **Logistic Regression** (linear, interpretable) and **Random Forest** (non-linear ensemble).
- **Production CLI Commands**: Fast, automated commands (`heart-disease-train` and `heart-disease-predict`) registered via `pyproject.toml`.
- **Comprehensive Clinical Metrics**: Reports Accuracy, Precision, Sensitivity/Recall, Specificity, F1-Score, Average Precision, and ROC-AUC.
- **CI/CD Integration**: Fully automated GitHub Actions workflow testing on Python 3.10, 3.11, and 3.12.

---

## Model Performance & Evaluation

In clinical decision support, optimizing for **Recall (Sensitivity)** and **ROC-AUC** is vital to minimize life-threatening false negatives (patients with disease mistakenly classified as healthy).

| Metric | Logistic Regression (Balanced) | Random Forest |
| :--- | :---: | :---: |
| **Accuracy** | **~86.9%** | ~85.2% |
| **Recall / Sensitivity** | **92.8%** | 89.3% |
| **ROC-AUC** | **95.0%** | 92.4% |
| **Precision** | 81.3% | 80.6% |
| **F1-Score** | **86.7%** | 84.7% |
| **Specificity** | 81.8% | 81.8% |

*Evaluated on a 20% stratified hold-out test set.*

---

## Dataset Details

The model is trained on the processed **Cleveland Clinic Heart Disease** database from the UCI Machine Learning Repository:

| Feature Name | Clinical Description | Value Range / Categories |
| :--- | :--- | :--- |
| `age` | Age in years | Integer (e.g., 29 - 77) |
| `sex` | Biological sex | `1` = Male, `0` = Female |
| `cp` | Chest pain type | `1` = Typical angina, `2` = Atypical angina, `3` = Non-anginal, `4` = Asymptomatic |
| `trestbps` | Resting blood pressure (on admission) | mm Hg (e.g., 94 - 200) |
| `chol` | Serum cholesterol | mg/dl (e.g., 126 - 564) |
| `fbs` | Fasting blood sugar > 120 mg/dl | `1` = True, `0` = False |
| `restecg` | Resting electrocardiographic results | `0` = Normal, `1` = ST-T wave abnormality, `2` = Left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved during exercise | Integer (e.g., 71 - 202) |
| `exang` | Exercise-induced angina | `1` = Yes, `0` = No |
| `oldpeak` | ST depression induced by exercise relative to rest | Float (e.g., 0.0 - 6.2) |
| `slope` | Slope of peak exercise ST segment | `1` = Upsloping, `2` = Flat, `3` = Downsloping |
| `ca` | Number of major vessels colored by flourosopy | `0` to `3` |
| `thal` | Thallium scintigraphy stress test | `3` = Normal, `6` = Fixed defect, `7` = Reversible defect |
| `target` | Diagnosis of heart disease | `0` = No disease (< 50% diameter narrowing), `1` = Diagnosed disease |

---

## Project Architecture

```
Raw CSV / UCI Data
       │
       ▼
load_cleveland_data() ──► Clean types & create binary target (0 or 1)
       │
       ▼
train_test_split() ────► Stratified train / test splits
       │
       ▼
Pipeline([
   ("imputer",    SimpleImputer(strategy="median")),
   ("scaler",     StandardScaler()),
   ("classifier", LogisticRegression / RandomForest)
])
       │
       ▼
evaluate_model() ──────► Confusion matrix & clinical metrics (ROC-AUC, Recall)
       │
       ▼
Artifacts Persisted ───► artifacts/heart_disease_model.joblib & artifacts/metrics.json
```

---

## Repository Structure

```text
├── .github/
│   └── workflows/
│       └── tests.yml               # Automated CI matrix tests (Python 3.10, 3.11, 3.12)
├── src/
│   └── heart_disease/
│       ├── __init__.py             # Package exports
│       ├── data.py                 # Robust data loading & schema validation
│       ├── model.py                # Pipeline construction & metrics computation
│       ├── train.py                # CLI script to train and save models
│       └── predict.py              # CLI script for validated model predictions
├── tests/
│   ├── test_data.py                # Dataset schema and target tests
│   ├── test_model.py               # Pipeline training, evaluation & persistence tests
│   └── test_predict.py             # Feature parsing and CLI validation tests
├── heart_disease_prediction.ipynb  # Interactive Jupyter / Colab analysis notebook
├── processed.cleveland.data        # UCI processed Cleveland data file
├── pyproject.toml                  # Modern Python project configuration & CLI entry points
├── requirements.txt                # Core dependencies
├── REPORT FOR HEART DISEASE PREDICTION.pdf  # Project technical documentation report
└── README.md                       # Main project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/22054204/Heart-Disease-Prediction-using-AI-ML-.git
cd Heart-Disease-Prediction-using-AI-ML-
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Package and Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

---

## CLI Usage

### 1. Model Training

Train the default **Logistic Regression** model:
```bash
heart-disease-train --data processed.cleveland.data
```

Train a **Random Forest** classifier:
```bash
heart-disease-train --data processed.cleveland.data --model-type random_forest
```

#### Optional Training Arguments:
- `--data`: Path to dataset file (default: `processed.cleveland.data`)
- `--output-dir`: Output folder for model and metrics (default: `artifacts`)
- `--model-type`: `logistic_regression` or `random_forest` (default: `logistic_regression`)
- `--test-size`: Holdout test fraction (default: `0.2`)
- `--random-state`: Random seed for reproducibility (default: `42`)

This outputs `artifacts/heart_disease_model.joblib` and `artifacts/metrics.json`.

---

### 2. Model Prediction

Run inference by providing the 13 clinical features in comma-separated order:
```bash
heart-disease-predict --features "63,1,3,145,233,1,0,150,0,2.3,0,0,1"
```

**Output:**
```json
{
  "prediction": 0,
  "diagnosis": "No Diagnosed Disease (Low Probability)",
  "probability_of_diagnosed_disease": 0.0348,
  "threshold": 0.5,
  "notice": "Educational model only; not medical advice or a diagnostic tool."
}
```

Example for a high-risk patient:
```bash
heart-disease-predict --features "67,1,4,160,286,0,2,108,1,1.5,2,3,3"
```

**Output:**
```json
{
  "prediction": 1,
  "diagnosis": "Diagnosed Heart Disease (High Probability)",
  "probability_of_diagnosed_disease": 0.997,
  "threshold": 0.5,
  "notice": "Educational model only; not medical advice or a diagnostic tool."
}
```

---

## Running Automated Tests

Run the full automated test suite with `pytest`:

```bash
pytest -v
```

All unit and integration tests covering dataset ingestion, model pipelines, evaluation metrics, and prediction validation will execute.

---

## Jupyter Notebook Exploration

The repository includes `heart_disease_prediction.ipynb` for interactive data analysis, correlation heatmaps, feature importance graphs, and algorithm benchmarking.

You can launch it locally:
```bash
jupyter notebook heart_disease_prediction.ipynb
```
or open it directly in [Google Colab](https://colab.research.google.com/). The notebook automatically locates the dataset whether run locally or inside Colab.

---

## Author

**Ranjan Sharma**  
- Project: Heart Disease Prediction using AI/ML  
- GitHub: [@22054204](https://github.com/22054204)

---

## License

This project is licensed under the [MIT License](LICENSE).
