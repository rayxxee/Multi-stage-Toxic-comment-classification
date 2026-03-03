# Toxic Comment Classifier - Assignment 2

> **Assignment 2 Submission**

## Project Overview

This project implements an end-to-end Machine Learning pipeline to detect toxicity in Social Media comments and posts. This submission heavily emphasizes rigorous evaluation, error analysis, reproducibility, and robust software engineering practices.

It utilizes a hierarchical pipeline composed of multiple models to ensure both speed and accuracy:

**NOTE:** since the models are 800mb in size i have uploaded them on drive you can download them from there and put them in the models folder you can also train them from scratch using the raw data

[Download Models](https://drive.google.com/file/d/1p2ombFd2cckU1RQBflyMC-sjYkAXwOdd/view?usp=sharing)

1. **Gatekeeper (Logistic Regression + TF-IDF)**: Acts as a fast first-pass filter. It classifies comments as safe or unsafe. We have updated the threshold to **0.9**—meaning if the Gatekeeper predicts a comment is safe with a high confidence ($P \ge 0.9$), the comment passes and exits the pipeline early.
2. **FastText Classifier**: Comments that are not flagged as confidently safe by the Gatekeeper ($P < 0.9$) are routed to the FastText model for a secondary, more robust evaluation. FastText handles sub-words, which is great for misspellings and unseen slang.
3. **DeBERTa Model**: *(Configured & mapped for future integration or deeper analytical experiments).*

### Alignment with Assignment Rubric

- **At least two models or configurations**: We implemented **Gatekeeper (LR)**, **FastText**, and an integrated **Pipeline Model**.
- **Evaluation using multiple metrics**: The project leverages `scikit-learn` to calculate Precision, Recall, F1 (Macro & Weighted), and ROC-AUC. It uses **Matplotlib/Seaborn** to generate visualizations for all models locally in the `plots/` folder.
- **Error Analysis**: Misclassified CSVs (`gatekeeper_errors.csv` and `pipeline_errors.csv`) are logged for detailed error transparency. The `report.md` dives into the analytical reasoning behind these metrics.
- **Model Serialization**: All models are natively dumped and loaded utilizing `joblib` inside the `models/` directory for immediate inference without retraining.
- **Reproducibility Controls**: We utilize `config/config.yaml` to enforce strict central mapping of hyper-parameters, data splittings, and global Python/NumPy fixed seeds (`set_seed(42)`).
- **Unit Tests**: Full unit testing is configured using **pytest**. You can verify the integrity of each isolated stage via the `test_models/` folder.

---

## 💻 Setup & Usage Instructions

### 1. Installation & Environment Setup

It's highly recommended to use the generated virtual environment.

Install the required dependencies via pip:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Place your `train.csv` file into the `data/raw/` directory.

### 3. Training the Models (Serialization Step)

Before running the evaluation pipeline, ensure your serialized models are generated:

```bash
python train_models/train_gatekeeper.py
python train_models/train_fasttext.py
# python train_models/train_deberta.py (optional, can skip for fast tests)
```

The resulting files (`gatekeeper.joblib`, `fasttext.joblib`) will populate the `models/` directory.

### 4. Running the Main Evaluation Pipeline

To execute the hierarchical pipeline (Clean -> Label -> Split -> Evaluate):

```bash
python main.py
```

- This script dynamically routes data.
- It produces rich visual curves (`plots/` folder) such as `*_roc.png`, `*_pr.png`, and `*_cm.png`.
- Evaluates individual standalone models before benchmarking the unified setup.
- Detailed stdout is also saved effectively in `pipeline.log`.

### 5. Running the Unit Tests

Execute `pytest` simply by calling:

```bash
pytest test_models/
```

This will confirm the determinism and architectural validity of the independent model structures.

---

## 📁 Project Structure

```text
assignment/
├── config/
│   └── config.yaml          # Central configuration, seeds, model params, and threshold setups
├── data/
│   ├── raw/                 # Input data (train.csv)
│   └── processed/           # Processed & cleaned CSV files
├── models/                  # Output directory containing `.joblib` serialized instances
├── plots/                   # Generated evaluation plots (ROC, CM, PR) and CSV error subsets
├── src/
│   ├── data/                # Data handlers (cleaner.py, labeller.py, splitter.py)
│   ├── models/              # Core model classes (gatekeeper_lr.py, fasttext_model.py, etc.)
│   ├── evaluation/          # Metrics aggregation & Model reporting logic
│   └── pipeline/            # Hierarchical execution classes
├── train_models/            # Individual training scripts to dump `joblib` artifacts
├── test_models/             # Unit tests for verification via Pytest
├── main.py                  # Single-entry execution script
├── report.md                # Extensive reflection on Trade-Offs and Analytical Reasoning 
└── requirements.txt         # Required Python packages
```
