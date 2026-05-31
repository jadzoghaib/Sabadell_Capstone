# Classical Machine Learning Pipeline

This directory contains the traditional machine learning workflow, which serves as the predictive baseline and explainability contrast for the LLM engineering experiments.

---

## Notebook Structure

The pipeline is split into three sequential notebooks:

### 1. Exploratory Data Analysis (`01_EDA.ipynb`)
* **Objective:** Conduct initial data profiling and target relationship visualizations on the anonymised LendingClub credit risk dataset.
* **Key Findings:**
  * Highly imbalanced target classes (~20% `Charged Off` / default, ~80% `Fully Paid`).
  * Analysis of key high-impact feature correlations: interest rates, credit grades, and Debt-to-Income (DTI) metrics.
  * Identification of borrower written descriptions (`desc`) as a rich source of unstructured qualitative context.

### 2. Feature Preprocessing (`02_Preprocessing.ipynb`)
* **Objective:** Perform robust feature engineering, scaling, and data splits to export clean tensors for model consumption.
* **Key Steps:**
  * Missing value imputations and categorical encoding (converting grades, employment durations, and states into numeric columns).
  * Aligning column schemas to ensure consistency across classical ML and LLM inputs (35 features).
  * Fit and export normalization parameters (`02_scaler.joblib`, `02_feature_columns.joblib`) to avoid feature leakages.
  * Splits dataset into deterministic samples (`tuning_sample.csv`, `robustness_batch.csv`, and `test_batch.csv`).

### 3. Baseline Modeling (`03_Modeling.ipynb`)
* **Objective:** Train, optimize, and benchmark three classical classifier families:
  1. **Logistic Regression (LR):** Standard linear risk baseline.
  2. **Artificial Neural Network (ANN):** Multi-layer dense Keras network.
  3. **XGBoost Classifier:** Tree-based ensemble, hyperparameter tuned with Optuna.
* **Key Outputs:**
  * Winning model serialized to `models/xgb_model.joblib`.
  * Baseline thresholds exported to `models/thresholds.joblib`.
  * Summary metrics written to `data/results/ml/03_model_performance.csv`.

---

## Baseline Performance Reference

| Model | Accuracy | Charged Off Precision | Charged Off Recall | Charged Off F1 | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 71.0% | 29.0% | 52.0% | 0.370 | 0.699 |
| **XGBoost (Tuned)** | 71.0% | 29.0% | 56.3% | 0.383 | 0.705 |

> [!NOTE]  
> The classical ML results indicate a distinct "performance ceiling" when relying strictly on structured credit attributes. The XGBoost baseline F1 of **0.383** serves as the primary benchmark that the LLM prompting and hybrid ensembling strategies aim to exceed.
