# Classical Machine Learning Pipeline

This directory contains the traditional machine learning workflow, which serves as the predictive baseline and explainability contrast for the LLM engineering experiments.

---

## Notebook Structure

The pipeline is split into four sequential notebooks:

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
  * Winning model serialized to `models/xgb_model.joblib` (LR and ANN also saved).
  * Baseline thresholds exported to `models/thresholds.joblib`.
  * Summary metrics written to `data/results/ml/03_model_performance.csv`.

### 4. Model Analysis & Explainability (`04_Model_Analysis.ipynb`)
* **Objective:** Deep-dive on the trained XGBoost model — the classical half of the explainability comparison required by the project brief.
* **Sections:** performance, calibration, **SHAP** feature attribution, a permutation-importance cross-check on feature reliance, and error analysis.

---

## Baseline Performance Reference

From `data/results/ml/03_model_performance.csv` (33% held-out split, tuned thresholds):

| Model | Accuracy | Charged Off Precision | Charged Off Recall | Charged Off F1 | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 68.5% | 29.1% | 59.3% | 0.390 | 0.708 |
| **XGBoost (Tuned)** | 69.0% | 29.5% | 59.1% | **0.393** | **0.712** |
| **ANN (Keras)** | 67.4% | 28.6% | 61.3% | 0.390 | 0.709 |

> [!NOTE]  
> The classical ML results indicate a distinct "performance ceiling" when relying strictly on structured credit attributes — all three families land within ~0.005 F1 of each other. The XGBoost baseline F1 of **0.393** serves as the primary benchmark that the LLM prompting and hybrid ensembling strategies aim to exceed.
>
> The final head-to-head against the LLMs happens on the separate 1000-loan held-out `test_batch.csv` in Phase 4 — see [`04_final_benchmark/README.md`](../llm_models/04_final_benchmark/README.md).
