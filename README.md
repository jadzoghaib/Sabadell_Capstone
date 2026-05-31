# Sabadell Capstone — Classical ML vs. LLM for Credit Scoring

An ESADE Capstone Project (2026) sponsored by **Banc Sabadell – IT & OPs**.  
**Authors:** Alessandro Mezzanotte, Jad Zoghaib, and Francesc (Cesc) Xavier.

---

## Project Overview

This repository contains the complete codebase, data pipelines, and offline experimental simulations for the Banc Sabadell credit risk project. 

The core objective is to compare a **traditional Machine Learning baseline (Gradient Boosting / XGBoost)** against an **LLM "reasoned decision" engine** (evaluated across `gpt-5.4`, `gemini-3.5-flash`, `claude-3.5-sonnet`, and others) for credit default prediction (binary classification: `Fully Paid` vs. `Charged Off`).

### Core Focus Areas
1. **Predictive Rigour:** Evaluating standard metrics (AUC, F1, Recall, Precision) on the minority class (`Charged Off`).
2. **Operational Economics:** Quantifying and integrating OpenAI automatic prompt caching economics (piecewise 50% discount on static prefixes exceeding 1,024 tokens), formatting/batching taxes, and API cost-benefit thresholds.
3. **Explainability & Trust:** Contrasting classical model explainability (SHAP feature importances) with LLM natural language rationales and qualitative reasoning "fingerprints" judged by `gpt-5.4`.
4. **Hybrid Architectures:** Evaluating a blended, confidence-gated routing system that pipelines low-uncertainty applications through XGBoost and routes high-uncertainty boundary cases to Llama-3.3-70b (via Groq) / GPT-5.4.

---

## Regulatory and Scope Constraints

> [!IMPORTANT]  
> **Offline Prototyping Constraint:** This repository represents an offline simulation sandbox and qualitative prototyping engine. There is **no production deployment path** or live connection to Banc Sabadell transaction lines, complying with bank safety guidelines.
>
> All experiments are conducted on an anonymised and de-identified subset of public LendingClub credit files (2012–2014, ~5,000 loans).

---

## Repository Structure

```
Sabadell_Capstone/
├── data/                       # Experimental datasets and model outputs
│   ├── raw/                    # Large source LendingClub dumps (gitignored)
│   ├── processed/              # Deterministic samples (tuning, robustness, test sets)
│   └── results/                # CSV tables, JSON records, PNG charts of all runs
│       ├── ml/                 # Classical ML metrics and model rankings
│       └── llm/                # Phased LLM predictions, reasonings, and pricing logs
├── models/                     # Frozen model checkpoints (XGBoost, LR, Keras ANN, thresholds)
├── notebooks/                  # Experimental Jupyter Notebooks
│   ├── ml_models/              # Traditional Machine Learning pipeline (Phases 1-3)
│   └── llm_models/             # Advanced LLM evaluations (Phases 1-4)
│       ├── 01_model_selection/ # Base model evaluation, consistency, effort, and calibration
│       ├── 02_prompt_variance/ # Prompt design, batching/formatting taxes, caching economics
│       ├── 03_hybrid/          # Blended ML + LLM ensembling and gated routing
│       └── 04_final_benchmark/ # Held-out final benchmark and financial simulations
├── CLAUDE.md                   # Developer handbook, run commands, and context ledger
├── requirements.txt            # System dependencies
└── README.md                   # This master documentation
```

---

## Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Configurations
LLM pipelines require API keys. Create a `.env` file inside `notebooks/llm_models/`:
```bash
cp notebooks/llm_models/.env.example notebooks/llm_models/.env
# Edit notebooks/llm_models/.env to add your keys:
# - OPENAI_API_KEY
# - GEMINI_API_KEY
# - GROQ_API_KEY
```

---

## Pipeline Execution Summary

1. **Traditional ML:** Run notebooks under `notebooks/ml_models/` in order (`01_EDA` → `02_Preprocessing` → `03_Modeling`). This trains the classical classifiers and exports `xgb_model.joblib` and preprocessing scalers.
2. **LLM Evaluation:** Run numbered folders inside `notebooks/llm_models/`:
   - **Phase 1:** Model selection, effort parameters, calibration curves, and first financial ledger (`01_model_selection/`).
   - **Phase 2:** System optimizations (batching and compact representations) and linguistic framing variants (`02_prompt_variance/`).
   - **Phase 3:** Groq-powered blending and gated routing strategies (`03_hybrid/`).
   - **Phase 4:** Conclusive benchmark on the locked test set and interactive executive yield dashboards (`04_final_benchmark/`).
