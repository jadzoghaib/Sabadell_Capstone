# Sabadell Capstone — Classical ML vs. LLM for Credit Scoring

An ESADE Capstone Project (2026) sponsored by **Banc Sabadell – IT & OPs**.  
**Authors:** Alessandro Mezzanotte, Jad Zoghaib, and Francesc Canavate Quero.

---

## Project Overview

This repository contains the complete codebase, data pipelines, and offline experimental simulations for the Banc Sabadell credit risk project. 

The core objective is to compare a **traditional Machine Learning baseline (Gradient Boosting / XGBoost)** against an **LLM "reasoned decision" engine** (evaluated across `gpt-5.4`, `gemini-2.5-pro`, `gemini-3.5-flash`, `claude-sonnet-4.6`, `claude-opus-4.8`, and Llama-3.3-70b) for credit default prediction (binary classification: `Fully Paid` vs. `Charged Off`).

### Core Focus Areas
1. **Predictive Rigour:** Evaluating standard metrics (AUC, F1, Recall, Precision) on the minority class (`Charged Off`).
2. **Operational Economics:** Quantifying and integrating OpenAI automatic prompt caching economics (piecewise 50% discount on static prefixes exceeding 1,024 tokens), formatting/batching taxes, and API cost-benefit thresholds.
3. **Explainability & Trust:** Contrasting classical model explainability (SHAP feature importances) with LLM natural language rationales and qualitative reasoning "fingerprints" judged by `gpt-5.4`.
4. **Hybrid Architectures:** Evaluating a blended, confidence-gated routing system that pipelines low-uncertainty applications through XGBoost and routes high-uncertainty boundary cases to Llama-3.3-70b (via Groq).
5. **Retrieval-Augmented Scoring (RAG):** Injecting *precedent loans* retrieved from a leakage-safe corpus as evidence, comparing Semantic-ID, dense-kNN, and hybrid (RRF) retrievers against a no-RAG control.

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
│   ├── ml_models/              # Traditional ML pipeline (EDA → preprocessing → modeling → analysis)
│   └── llm_models/             # LLM evaluations (Phases 1-5)
│       ├── 01_model_selection/ # Base model evaluation, consistency, effort, and calibration
│       ├── 02_prompt_variance/ # Prompt design, batching/formatting taxes, caching economics
│       ├── 03_hybrid/          # Blended ML + LLM ensembling and gated routing
│       ├── 04_final_benchmark/ # Held-out final benchmark, SHAP, and decision P&L
│       └── 05_rag/             # Retrieval-augmented scoring (Semantic-ID / dense / hybrid)
├── CLAUDE.md                   # Developer handbook, run commands, and context ledger
├── requirements.txt            # System dependencies
└── README.md                   # This master documentation
```

---

## Setup & Installation

### 1. Prerequisites
The project runs on **Python 3.11** (from the `sabadell` conda env). Clone the repository and install the dependencies:
```bash
conda create -n sabadell python=3.11 -y
conda activate sabadell
pip install -r requirements.txt
python -m ipykernel install --user --name sabadell --display-name "Python (sabadell)"
```
Then select the **Python (sabadell)** kernel for every notebook.

### 2. Environment Configurations
LLM pipelines require API keys. Create a `.env` file inside `notebooks/llm_models/` (gitignored) with the keys you need:
```env
OPENAI_API_KEY=...            # Phase 1, 2, 4, 5
GEMINI_API_KEY_PRO=...        # Phase 1 (Gemini 2.5 Pro, via Vertex AI)
GEMINI_API_KEY_FLASH=...      # Phase 1 (Gemini 3.5 Flash)
ANTHROPIC_API_KEY=...         # Phase 1 (Claude Sonnet 4.6 / Opus 4.8)
NVIDIA_API_KEY=...            # Phase 2 (Llama-3.3-70b via NVIDIA NIM)
GROQ_API_KEY=...              # Phase 3 (Llama-3.3-70b via Groq)
GCP_PROJECT_ID=capstonesabadell   # Vertex AI project for Gemini logprobs
GCP_LOCATION=europe-west4         # Vertex AI region
```
See [`CLAUDE.md`](CLAUDE.md) for the full key/provider matrix and spare-key conventions.

---

## Pipeline Execution Summary

1. **Traditional ML:** Run notebooks under `notebooks/ml_models/` in order (`01_EDA` → `02_Preprocessing` → `03_Modeling` → `04_Model_Analysis`). This trains the classical classifiers, exports `xgb_model.joblib` + preprocessing scalers, and produces the XGBoost SHAP analysis.
2. **LLM Evaluation:** Run numbered folders inside `notebooks/llm_models/`. Because `models/` and the data samples are tracked, the LLM notebooks can run without re-running the ML pipeline:
   - **Phase 1:** Model selection, reasoning-effort sweep, calibration curves, and first financial ledger (`01_model_selection/`).
   - **Phase 2:** System optimizations (batching and compact representations) and prompt-framing variants (`02_prompt_variance/`).
   - **Phase 3:** Groq-powered blending and gated routing strategies (`03_hybrid/`).
   - **Phase 5:** Retrieval-augmented scoring with precedent loans, evaluated on the validation batch (`05_rag/`).
   - **Phase 4 (run last):** Conclusive benchmark on the locked test set, SHAP-vs-LLM-rationale explainability, and case-by-case decision P&L (`04_final_benchmark/`).

> [!NOTE]
> **Strict test holdout:** `data/processed/test_batch.csv` is loaded only by Phase 4. Phases 1–3 and 5 use `tuning_sample.csv` / `robustness_batch.csv`.
