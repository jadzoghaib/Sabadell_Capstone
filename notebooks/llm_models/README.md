# LLM Engineering & Evaluation Pipeline

This directory houses the phased LLM experimentation, evaluation frameworks, and utility tools developed to benchmark Generative AI performance against traditional credit scoring models.

---

## Shared Utility Modules

To ensure absolute experimental reproducibility, all LLM notebooks draw from central helper modules:

* **`llm_utils.py`:** The core operational engine. Handles data loading, ML feature re-encoding, pricing metrics calculation, token count validation, API calls (wrapping OpenAI, Gemini, Anthropic, and NVIDIA), prediction probability extraction via token logprobs, and automatic append-only logging to `data/results/llm/llm_calls.csv`.
* **`llm_pricing.py`:** The pricing catalog. Contains the official price list (USD per 1,000 input/output tokens) for all evaluated model families.
* **`sample_generation.py`:** Generates reproducible, mutually-exclusive datasets from LendingClub splits (`tuning_sample`, `robustness_batch`, `test_batch`), preserving target class ratios.

---

## Phased Pipeline Flow

```mermaid
graph TD
    A[ML Data Splits] --> B[Phase 1: Model Selection <br> 01_model_selection]
    B -->|Pick GPT-5.4| C[Phase 2: Prompt Structuring & Variance <br> 02_prompt_variance]
    C -->|Prompts & Caching| D[Phase 3: Hybrid Blending <br> 03_hybrid]
    C -->|Precedent Retrieval| F[Phase 5: RAG <br> 05_rag]
    D -->|Locked Ensemble| E[Phase 4: Final Benchmark & Simulation <br> 04_final_benchmark]
    F -.validation only.-> E
```

> Phases 3 and 5 both feed insight into Phase 4 but evaluate only on the validation batches;
> Phase 4 is the only stage that touches `test_batch.csv`.

### [Phase 1: Model Selection & Calibration](file:///Users/alemz/Projects/Github/Sabadell_Capstone/notebooks/llm_models/01_model_selection)
* **Goal:** Evaluate foundation models across multiple providers to pick the ideal platform and configuration.
* **Focus:** Provider comparisons (Gemini vs. Claude vs. GPT), description-impact studies (`±desc`), reasoning-effort sweeps, and confidence/calibration checks (Brier scores & ECE).
* **Gate:** Executive cost ledgers and qual-analysis fingerprints (`01f`).

### [Phase 2: Prompt Structuring & Variance](file:///Users/alemz/Projects/Github/Sabadell_Capstone/notebooks/llm_models/02_prompt_variance)
* **Goal:** Optimize prompting structure and minimize operational costs.
* **Focus:** Isolating formatting taxes (TOON syntax) and batching taxes, linguistic system prompt testing (7 variants), and OpenAI automatic prompt caching economics (50% input discount on prompts > 1,024 tokens).
* **Gate:** Prompt-variance cost ledgers and qual-analysis fingerprints (`02c`).

### [Phase 3: Hybrid Blended LLM/ML Approach](file:///Users/alemz/Projects/Github/Sabadell_Capstone/notebooks/llm_models/03_hybrid)
* **Goal:** Pipeline applications through a combination of classical ML (XGBoost) and high-reasoning LLMs.
* **Focus:** Soft-probability ensembling, confidence-gated routing, and GPT-5.4 execution. The test set remains strictly untouched here.

### [Phase 5: Retrieval-Augmented (RAG) Scoring](file:///Users/alemz/Projects/Github/Sabadell_Capstone/notebooks/llm_models/05_rag)
* **Goal:** Test whether injecting *precedent loans* as evidence beats judging each applicant in isolation.
* **Focus:** Three retrievers — Semantic-ID + multi-stage (`05a`), full-corpus dense kNN (`05b`), and RRF hybrid (`05c`) — each against a shared no-RAG GPT-5.4 control and XGBoost, on a leakage-safe corpus. **Evaluated on `robustness_batch` (validation) — the test set is never loaded here.**

### [Phase 4: Final Benchmark & Explainability](file:///Users/alemz/Projects/Github/Sabadell_Capstone/notebooks/llm_models/04_final_benchmark)
* **Goal:** Establish final comparative benchmarks and translate predictive performance into commercial business metrics. **Run last.**
* **Focus:** A single notebook, `04_Final_Test_Analysis.ipynb` — runs the finalist prompts × GPT-5.4 on the locked `test_batch.csv`, applies the Phase-3 hybrid params post-hoc, tunes the decision threshold, computes a case-by-case decision P&L on real LendingClub cashflows, and contrasts XGBoost SHAP against LLM rationales.
