# Phase 2 — Prompt Structuring, Batching & Variance

Branch: `promptvar-gpt5.4`  
Model: `gpt-5.4` (and legacy Llama-3.3-70b comparison)  
Dataset: LendingClub 2012–2014 credit risk sample (100 loans, binary: Fully Paid / Charged Off)

This folder contains two notebooks exploring how prompt engineering and prompt execution architectures impact LLM credit risk predictions:
1. `02a_Batching_Formatting_Tax.ipynb` — System optimization analysis (Formatting & Batching).
2. `02b_Prompt_Variance.ipynb` — Linguistic optimization analysis (System prompts & Personas).

---

## 02a — Batching & Formatting Tax (`02a_Batching_Formatting_Tax.ipynb`)

This notebook isolates and quantifies the predictive and operational impacts of prompt batching and compact TOON formatting using `gpt-5.4`. It compares four controlled conditions over 3 runs each to establish a statistical baseline for the "formatting tax" and "batching tax":

* **Condition 1 (Control):** Individual API calls (1 per loan) using natural language key-value formats. (Reused from `01b_Consistency`).
* **Condition 2 (Tooning Only):** Individual API calls (1 per loan) using the compact pipe-delimited TOON format.
* **Condition 3 (Batching Only):** Batched API calls (all 100 in 1 call) using verbose natural language keys repeated for each loan.
* **Condition 4 (Both together):** Batched API calls (all 100 in 1 call) using the compact pipe-delimited TOON format.

### Key Outputs
* `data/results/llm/02a_tax_metrics.csv` — Accuracy, F1, Recall, Token Counts, Cost, and Latency for all 4 conditions.
* `data/results/llm/02a_tax_comparison.png` — Visual breakdown of the predictive and cost deltas.

---

## 02b — Prompt Variance (`02b_Prompt_Variance.ipynb`)

Given a fixed model (`gpt-5.4`), how much does the semantic framing of the prompt matter? This notebook tests 6 distinct prompt variants through the standard 3-phase evaluation rigour (Comparison, Consistency, and Robustness):

| # | Name | What changes |
|---|------|--------------|
| 0 | `baseline` | Current production prompt — control group |
| 1 | `conservative` | Role reframed as risk-averse underwriter; biased toward flagging defaults when uncertain |
| 2 | `chain_of_thought` | Explicit step-by-step reasoning through risk signals before predicting |
| 3 | `few_shot` | 4 labeled training examples prepended to each user prompt (requires raw CSV) |
| 4 | `top_features_only` | Only the 8 most XGBoost-important features passed in (int_rate, sub_grade, dti, …) |
| 5 | `structured_4factor` | Explicit 4-factor evaluation framework in the system prompt |

### Key Outputs
* `data/results/llm/02b_phase1_metrics.csv` — Accuracy, AUC, Charged Off F1 for every variant.
* `data/results/llm/02b_phase2_consistency.csv` — Per-run metrics for the winning variant (3 runs).
* `data/results/llm/02b_phase3_robustness.csv` — Original sample vs new batch performance for the winner.
* `data/results/llm/02b_predictions.csv` — Per-loan predictions, actuals, and correctness across all phases.
* `data/results/llm/02b_reasonings.jsonl` — Full LLM reasoning text per loan per variant.
* `data/results/llm/02b_qualitative.json` — Judge-derived qualitative characterisations of each variant's reasoning fingerprint.

---

## Running the Notebooks

Ensure the OpenAI API key is configured in `notebooks/llm_models/.env`:
```env
OPENAI_API_KEY=sk-...
```

Run notebooks using `jupyter` or compile them via `nbconvert`:
```bash
# To run the Batching & Formatting Tax Notebook (02a)
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --inplace notebooks/llm_models/02_prompt_variance/02a_Batching_Formatting_Tax.ipynb

# To run the Prompt Variance Notebook (02b)
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --inplace notebooks/llm_models/02_prompt_variance/02b_Prompt_Variance.ipynb
```
