# Phase 2 — Prompt Structuring, Batching & Variance

Model: `gpt-5.4` (and legacy Llama-3.3-70b comparison)  
Dataset: LendingClub 2012–2014 credit risk sample (100 loans, binary: Fully Paid / Charged Off)

This folder contains three notebooks exploring how prompt engineering, compact formats, and prompt execution architectures impact credit risk predictions and operational costs:
1. `02a_Batching_Formatting_Tax.ipynb` — System optimization analysis (Formatting & Batching).
2. `02b_Prompt_Variance.ipynb` — Linguistic optimization analysis (System prompts & Personas).
3. `02c_Qualitative_Financial_Analysis.ipynb` — Phase 2 qualitative and financial analysis gate.

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
* `02a_Batching_Formatting_Tax_Analysis.md` — Supplemental markdown summarizing key optimization findings.

---

## 02b — Prompt Variance (`02b_Prompt_Variance.ipynb`)

Given a fixed model (`gpt-5.4`), how much does the semantic framing of the prompt matter? This notebook tests 7 distinct prompt variants through the standard 3-phase evaluation rigour (Comparison, Consistency, and Robustness):

| # | Name | Description / Framing |
|---|------|--------------|
| 0 | `baseline` | Current production prompt — control group. |
| 1 | `conservative` | Role reframed as risk-averse underwriter; biased toward flagging defaults when uncertain. |
| 2 | `chain_of_thought` | Explicit step-by-step reasoning through risk signals before predicting. |
| 3 | `few_shot` | 4 labeled training examples prepended to each user prompt. |
| 4 | `top_features_only` | Only the 8 most XGBoost-important features passed in (`int_rate`, `sub_grade`, `dti`, etc.). |
| 5 | `structured_4factor` | Explicit 4-factor evaluation framework in the system prompt. |
| 6 | `risk_signal_guide` | Incorporates detailed risk flags and credit rating guidelines. |

### Key Outputs
* `data/results/llm/02b_phase1_metrics.csv` — Accuracy, AUC, Charged Off F1 for every variant.
* `data/results/llm/02b_phase2_consistency.csv` — Per-run metrics for the winning variant (3 runs).
* `data/results/llm/02b_phase3_robustness.csv` — Original sample vs. new batch performance for the winner.
* `data/results/llm/02b_predictions.csv` — Per-loan predictions, actuals, and correctness.
* `data/results/llm/02b_reasonings.jsonl` — Full LLM reasoning text per loan per variant.
* `data/results/llm/02b_variant_comparison.png` — Visual chart of variant metrics.

> ⚠️ **Key finding — read the consistency phase, not just the comparison.** It is tempting to crown `chain_of_thought` from `02b_phase1_metrics.csv`, where it has the top single-run accuracy (0.83 vs the base prompt's 0.81). That is a single-run artefact. On Charged-Off F1 the base prompt already wins there (0.387 vs 0.370), and under the Phase-2 consistency check CoT regresses to a 3-run mean of **0.312**, below the base prompt's **0.327**. **The base prompt is the winner; no engineered variant reliably beats it.** Always cross-check `02b_phase2_consistency.csv` before declaring a prompt winner (same trap as high `reasoning_effort` in 01d).

---

## 02c — Qualitative & Financial Analysis Gate (`02c_Qualitative_Financial_Analysis.ipynb`)

This notebook serves as the Phase 2 analysis gate, evaluating prompt caching economics, business trade-offs, and reasoning fingerprints:
* **Cost Simulation:** Models OpenAI's automatic prompt caching economics (piecewise 50% discount on static prefix input tokens, triggered only when the total prompt length exceeds 1,024 tokens).
* **Qualitative Fingerprinting:** Runs a `gpt-5.4` judge that profiles and categorizes each prompt variant's credit reasoning style based on their full written rationales.
* **Outputs:** `02c_qualitative_financial.json`.

---

## 02d — Prompt-Variant Model Analysis (`02d_Model_Analysis.ipynb`)

The Phase-2 counterpart to `01g`: the same feature-reliance and error-analysis lens, but the axis is prompt *design* with GPT-5.4 held fixed. For each of the 6 variants it fits a global surrogate to read which features that variant's decisions track, and tallies the portfolio error cost. Reads `02b_predictions.csv` — **no API calls**, and `test_batch` is never touched.

---

## Running the Notebooks

Ensure your OpenAI API key is configured in `notebooks/llm_models/.env`:
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

# To run the Qualitative Financial Analysis Gate (02c)
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --inplace notebooks/llm_models/02_prompt_variance/02c_Qualitative_Financial_Analysis.ipynb
```
