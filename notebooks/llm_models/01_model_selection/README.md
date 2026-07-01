# Phase 1 — Model Selection, Consistency & Calibration

This folder contains the experiments conducted during **Phase 1** to identify the most performant base LLM and its optimal operational parameters for credit default prediction.

---

## Notebooks & Modules

### 1. Model Comparison (`01a_Model_Comparison.ipynb`)
* **Objective:** Run a head-to-head comparison across foundation models (`gpt-5.4`, `gemini-2.5-pro`, `gemini-3.5-flash`, `claude-sonnet-4.6`, `claude-opus-4.8`) on a 100-loan credit risk sample under two distinct conditions:
  * **Structured Features Only (`no_desc`):** Apples-to-apples comparison with the traditional XGBoost model using only numeric/categorical credit scores.
  * **Structured Features + Narrative (`with_desc`):** Appending unstructured qualitative borrower descriptions (`desc` column).
* **Outputs:** `01a_predictions.csv` and `01a_metrics.csv`.

### 2. Prediction Consistency (`01b_Consistency.ipynb`)
* **Objective:** Quantify LLM prediction variance by running the winning model (`gpt-5.4`) three identical times over the same 100 loans. Calculates per-loan stability scores and metric deviation.
* **Outputs:** `01b_predictions.csv` and `01b_metrics.csv`.

### 3. Out-of-Sample Robustness (`01c_Robustness.ipynb`)
* **Objective:** Test model generalization capabilities by evaluating `gpt-5.4` on a completely new, out-of-sample batch of 100 loans with a higher baseline default rate.
* **Outputs:** `01c_predictions.csv` and `01c_metrics.csv`.

### 4. Reasoning Effort Analysis (`01d_reasoning_effort_runs.ipynb` / `01d_Reasoning_Effort_Analysis.md`)
* **Objective:** Sweep GPT-5.4's `reasoning_effort` parameter (low → high) to map the cost/accuracy curve and pick the effort that maximizes AUC while remaining financially viable. The chosen effort is carried into the Phase-4 benchmark.
* **Watch-out:** higher effort's best *single* run (0.80 acc / 0.375 CO-F1) does **not** survive the consistency check — its 3-run mean regresses to 0.753 / 0.300. Judge by the multi-run mean, not a lucky single run.
* **Outputs:** `01d_predictions.csv` and `01d_metrics.csv`.

### 5. Confidence Calibration (`01e_Confidence_Calibration.ipynb`)
* **Objective:** Meta-analysis checking if LLMs are "overconfident" when predicting defaults.
* **Focus:** Extracts raw logprobs of the output token to calculate Expected Calibration Error (ECE), Brier scores, reliability diagrams, and maps confidence vs. stability and correctness.
* **Outputs:** `01e_confidence_metrics.csv` and evaluation plots.

### 6. Phase 1 Qualitative & Financial Analysis Gate (`01f_Qualitative_Financial_Analysis.ipynb`)
* **Objective:** First qualitative "gate". Spends API credits to run a `gpt-5.4` judge that extracts a qualitative "reasoning fingerprint" for each model/effort combination. Builds a comprehensive credit portfolio ledger showing baseline financial yields.
* **Outputs:** `01f_qualitative_financial.json`.

---

## Phase 1 Conclusions

1. **GPT-5.4 Superiority:** GPT-5.4 emerges as the most stable and performant base model, achieving a **0.412 F1** in the `with_desc` condition.
2. **Narrative Impact:** Appending qualitative borrower text (`desc`) significantly improves GPT-5.4's predictive accuracy but degrades both Gemini models, highlighting a stark difference in how models handle unstructured qualitative data.
3. **Consistency:** GPT-5.4 exhibits very high stability across identical runs (std dev < 1.15%), validating that single runs provide dependable metrics for this model family.
