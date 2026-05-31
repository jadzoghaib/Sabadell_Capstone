# Prompt Variance Analysis — `02_Prompt_Variance.ipynb`

Branch: `prompt-variance`  
Model: `meta/llama-3.3-70b-instruct` via NVIDIA NIM  
Dataset: LendingClub 2012–2014 credit risk sample (100 loans, binary: Fully Paid / Charged Off)

---

## What this is

The model selection notebooks (01a/01b/01c) established *which LLM* predicts credit risk best. This notebook asks a different question: **given a fixed model, how much does the prompt design matter?**

Six prompt variants are tested through the same three-phase evaluation pipeline used for model selection — comparison, consistency, and robustness — so the results are directly comparable in methodology.

---

## Prompt variants

| # | Name | What changes |
|---|------|--------------|
| 0 | `baseline` | Current production prompt — control group |
| 1 | `conservative` | Role reframed as risk-averse underwriter; biased toward flagging defaults when uncertain |
| 2 | `chain_of_thought` | Explicit step-by-step reasoning through risk signals before predicting |
| 3 | `few_shot` | 4 labeled training examples prepended to each user prompt (requires raw CSV) |
| 4 | `top_features_only` | Only the 8 most XGBoost-important features passed in (int_rate, sub_grade, dti, …) |
| 5 | `structured_4factor` | Explicit 4-factor evaluation framework in the system prompt |

> `few_shot` is skipped automatically if the raw LendingClub CSV is unavailable, since it needs training examples. The other 5 variants always run.

---

## Three-phase evaluation

### Phase 1 — Comparison
All variants run on the same 100-loan evaluation sample. Primary metric: **Charged Off F1** (minority class, most relevant for credit risk). The best-performing variant is selected as the winner.

### Phase 2 — Consistency
The winning variant runs **3 times** on the same sample. Measures:
- Per-run metric variance (accuracy, F1, AUC)
- Per-loan flip rate (how often individual predictions change between runs)

A well-designed prompt should be stable — high flip rate is a red flag even if mean performance looks good.

### Phase 3 — Robustness
The winning variant runs once on a **new held-out batch** of 100 loans (different from the evaluation sample). Measures how much performance drops when the distribution shifts slightly.

---

## How to run

**Requirements:** Python environment with `openai`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `python-dotenv`.  
The NVIDIA NIM API key must be set in `.env` at the repo root or in `notebooks/llm_models/.env`:

```
NVIDIA_API_KEY=nvapi-...
```

**Execute the notebook:**

```bash
# From repo root, using the correct conda/micromamba environment
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=<your_env> \
  --inplace notebooks/llm_models/prompt_variance/02_Prompt_Variance.ipynb
```

Each phase makes ~100 API calls, so the full run takes roughly 15–30 minutes depending on NVIDIA NIM throughput.

---

## Outputs

All results land in `data/results/llm/`:

| File | Contents |
|------|----------|
| `02_phase1_metrics.csv` | Accuracy, AUC, Charged Off F1 for every variant |
| `02_phase2_consistency.csv` | Per-run metrics for the winning variant (3 runs) |
| `02_phase3_robustness.csv` | Original sample vs new batch performance for the winner |
| `02_predictions.csv` | Per-loan predictions, actuals, and correctness across all phases |
| `02_reasonings.jsonl` | Full LLM reasoning text per loan per variant — input for promptfoo |

Call log (tokens, cost, latency per call) is appended to `data/results/llm/llm_calls.csv` automatically by `llm_utils.py`.

---

## Qualitative characterisation with promptfoo

After the notebook runs, `promptfoo/prepare_tests.py` samples 10 reasonings per variant from `02_reasonings.jsonl` and writes `promptfoo/tests.yaml`. Then `npx promptfoo@latest eval` uses an LLM judge to write a 4–6 sentence qualitative characterisation of each variant's *reasoning fingerprint* — what features it anchors on, its risk posture, and its blind spots.

```bash
# From repo root
python promptfoo/prepare_tests.py
npx promptfoo@latest eval --config promptfoo/promptfooconfig.yaml
```

Results: `promptfoo/results/qualitative_characterisations.json`

---

## Key design decisions

**Why Charged Off F1 as the primary metric?**  
Charged Off is the minority class (~20% of loans). A model that predicts "Fully Paid" for everything gets 80% accuracy but is useless for risk management. F1 on the minority class captures both precision and recall for defaults.

**Why the same 100-loan sample across Phase 1 and Phase 2?**  
Controlling the sample isolates prompt effects from data variance. Phase 3 then deliberately introduces a new sample to test whether those effects hold out-of-sample.

**Why not compare against the XGBoost baseline here?**  
The classical ML comparison is covered in the model selection notebooks (01a/01b/01c). This notebook's scope is prompt engineering, not model selection — adding the XGBoost baseline here would require running `03_Modeling.ipynb` first and conflates two separate questions.
