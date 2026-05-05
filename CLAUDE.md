# Sabadell Capstone

ESADE Capstone Project 2026 sponsored by **Banc Sabadell – IT & OPs**, by
students **Alessandro, Jad, and Cesc**. Title: *Classical ML vs. LLM for
Credit Scoring (offline simulation + explainability)*. Final presentation is
at Banc Sabadell HQ in Sant Cugat del Vallès.

## Project brief (from `P280 - Classical ML vs. LLM for Credit Scoring - Banc Sabadell.pdf`)

- Compare a **traditional ML model (e.g. Gradient Boosting)** against an LLM
  "reasoned decision" produced via prompting on **anonymised or synthetic
  credit files**.
- Required metrics: **AUC, F1, stability, and cost**. Plus a side-by-side of
  **explainability approaches: SHAP for the ML model vs. LLM rationales.**
- **Regulatory frame:** offline prototyping and risk/explainability analysis
  only — *no production use*. This is a hard constraint from the sponsor; do
  not propose deployment, real customer data, or anything that crosses into
  production paths.

### Timeline (2026)

| When  | Milestone |
| ----- | --------- |
| Feb   | Kick-off, data setup & access |
| Mar   | 1st follow-up — open-source models + project proposal |
| Apr   | 2nd follow-up — project progress (this is what `reports/Progress report 1.pdf` was for) |
| Jun   | 3rd follow-up — **main-goal final presentation** |
| Jul   | Final meeting — **stretch-goal final presentation** at Sabadell HQ |

### Scope gaps to be aware of

The brief asks for things the current code does not yet do — flag these when
relevant rather than assuming they're done:

- **AUC**: infra is in place but the experiments need a re-run to produce
  numbers. `call_llm(..., with_logprobs=True)` requests top-K token logprobs
  from OpenAI / Gemini and `_extract_prediction_prob` parses them into a
  normalised `P("1") / (P("1") + P("0"))` per call. `run_llm_experiment` now
  defaults `with_logprobs=True`, persists the per-call probability to
  `llm_calls.csv` (column `prob_fully_paid`), and `evaluate_predictions`
  reports AUC at run time when probabilities are passed. **Anthropic does
  not expose logprobs** — those calls record `NaN` and AUC for that provider
  is `n/a`. For ML, AUC is trivial since the saved models already do
  `predict_proba`; just compute `roc_auc_score(y_true, probs)` from the
  saved test split. *Re-running the 04a/04b/04c notebooks is what produces
  the actual AUC values for the report.*
- **Per-call LLM log**: implemented. Every successful run of
  `llm_utils.run_llm_experiment` appends per-call rows to
  `data/results/llm/llm_calls.csv` with columns: `timestamp, experiment_id,
  label, desc_tag, provider, model, row_index, input_tokens, output_tokens,
  input_price_per_1k_usd, output_price_per_1k_usd, cost_usd, prob_fully_paid`.
  Prices come from `notebooks/llm_models/llm_pricing.py` — update that file
  when providers change rates; historical rows already on disk keep whatever
  price was logged at call time. `prob_fully_paid` is the per-call
  P(prediction=1) extracted from token logprobs (OpenAI/Gemini); Anthropic
  rows record `NaN`. **Interrupted runs (KeyboardInterrupt or exceptions)
  drop their buffer and never write**, so `llm_calls.csv` only reflects
  fully-completed experiments. To compute AUC later, join `llm_calls.csv` to
  the per-experiment results CSV on `(experiment_id, row_index)` for the
  ground-truth labels.
- **SHAP** is not in any notebook. The XGBoost model in `models/xgb_model.joblib`
  is SHAP-friendly; the explainability comparison is the natural place to add it.
- **Anonymised/synthetic data**: the brief mentions this; the team is using
  the public LendingClub dataset, which is already de-identified — fine for
  offline prototyping, but worth noting if the sponsor asks.

## Project context

Loan-default prediction on the LendingClub "accepted" dataset (2012–2014, ~5k loans, binary
`Fully Paid` vs `Charged Off`), used as a testbed to compare **traditional ML** against
**LLM-based classification** on the same task. `Charged Off` is the minority
class — overall accuracy is misleading on its own, so the metrics that matter
are precision / recall / F1 (and AUC, once added) on the `Charged Off` class.

Two parallel pipelines:

1. **ML pipeline** — `notebooks/01_EDA.ipynb` → `02_Preprocessing.ipynb` → `03_Modeling.ipynb`
   (Logistic Regression, XGBoost tuned with Optuna, Keras ANN).
2. **LLM evaluation** — `notebooks/llm_models/` is split into two phases:
   `model_selection/` (concluded — picked GPT-5 over Gemini Pro/Flash by comparing
   accuracy, consistency, and robustness ±desc) and `optimization/` (the active
   phase — improving GPT-5 specifically on the **no-desc / structured-features
   only** condition, per the supervisor's redirect).

## Where things stand (Apr 2026, from `reports/Progress report 1.pdf`)

- **XGBoost (structured features only):** 71% accuracy, Charged Off
  precision 29% / recall 56.3% / F1 0.383. Other public Kaggle notebooks on
  the same dataset land in the same ballpark, so the team's working hypothesis
  is that **the structured features aren't strong enough signal for ML to push
  much further** — pursuing big ML gains is probably not the highest-leverage
  thing to work on.
- **LLMs on the same 100-loan sample (best per model):**
  GPT-5 + desc **80% / F1 0.412** (best overall) > GPT-5 no-desc 73% > XGBoost 71%
  > Gemini Flash no-desc 61% > Gemini Pro no-desc 58%. Adding `desc` helps GPT-5
  but *hurts* both Gemini variants — the borrower description is not uniformly
  useful across models.
- **Consistency** (GPT-5, 3 runs × same 100 loans): mean 78.3% no-desc / 79.3%
  desc, std ~1.15%; per-loan stability 92% no-desc, 93% desc. GPT-5 is stable
  enough that reporting a single run is defensible.
- **Robustness** (held-out 100-loan batch — harder, 23 Charged Off vs 16):
  every model drops a bit, but the **ranking holds**: GPT-5 desc 76% > GPT-5
  no-desc 72% > XGBoost 70%. The desc → +accuracy effect for GPT-5 shrinks
  on the harder batch (−4pp vs original) but doesn't flip.

When proposing changes, weigh them against this picture. The Sabadell
supervisor (post-meeting, May 2026) has redirected the work toward
**improving GPT-5 in the no-desc / structured-features-only condition** —
i.e. the apples-to-apples comparison against XGBoost. So the live question
is *how to push GPT-5 no-desc higher* (prompt engineering, few-shot
examples, calibration, chain-of-thought, etc.), not the desc-vs-no-desc
contrast and not further tuning of XGBoost. Do not propose new desc-side
experiments unless explicitly asked.

## Repo layout

```
data/
  raw/         accepted_2007_to_2018Q4.csv.gz, lending_club_loan_two.csv.zip   (gitignored)
  processed/   02_processed_data.npz, 02_llm_sample.csv, 02_scaler.joblib,
               02_feature_columns.joblib, 04c_new_batch_sample.csv             (gitignored)
  results/
    ml/        03_model_performance.csv                                        (gitignored)
    llm/       04a_predictions.csv, 04a_metrics.csv,
               04b_predictions.csv, 04b_metrics.csv,
               04c_predictions.csv, 04c_metrics.csv,
               llm_calls.csv (per-call log) + archive/ pilots                  (gitignored)
models/        xgb_model.joblib, lr_model.joblib, ann_model.keras,
               thresholds.joblib                                                (gitignored)
notebooks/
  01_EDA.ipynb
  02_Preprocessing.ipynb     # writes data/processed/02_*
  03_Modeling.ipynb          # writes models/* and data/results/ml/03_model_performance.csv
  llm_models/
    .env                     # API keys, gitignored — see "API keys" below
    llm_utils.py             # shared: data loading, ML re-encoding, prompts, API calls, eval, cost logging
    llm_pricing.py           # per-model USD/1k token prices used by the cost logger
    model_selection/         # PHASE 1 (concluded): pick the LLM
      00_Sample_New_Batch.ipynb     # builds data/processed/04c_new_batch_sample.csv
      04a_Model_Comparison.ipynb    # GPT-5 / Gemini Pro / Gemini Flash, ±desc → 04a_predictions.csv + 04a_metrics.csv
      04b_Consistency.ipynb         # GPT-5 only, 3 runs × 2 conditions → 04b_predictions.csv + 04b_metrics.csv
      04c_Robustness.ipynb          # GPT-5 on held-out batch → 04c_predictions.csv + 04c_metrics.csv
    optimization/            # PHASE 2 (active): improve GPT-5 no-desc
      (empty — populate with new experiments here)
    archive/                 # earlier pilots, kept for reference only
reports/       Progress report 1.pdf, output.png
```

`.keep` files preserve the empty directories under git; everything else under
`data/`, `models/`, and `notebooks/llm_models/.env` is excluded by `.gitignore`.

## Run order

ML pipeline must run first end-to-end on a fresh checkout before any LLM notebook,
because `04a/04b/04c` load `data/processed/02_llm_sample.csv` and (via `llm_utils.run_ml_on_sample`)
the saved scaler, feature columns, and XGBoost model:

```
01_EDA → 02_Preprocessing → 03_Modeling   [generates everything in data/processed/ and models/]
00_Sample_New_Batch                       [only needed before 04c]
04a_Model_Comparison, 04b_Consistency, 04c_Robustness   [independent, any order]
```

`02_Preprocessing.ipynb` does a 67/33 train/test split with `random_state=42`.
`00_Sample_New_Batch.ipynb` uses `random_state=99` and dedupes against
`02_llm_sample.csv` on `(loan_amnt, int_rate, annual_inc)`.

## API keys

`notebooks/llm_models/.env` (gitignored) holds:

```
GEMINI_API_KEY_FLASH=...
GEMINI_API_KEY_PRO=...
OPENAI_API_KEY=...
```

`llm_utils.load_api_key(api_provider, model)` reads this file. Gemini keys are
selected per-model (Flash vs Pro). Add `ANTHROPIC_API_KEY` here if running the
Anthropic branch — the code path exists in `llm_utils.call_llm` but isn't used
in current notebooks.

## Environment

- Python: **3.13** (project is run from the `pyclass` micromamba env at
  `~/micromamba/envs/pyclass/`).
- No lockfile is committed. `requirements.txt` lists the packages that the
  notebooks and `llm_utils.py` import; pin versions there if you need reproducibility.
- Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`,
  `tensorflow` (Keras), `joblib`, `matplotlib`, `seaborn`, `python-dotenv`,
  `google-genai`, `openai`, `anthropic`.

## Conventions / gotchas

- **Never commit data, models, or `.env`.** `.gitignore` already covers
  `*.csv`, `*.joblib`, `*.npz`, `*.keras`, etc., but staging with `git add -A`
  from a fresh clone with raw data present is still the easiest way to leak.
  Stage by path.
- **Path resolution**: `llm_utils.py` anchors `DATA_DIR`, `MODEL_DIR`,
  `RESULTS_DIR`, and `RAW_DATA_PATH` to `Path(__file__).resolve().parent.parent.parent`
  (the repo root), so any notebook under `notebooks/llm_models/**` can import
  `llm_utils` and read/write the right files regardless of nesting depth. The
  ML notebooks (`01/02/03`) under `notebooks/` still use plain relative paths
  (`../data/...`) — run them from their own directory.
- **Importing `llm_utils` from a subfolder**: notebooks living under
  `notebooks/llm_models/<subfolder>/` need a one-liner `import sys;
  sys.path.insert(0, "..")` before `from llm_utils import ...`. This is
  already in place in every `model_selection/` notebook; copy the pattern
  for new notebooks under `optimization/`.
- **`llm_utils.LLM_FEATURES`** is the canonical 21-feature list that LLMs see
  (excludes the target, `desc`, and identifiers). If you change features in
  `02_Preprocessing.ipynb`, update `LLM_FEATURES` and `FEATURE_DESCRIPTIONS`
  in `llm_utils.py` to match, otherwise `run_ml_on_sample` will misalign columns.
- **Retry logic** for LLM calls: `llm_utils.call_llm` retries 5× with
  exponential backoff on 503/429/`UNAVAILABLE`. If a run hangs, that's the
  loop — kill the cell rather than waiting it out.
- **GPT-5 model name**: the notebooks pass the literal string `gpt-5` to the
  OpenAI SDK. If that route 404s, check the model registry and update the
  string in the notebook (not in `llm_utils.py`).
- **Branch**: development happens on `alessandro`, integrated into `main` via
  PR. Don't push directly to `main`.
- **Commit attribution**: do **not** add a `Co-Authored-By: Claude …` trailer
  to commits. Commits should be attributed only to the human GitHub user
  (`Alessandro Mezzanotte`). When committing on this repo, omit the
  Claude co-author line entirely.

## Reports

`reports/Progress report 1.pdf` is the latest written-up summary of findings
(Apr 3 2026). There is no committed script that generates it; it's authored
ad-hoc. The headline numbers in this file's "Where things stand" section come
from that PDF — if you re-run experiments and the numbers shift, update both.
