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

- **AUC**: ML side already has it — `03_Modeling.ipynb` computes
  `roc_auc_test` per model and saves it in `data/results/ml/03_model_performance.csv`
  (current numbers: XGBoost 0.705, LR 0.699; ANN missing because the saved
  CSV is from a partial run). LLM side: infra is in place but experiments
  need a re-run to produce numbers. `call_llm(..., with_logprobs=True)`
  requests top-K token logprobs from OpenAI / Gemini and
  `_extract_prediction_prob` normalises `P("1") / (P("1") + P("0"))` per
  call. `run_llm_experiment` defaults `with_logprobs=True`, persists the
  per-call probability to `llm_calls.csv` (column `prob_fully_paid`), and
  `evaluate_predictions` reports AUC at run time when probabilities are
  passed. **Anthropic does not expose logprobs** — those calls record
  `NaN` and AUC for that provider is `n/a`. *Re-running the 01a/01b/01c
  notebooks is what produces the actual LLM AUC values; XGBoost AUC ≈ 0.705
  is the bar.*
- **Per-call LLM log**: implemented. Every successful run of
  `llm_utils.run_llm_experiment` appends per-call rows to
  `data/results/llm/llm_calls.csv` with columns: `timestamp, notebook_id,
  label, desc_tag, provider, model, row_index, input_tokens, output_tokens,
  input_price_per_1k_usd, output_price_per_1k_usd, cost_usd,
  prob_fully_paid, reasoning_effort`.
  `notebook_id` records the name of the subfolder from which the notebook was
  executed (e.g. `01_model_selection`, `02_prompt_variance`) so you can
  always trace which phase generated each row.
  Prices come from `notebooks/llm_models/llm_pricing.py` — update that file
  when providers change rates; historical rows already on disk keep whatever
  price was logged at call time. `prob_fully_paid` is the per-call
  P(prediction=1) extracted from token logprobs (OpenAI/Gemini); Anthropic
  rows record `NaN`. **Interrupted runs (KeyboardInterrupt or exceptions)
  drop their buffer and never write**, so `llm_calls.csv` only reflects
  fully-completed experiments.
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

1. **ML pipeline** — `notebooks/ml_models/01_EDA.ipynb` → `02_Preprocessing.ipynb` → `03_Modeling.ipynb`
   (Logistic Regression, XGBoost tuned with Optuna, Keras ANN).
2. **LLM evaluation** — `notebooks/llm_models/` runs as five numbered phases:
   - `01_model_selection/` — pick the model **and its config**: GPT-5.4 vs
     Gemini 2.5 Pro vs Gemini 3.5 Flash vs Claude Sonnet 4.6 / Opus 4.8
     by accuracy, consistency, robustness ±desc, a GPT reasoning-effort
     sweep, and a confidence/calibration meta-analysis. `01f` is the
     phase-1 **analysis gate** — a GPT-5.4 "judge" that fingerprints each
     model/effort's reasoning style plus a portfolio cost ledger.
   - `02_prompt_variance/` — prompt formatting, batching, and design:
     - `02a_Batching_Formatting_Tax.ipynb` — isolates and quantifies the "batching tax" and "formatting tax" on credit risk predictions using GPT-5.4.
     - `02b_Prompt_Variance.ipynb` — how much prompt *design* matters across 7 variants (currently GPT-5.4; older Llama-3.3-70b run archived under `data/results/llm/llama_02/`).
       **⚠️ Do not conclude chain-of-thought wins from `02b_phase1_metrics.csv` alone.** That single comparison run gives CoT the top *accuracy* (0.83 vs base 0.81), which fools quick readers — but the base prompt already beats CoT on **Charged-Off F1** even there (0.387 vs 0.370), and CoT then *regresses* under the Phase-2 consistency check (3-run mean 0.312 CO-F1, below the base prompt's 0.327). **The base prompt is the winner; no engineered variant reliably beats it.** Always cross-check `02b_phase2_consistency.csv` before ranking prompts — same single-run-then-regress trap as high `reasoning_effort` in 01d (see `01d_high_consistency_metrics.csv`: best single run 0.80/0.375 but 3-run mean 0.753/0.300).
     - `02c_Qualitative_Financial_Analysis.ipynb` — phase-2 **analysis gate** (mirror of `01f`): GPT-5.4 judge fingerprints each prompt variant + cost ledger. Needs `02b`'s outputs first.
   - `03_hybrid/` — Jad's **blended XGBoost + LLM** exploration using
     Llama-3.3-70b via **Groq** — soft-probability blend, confidence-gated
     routing, and 5A/5B risk scorers — testing whether the two models'
     complementary strengths beat either solo on Charged-Off F1. Tuned on
     `tuning_sample`, strategy-selected on `robustness_batch`. **The test
     set is strictly held out** — never loaded or evaluated here.
   - `04_final_benchmark/` — **the spine, run last** — finalist prompts ×
     GPT-5.4 at the chosen effort on the untouched test batch, tunes the
     decision threshold, reports metrics **alongside cost** vs the XGBoost
     baseline. Also loads the frozen hybrid strategy parameters from Phase 3
     (`03_locked_params.csv`) and applies them **post-hoc** to the test
     predictions — no extra API calls needed for the ensembled comparisons.
     Threshold tuning lives here and *only* here, on the actual finalists —
     the conclusive, presentable result.
   - `05_rag/` — Jad's **retrieval-augmented (RAG)** exploration: instead of
     judging each applicant in isolation, retrieve *precedent loans* from a
     leakage-safe corpus and inject them as evidence. Three retrievers — `05a`
     TIGER Semantic IDs + multi-stage (RAG-FLARKO), `05b` full-corpus dense
     kNN, `05c` RRF hybrid of A+B — each vs a no-RAG LLM and the XGBoost
     baseline. Defaults to NVIDIA Llama-3.3-70B + sentence-transformers
     (TF-IDF fallback). **Evaluated on `robustness_batch` (validation), NOT the
     test set** — the held-out `test_batch` stays a Phase-4-only holdout. The
     retrieval corpus (`data/processed/rag_corpus.csv`, built by
     `rag_utils.build_rag_corpus`) is the full 2012-2014 frame minus every eval
     batch; `rag_utils.assert_no_leakage` enforces corpus ∩ eval = ∅ every run.
     The committed corpus is the ~100-row dev fallback (`tuning_sample`, no raw
     `.gz`); drop the raw file in and rerun `00` with `force=True` for the real
     large corpus.

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
data/   # tracked for collab so teammates can pull results without re-running
  raw/         accepted_2007_to_2018Q4.csv.gz, lending_club_loan_two.csv.zip   (gitignored — huge source dumps)
  processed/   02_processed_data.npz                                          (gitignored — 109MB, over GitHub's limit)
               tuning_sample.csv, robustness_batch.csv, test_batch.csv,       (tracked)
               02_scaler.joblib, 02_feature_columns.joblib                    (the 3 samples
               share one 35-col schema; see "Samples" below)
               rag_corpus.csv                                                 (tracked — Phase 5
               RAG retrieval corpus; committed file is the ~100-row dev fallback)
  results/
    ml/        03_model_performance.csv                                        (tracked)
    llm/       01a_*.csv (model comparison), 01b_*.csv (consistency),
               01c_*.csv (robustness), 01d_*.csv (reasoning effort),
               01e_confidence_metrics.csv + 01e_*.png (calibration),
               01f_qualitative_financial.json (phase-1 judge fingerprints),
               02a_tax_*.csv/.png (batching/formatting tax),
               02b_*.csv/.jsonl/.png (prompt variance),
               02c_qualitative_financial.json (phase-2 judge fingerprints —
               written when 02c runs; needs 02b first),
               llama_02/ (archived older Llama-3.3-70b prompt-variance run),
               03_blend_leaderboard.csv + 03_locked_params.csv + 03_*.png
               (hybrid — written when 03 runs),
               04_final_benchmark.csv + 04_*.png (final benchmark — written
               when 04b runs),
               05a_*/05b_*/05c_* summary.csv + predictions.csv + .png (RAG —
               written when 05a/b/c run; eval on robustness_batch),
               llm_calls.csv (per-call cost log)                               (tracked)
models/        xgb_model.joblib, lr_model.joblib, ann_model.keras,
               thresholds.joblib                                              (TRACKED — small, and
               the LLM notebooks load xgb_model + scaler + thresholds via run_ml_on_sample)
notebooks/
  ml_models/                 # classical ML pipeline (parallel to llm_models/)
    01_EDA.ipynb
    02_Preprocessing.ipynb   # writes data/processed/02_*
    03_Modeling.ipynb        # writes models/* and data/results/ml/03_model_performance.csv
  llm_models/
    .env                     # API keys + GCP config, gitignored — see "API keys" below
    llm_utils.py             # shared: data loading, ML re-encoding, prompts, API calls, eval, cost logging
    llm_pricing.py           # per-model USD/1k token prices used by the cost logger
    sample_generation.py     # the 3 samples: get_tuning_sample/get_robustness_batch/get_test_batch
                             # (load-if-exists + force; deterministic; mutually exclusive)
    01_model_selection/      # PHASE 1: pick the model AND its config → GPT-5.4
      01a_Model_Comparison.ipynb    # GPT-5.4 / Gemini 2.5 Pro / Gemini 3.5 Flash / Claude Sonnet 4.6 / Claude Opus 4.8
                                    # ±desc → 01a_predictions.csv + 01a_metrics.csv
      01b_Consistency.ipynb         # GPT-5.4 only, 3 runs × 2 conditions → 01b_predictions.csv + 01b_metrics.csv
      01c_Robustness.ipynb          # GPT-5.4 on held-out batch → 01c_predictions.csv + 01c_metrics.csv
      01d_reasoning_effort_runs.ipynb  # GPT-5.4 reasoning_effort sweep → 01d_predictions/01d_metrics
                                       # (pick best effort by AUC; carry it into the final benchmark)
      01e_Confidence_Calibration.ipynb # META-ANALYSIS (no API cost): reads llm_calls.csv +
                                       # tuning labels → calibration (ECE/Brier/reliability),
                                       # confidence-vs-correctness, confidence-vs-stability (01b link),
                                       # cross-model. OpenAI+Gemini only (Anthropic has no logprobs).
                                       # Run LAST in phase 1 — it reads what 01a/01b/01d logged.
                                       # → 01e_confidence_metrics.csv + 01e_*.png
      01f_Qualitative_Financial_Analysis.ipynb # PHASE-1 ANALYSIS GATE (spends API $:
                                       # GPT-5.4 judge). Part 1 = portfolio cost ledger for
                                       # 01a models + 01d efforts (per-loan cost pulled from
                                       # llm_calls.csv). Part 2 = GPT-5.4 "reasoning fingerprint"
                                       # per model/effort. → 01f_qualitative_financial.json
    02_prompt_variance/      # PHASE 2: prompt structuring & design
      02a_Batching_Formatting_Tax.ipynb  # 4 formatting/batching conditions × 3 runs → 02a_*.csv
      02b_Prompt_Variance.ipynb          # 7 prompt variants × comparison/consistency/robustness/±desc → 02b_*.csv
      02c_Qualitative_Financial_Analysis.ipynb # PHASE-2 ANALYSIS GATE (spends API $:
                                         # GPT-5.4 judge). Part 1 = cost ledger for 02b variants.
                                         # Part 2 = GPT-5.4 reasoning fingerprint per variant.
                                         # Needs 02b's outputs first. → 02c_qualitative_financial.json
    03_hybrid/               # PHASE 3: blended XGBoost + LLM (Jad). Beat both solo on Charged-Off F1?
      03_Blended_LLM_ML.ipynb       # Uses Groq (Llama-3.3-70b-versatile). Batched + cached signals.
                                    # Tuned on tuning_sample, strategy-selected on robustness_batch.
                                    # TEST SET IS NEVER LOADED — strict holdout.
                                    # Strategies: soft blend, confidence gate, 5A/5B risk scorers,
                                    # sentence embeddings (Part 7, optional — needs sentence-transformers).
                                    # → 03_blend_leaderboard.csv + 03_locked_params.csv + 03_*.png
    04_final_benchmark/      # PHASE 4 (the spine, run last): finalists × GPT-5.4, threshold + COST vs XGBoost
      04_Final_Test_Analysis.ipynb   # FINAL BENCHMARK & DASHBOARD: fill in FINALISTS, then run.
                                    # Loads 03_locked_params.csv to apply hybrid ensembles post-hoc.
                                    # → 04_final_benchmark.csv + 04_benchmark_f1_vs_cost.png
                                    # Threshold tuning lives ONLY here, on the actual finalists.
                                    # Pulls metrics + per-loan cost + qualitative fingerprint into one table.
    05_rag/                  # PHASE 5: retrieval-augmented credit scoring (Jad). Precedent loans as evidence.
      rag_utils.py                  # corpus builder (build_rag_corpus), Embedder, ResidualKMeansQuantizer
                                    # (Semantic IDs), retrieval primitives, RRF, assert_no_leakage.
      00_Build_RAG_Dataset.ipynb    # Builds data/processed/rag_corpus.csv (full frame − eval batches;
                                    # ~100-row tuning_sample dev fallback when no raw .gz).
      05a_RAG_Generative_SemanticID.ipynb  # TIGER Semantic IDs + multi-stage (RAG-FLARKO)
      05b_RAG_FullCorpus_Retrieval.ipynb   # full-corpus dense kNN
      05c_RAG_Hybrid.ipynb                 # RRF fusion of A+B + A/B retriever-overlap diagnostic
                                    # All eval on robustness_batch (validation) — TEST SET NEVER LOADED.
                                    # NVIDIA Llama-3.3-70B + sentence-transformers (TF-IDF fallback).
                                    # → 05{a,b,c}_summary.csv + 05{a,b,c}_predictions.csv + 05*_*.png
    # NAMING: file prefix = phase (01a, 02, 03, 04, 05). Result CSVs share the same
    # prefix as the notebook that writes them.
reports/       Progress report 1.pdf, output.png
```

**Collaboration policy (changed May 2026):** `data/processed/` samples, `models/`,
and all of `data/results/` are now **tracked** so teammates can `git pull` and
explore results without re-running notebooks. Only genuinely-excluded items:
`data/raw/*` (huge source dumps), `data/processed/02_processed_data.npz` (109MB,
over GitHub's limit), and `.env`. `.keep` files preserve otherwise-empty dirs.

## Run order

Because models are now tracked, a teammate can run the LLM notebooks **without**
re-running the ML pipeline (the LLM notebooks load `tuning_sample.csv` + the saved
scaler/feature-columns/XGBoost model via `llm_utils.run_ml_on_sample`, all in git).
Only re-run the ML pipeline if you change preprocessing/features:

```
01_EDA → 02_Preprocessing → 03_Modeling   [regenerates data/processed/ and models/]
python llm_models/sample_generation.py    [generates robustness_batch + test_batch; only before 01c / benchmark]
01_model_selection/01a,01b,01c,01d        [independent, any order]
01_model_selection/01e                    [reads 01a/01b/01d's logprobs from llm_calls.csv]
01_model_selection/01f                    [analysis gate — after 01a+01d; GPT-5.4 judge → 01f_qualitative_financial.json]
02_prompt_variance/02a                   [writes 02a_tax_metrics.csv]
02_prompt_variance/02b                   [writes 02b_predictions.csv + 02b_phase1_metrics.csv]
02_prompt_variance/02c                   [analysis gate — AFTER 02b; GPT-5.4 judge → 02c_qualitative_financial.json]
03_hybrid/03_Blended_LLM_ML              [standalone — uses Groq directly, does NOT read 02_predictions.csv]
05_rag/00_Build_RAG_Dataset              [builds rag_corpus.csv; run before 05a/b/c]
05_rag/05a,05b,05c                       [standalone — eval on robustness_batch; run 05a+05b before 05c for its leaderboard]
04_final_benchmark/04_Final_Test_Analysis   [run LAST — final benchmark and dashboard, loads 03_locked_params.csv]
```

`02_Preprocessing.ipynb` does a 67/33 train/test split with `random_state=42`.

## Samples

Three role-based samples, **all generated through `llm_models/sample_generation.py`**
(the held-out batch logic used to be scattered across `00_Sample_New_Batch` + a
`sample_new_batch` helper — now centralized):

| Sample | Role | Seed | Made by |
| ------ | ---- | ---- | ------- |
| `tuning_sample.csv` | dev / tuning (01a/b/d, 02, 03 blend tuning, 04 threshold tuning) | 42 | `ml_models/02_Preprocessing` (stratified); util loads it |
| `robustness_batch.csv` | robustness check (01c) + **validation / strategy selection** (03 hybrid) | 99 | `get_robustness_batch()` — excludes tuning |
| `test_batch.csv` | **final test, touched once in 04 only** | 2024 | `get_test_batch()` — excludes tuning + robustness |

- **`get_*()` is load-if-exists** (never clobbers a committed sample); pass
  `force=True` to regenerate. Deterministic given the raw CSV.
- **All three share one 35-column schema** (incl. FICO, delinquency, inquiries,
  `emp_length`, `credit_history_years`) so `run_ml_on_sample` and the LLM see the
  same features on every batch. *(The old `held_out_batch.csv` was missing the 9
  newer features — XGBoost was scoring it with FICO=0, etc. Fixed by regenerating
  all batches through the util.)*
- `run_ml_on_sample` encodes `term`/`emp_length` via `is_numeric_dtype` checks
  (not `== object`) so it works under pandas versions that infer `str` dtype.

## Strict test holdout protocol

**`test_batch.csv` must NEVER be loaded, queried, or evaluated outside of
`04b_Final_Benchmark.ipynb`.** This is a non-negotiable rule established
May 2026.

- **Phase 3 (03_hybrid)** tunes hyperparameters on `tuning_sample` and
  selects the winning strategy on `robustness_batch` (validation). It saves
  `03_locked_params.csv` with the frozen parameters.
- **Phase 4 (04_Final_Test_Analysis.ipynb)** is the ONLY notebook that loads
  `test_batch.csv`. It runs the finalist LLM prompts and XGBoost on the
  test set, then applies the Phase 3 hybrid strategies **post-hoc** using
  the locked parameters — requiring zero additional API calls for the
  ensembled comparisons.
- If you create a new notebook that needs to evaluate model performance,
  use `tuning_sample.csv` or `robustness_batch.csv`. **Never import or
  reference `test_batch.csv`** outside Phase 4.

## API keys

`notebooks/llm_models/.env` (gitignored) holds:

```
OPENAI_API_KEY=...
OPENAI_API_KEY_2=...          # spare
OPENAI_API_KEY_3=...          # spare
GEMINI_API_KEY_PRO=...
GEMINI_API_KEY_FLASH=...
ANTHROPIC_API_KEY=...
ANTHROPIC_API_KEY_2=...       # spare
NVIDIA_API_KEY=...            # phase 02 NVIDIA NIM (Llama)
NVIDIA_API_KEY_2=...          # spare
GROQ_API_KEY=...              # phase 03 Groq (Llama) — add this before running 03_hybrid
GCP_PROJECT_ID=capstonesabadell   # Vertex AI project for Gemini logprobs
GCP_LOCATION=europe-west4        # Netherlands — reliable for newest Gemini models in EU
```

`llm_utils.load_api_key(api_provider, model)` reads this file. Gemini keys are
selected per-model (Flash vs Pro). The `GCP_*` variables are used by the Gemini
provider in `call_llm` to route through Vertex AI when a project is configured.

**Supported providers** (each with retry logic in `call_llm`):
| Provider | `api_provider` value | Key env var | Used in |
|----------|---------------------|-------------|---------|
| OpenAI   | `"openai"` | `OPENAI_API_KEY` | 01a–01e, 04 |
| Google Gemini | `"gemini"` | `GEMINI_API_KEY_PRO` / `_FLASH` | 01a |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | 01a |
| NVIDIA NIM | `"nvidia"` | `NVIDIA_API_KEY` | 02 |
| Groq | `"groq"` | `GROQ_API_KEY` | 03 |

## Environment

- Python: **3.11** (project runs from the `sabadell` conda env at
  `/opt/homebrew/Caskroom/miniforge/base/envs/sabadell/`).
- Recreate from scratch with:
  ```
  conda create -n sabadell python=3.11 -y
  conda activate sabadell
  pip install -r requirements.txt
  python -m ipykernel install --user --name sabadell --display-name "Python (sabadell)"
  ```
  Then in Jupyter / VS Code, pick the **Python (sabadell)** kernel for every notebook.
- No lockfile is committed. `requirements.txt` lists the packages that the
  notebooks and `llm_utils.py` import; pin versions there if you need reproducibility.
- Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`,
  `tensorflow` (Keras), `shap`, `statsmodels`, `joblib`, `matplotlib`,
  `seaborn`, `python-dotenv`, `pyyaml`, `google-genai`, `openai`, `anthropic`.
- `sentence-transformers` is optional — only needed for Part 7 of
  `03_Blended_LLM_ML.ipynb` (borrower-description embeddings). That section
  skips gracefully if it's not installed.

## Conventions / gotchas

- **Commit policy (changed May 2026):** processed samples, `models/`, and all of
  `data/results/` are now tracked for collaboration — commit them. The hard
  exclusions are `data/raw/*`, `data/processed/02_processed_data.npz` (109MB), and
  `.env` (secrets). When committing, the easiest leak is `data/raw/` if it's
  present locally — stage by path, and never commit `.env`.
- **Path resolution**: `llm_utils.py` anchors `DATA_DIR`, `MODEL_DIR`,
  `RESULTS_DIR`, and `RAW_DATA_PATH` to `Path(__file__).resolve().parent.parent.parent`
  (the repo root), so any notebook under `notebooks/llm_models/**` can import
  `llm_utils` and read/write the right files regardless of nesting depth. The
  ML notebooks (`01/02/03`) under `notebooks/ml_models/` use plain relative paths
  (`../../data/...`, `../../models/...`) — run them from their own directory.
- **Importing `llm_utils` from a subfolder**: notebooks living under
  `notebooks/llm_models/<subfolder>/` need a one-liner `import sys;
  sys.path.insert(0, "..")` before `from llm_utils import ...`. This is
  already in place in every phase notebook; copy the pattern for new notebooks
  under `03_hybrid/` or `04_final_benchmark/`.
- **`llm_utils.LLM_FEATURES`** is the canonical **30-feature** list that LLMs see
  (the original 21 + FICO, delinquency, inquiries, `emp_length`,
  `credit_history_years` — matching the ML feature set). Excludes the target,
  `desc`, and identifiers. If you change features in `02_Preprocessing.ipynb`,
  update `LLM_FEATURES` and `FEATURE_DESCRIPTIONS` in `llm_utils.py` to match,
  otherwise `run_ml_on_sample` will misalign columns.
- **Retry logic** for LLM calls: `llm_utils.call_llm` retries 8× with
  exponential backoff on 503/429/502/504/connection errors. Groq has
  custom retry-after header parsing and day-scale quota detection (fails
  fast if RPD/TPD is exhausted rather than sleeping for hours). If a run
  hangs, that's the loop — kill the cell rather than waiting it out.
- **GPT-5.4 model name**: the notebooks pass the literal string `gpt-5.4` to
  the OpenAI SDK (resolved to `gpt-5.4-2026-03-05`). The final benchmark uses
  `MODEL = 'gpt-5.4'`. If that route 404s, check the model registry and
  update the string in the notebook (not in `llm_utils.py`).
- **Groq rate limits**: Groq's free tier is tight (30 RPM / 1000 RPD / 12000
  TPM). The `03_hybrid` notebook uses batched requests (50 loans/call for
  binary, 20 loans/call for 5B) and a 2-second throttle between calls to
  stay under these limits. Signal results are cached per-sample as
  `03_groq_{signal}_{sample}.csv` so a quota interruption resumes without
  re-burning tokens.
- **`llm_calls.csv` column schema** (14 columns): `timestamp, notebook_id,
  label, desc_tag, provider, model, row_index, input_tokens, output_tokens,
  input_price_per_1k_usd, output_price_per_1k_usd, cost_usd,
  prob_fully_paid, reasoning_effort`. The `notebook_id` column records the
  executing notebook's **filename** (e.g. `01a_Model_Comparison.ipynb`); outside
  Jupyter it falls back to the cwd folder name, so pass `notebook_id=` to
  `run_llm_experiment` explicitly when running from a script. **Do not add or
  remove columns** — pandas pivot tables in the analysis notebooks depend on
  this schema.
- **`llm_calls.csv` is written idempotently + atomically** (`_append_llm_calls`
  in `llm_utils.py`): re-running an experiment **replaces** its own rows
  (keyed on `notebook_id, label, desc_tag`) instead of appending duplicates,
  and writes go through a temp file + atomic rename so a crash can't corrupt
  the log. Every logical run uses a distinct `label` (e.g. `... Run 1
  (no_desc)`, `... | run1`, `reasoning=high`), so a re-run never clobbers a
  sibling. Use `llm_utils.dedup_llm_calls()` once to collapse any legacy
  duplicates from before this writer existed.
- **Predictions files embed cost** (01a/01b/01c/01d/02b): `run_llm_experiment`
  returns per-loan `input_tokens`/`output_tokens`/`cost_usd`, and the export
  cells write them into the `*_predictions.csv`, so cost is derivable from the
  predictions alone and can't silently drift from `llm_calls.csv`. (02a's
  `Control (Natural Indiv)` row is intentionally **derived from 01b's no_desc
  runs** via `load_baseline_metrics()` — it is not a separately-logged run.)
- **Determinism / loud failures**: `run_llm_experiment` accepts `temperature`
  and `seed` (forwarded to every provider; reasoning models ignore temperature —
  use `reasoning_effort`), plus `strict=True`/`max_fail_frac` to raise (after
  logging successful rows) if too many predictions fail, so a half-broken run
  can't masquerade as complete.
- **Phase 3 output files**: `03_locked_params.csv` contains the frozen
  strategy hyperparameters (alpha, threshold, band) chosen on the validation
  set. Phase 4 loads this file automatically for post-hoc ensembling on the
  test set. If you change the Phase 3 strategies, re-run Phase 3 to
  regenerate this file before running Phase 4.
- **No hardcoded feature lists in notebooks**: the `top_features_only`
  prompt variant in Phase 2 (`02b_Prompt_Variance`) derives its features
  LIVE from XGBoost importances via `llm_utils.top_xgb_features(n=8)` — never
  hard-coded. If you retrain XGBoost, the feature list updates automatically.
  Exception: Phase 3's batched Groq calls use a static `TOP_FEATURES` list
  for the binary signal (matching the Phase 1 winner prompt structure).
- **Branch**: development happens on `main`. Don't create feature branches
  unless working on experimental changes that may need reverting.
- **Commit attribution**: do **not** add a `Co-Authored-By: Claude …` trailer
  to commits. Commits should be attributed only to the human GitHub user
  (`Alessandro Mezzanotte`). When committing on this repo, omit the
  Claude co-author line entirely.

## Model lineup (May 2026)

| Model | Provider | `api_provider` | Phase | Logprobs |
|-------|----------|---------------|-------|----------|
| GPT-5.4 | OpenAI | `"openai"` | 01, 04 | ✅ |
| Gemini 2.5 Pro | Google (Vertex AI) | `"gemini"` | 01 | ✅ |
| Gemini 3.5 Flash | Google (AI Studio) | `"gemini"` | 01 | ❌ |
| Claude Sonnet 4.6 | Anthropic | `"anthropic"` | 01 | ❌ |
| Claude Opus 4.8 | Anthropic | `"anthropic"` | 01 | ❌ |
| Llama-3.3-70b-instruct | NVIDIA NIM | `"nvidia"` | 02 | ❌ |
| Llama-3.3-70b-versatile | Groq | `"groq"` | 03 | ❌ |

Pricing for all models lives in `llm_pricing.py` (`PRICES` dict). Legacy
entries are kept so historical `llm_calls.csv` rows stay accurate.

## Reports

`reports/Progress report 1.pdf` is the latest written-up summary of findings
(Apr 3 2026). There is no committed script that generates it; it's authored
ad-hoc. The headline numbers in this file's "Where things stand" section come
from that PDF — if you re-run experiments and the numbers shift, update both.
