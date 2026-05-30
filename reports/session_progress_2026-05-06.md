# Capstone progress — May 6, 2026 working session

Snapshot of everything moved or built in this single session. The April
progress report is the prior baseline; everything below is *delta* against it.

## Headline numbers

| Surface | Before this session | After this session |
| --- | --- | --- |
| ML XGBoost AUC | (none reported) | **0.712** |
| ML XGBoost F1 (Charged Off) | 0.383 | **0.394** |
| ML LR AUC / F1 | (none) / 0.386 | **0.708 / 0.390** |
| ML ANN AUC / F1 | (missing — partial run) | **0.709 / 0.390** |
| Method-cleanness of ML metrics | three sources of test-set leakage | proper train / val / test split |
| LLM cost tracking | none | per-call USD logged automatically |
| LLM AUC | not computable | infra ready (logprob extraction) |
| LLM feature parity with ML | LLM saw 21 features; ML saw 30 | both see the same 30 |
| Repo organisation | flat `notebooks/llm_models/` | split into `model_selection/` and `optimization/` |
| Project Python env | dead `pyclass` reference | fresh `sabadell` conda env, Jupyter kernel registered |

---

## 1. Machine-learning pipeline — audit and rebuild

### Issues found

1. **Test-set leakage (three places).** Optuna's objective scored against
   `roc_auc_score(y_test, ...)`, the ANN's `validation_data=(X_test, y_test)`
   was used by `EarlyStopping(restore_best_weights=True)`, and
   `find_best_threshold(y_test, probs)` picked the operating point on the
   same data the metrics were reported on. All three made the reported
   metrics optimistic vs. true held-out generalisation.
2. **`scale_pos_weight` was inverted.** XGBoost was being told to up-weight
   the *majority* (Fully Paid) class, the opposite of what imbalance
   correction requires. The fact that LR (`class_weight='balanced'`) was
   nearly tied with XGBoost on minority-class recall suggested the bug.
3. **Strong predictive features were dropped.** The raw LendingClub frame
   has 151 columns; only 21 were kept. Notably absent: FICO score, recent
   delinquencies, recent inquiries, credit-history length.
4. **Count features were binarised.** `pub_rec`, `mort_acc`,
   `pub_rec_bankruptcies` were collapsed to 0/1, throwing away the
   "1 bankruptcy vs. 3 bankruptcies" distinction.
5. **`emp_length` was dropped despite showing a clean monotonic gradient.**
6. **`earliest_cr_line` was reduced to a year integer**, when
   `(issue_d − earliest_cr_line)` (credit-history length) is the
   directly meaningful quantity.

### Fixes applied

- New stratified train / **val** / test split. Optuna scores against val,
  ANN early stopping monitors val, threshold is picked on val,
  test stays untouched until final eval.
- `scale_pos_weight = n_neg / n_pos` (correct direction).
- Added: `fico_range_low`, `fico_range_high`, `delinq_2yrs`, `inq_last_6mths`,
  `mths_since_last_delinq` (with sentinel `999` + sister `has_past_delinq` flag),
  `acc_open_past_24mths`.
- Replaced binarisation with raw counts.
- `emp_length` ordinal-encoded (0–10).
- `credit_history_years = (issue_d − earliest_cr_line) / 365.25`.

### What actually moved

| Model | AUC (old → new) | F1 Charged Off (old → new) | Recall Charged Off |
| --- | --- | --- | --- |
| LR | 0.699 → **0.708** | 0.386 → 0.390 | 0.596 → 0.593 |
| XGBoost | 0.705 → **0.712** | 0.390 → 0.394 | 0.583 → **0.619** |
| ANN | (partial) → **0.709** | (partial) → 0.390 | — |

The audit prediction was AUC ~ 0.74–0.78. Reality: ~ +0.007. Two reasons:

- The leakage was small in absolute terms because the test set is very
  large (~135k rows) — best-of-50-trial Optuna inflation on 135k is
  marginal.
- FICO turned out to be largely redundant with `sub_grade` (LendingClub's
  internal grade is computed from FICO + a few other things, so adding raw
  FICO on top of `sub_grade` adds little marginal information).

**The substantive finding for the supervisor:** structured features cap at
**AUC ≈ 0.71** on this dataset, even with FICO + delinquencies + inquiries
explicitly added. LR is at 0.708, XGBoost at 0.712, ANN at 0.709. Wildly
different model capacities all converge on the same number — the ceiling is
the features, not the model. This empirically substantiates the team's
working hypothesis that further ML tuning is low-leverage.

### What this implies for the LLM optimisation

The bar GPT-5 (no-desc) needs to clear is **AUC ≈ 0.71, F1 ≈ 0.39**. If the
LLM beats this from the same 30 structured features, that's evidence its
reasoning extracts something the ML models miss.

---

## 2. LLM infrastructure — major upgrades

### Cost tracking

`notebooks/llm_models/llm_pricing.py` (new): per-model USD/1k token prices
for `gpt-5`, `gemini-2.5-pro`, `gemini-2.5-flash`, `claude-sonnet-4-20250514`.

`run_llm_experiment` now appends a row per successful API call to
`data/results/llm/llm_calls.csv`. Schema:

```
timestamp, experiment_id, label, desc_tag, provider, model, row_index,
input_tokens, output_tokens, input_price_per_1k_usd, output_price_per_1k_usd,
cost_usd, prob_fully_paid, reasoning_effort
```

Interrupted runs (KeyboardInterrupt or exceptions) drop the buffer, so the
file only ever contains data from fully-completed experiments.

### Logprob → AUC infrastructure

OpenAI and Gemini both expose token logprobs; the framework now requests
them and extracts `P("1") / (P("1") + P("0"))` per call as
`prob_fully_paid`. Anthropic doesn't expose logprobs, so its rows are NaN.

`evaluate_predictions(...)` accepts an optional `probabilities` array and
adds AUC to the metrics dict when present.

### Feature symmetry

`LLM_FEATURES` was 21 features (pre-audit). Now 30, matching the ML
feature set: added FICO range, delinq, inquiries, credit-history-years,
employment length, has_past_delinq, mths_since_last_delinq, accounts opened
past 24 months. Plus a label fix: `pub_rec`, `mort_acc`,
`pub_rec_bankruptcies` are now described as "Number of …" instead of
"(0/1)" since they're raw counts post-audit.

`mths_since_last_delinq` uses sentinel `999` for "no recorded delinquency";
the prompt formatter renders that as the string "no delinquency on record"
so the LLM doesn't try to interpret 999 as a real count.

### Reasoning effort + prompt overrides

`call_llm(..., reasoning_effort=None)` and
`run_llm_experiment(..., reasoning_effort=None, system_prompt=None)` both
accept these new optional kwargs. `reasoning_effort` is OpenAI-only;
non-OpenAI providers warn-once and ignore.

### Path resolution

`llm_utils.DATA_DIR / MODEL_DIR / RESULTS_DIR / RAW_DATA_PATH` are now
anchored to the repo root via `Path(__file__).resolve().parent.parent.parent`,
so notebooks at any depth under `notebooks/llm_models/**` find the right
files regardless of working directory.

---

## 3. Repository reorganisation

### Phase split

```
notebooks/llm_models/
  llm_utils.py            # shared utilities
  llm_pricing.py          # per-model USD prices
  model_selection/        # PHASE 1 (concluded — GPT-5 picked)
    00_Sample_New_Batch.ipynb
    04a_Model_Comparison.ipynb
    04b_Consistency.ipynb
    04c_Robustness.ipynb
  optimization/           # PHASE 2 (active — improve GPT-5 no-desc)
    05_reasoning_effort_runs.ipynb     (new)
    06_threshold_tune_and_test.ipynb   (new)
  archive/                # earlier pilots, untouched
```

### Result-file consolidation

| Before | After |
| --- | --- |
| 6 separate `04a_<model>_<desc>.csv` | 1 `04a_predictions.csv` with `model` + `desc_tag` columns |
| 2 `04b_consistency_predictions_*.csv` + 3 metrics CSVs | 1 `04b_predictions.csv` + 1 `04b_metrics.csv` |
| 2 `04c_robustness_<desc>_results.csv` + 1 metrics | 1 `04c_predictions.csv` + 1 `04c_metrics.csv` |
| Misnamed `04a_model_comparison_metrics.csv` | `04a_metrics.csv` |
| `data/processed/03_model_performance.csv` (output in input dir) | moved to `data/results/ml/` |
| `data/results/llm/04c_new_batch_sample.csv` (input in output dir) | moved to `data/processed/` |
| Unused `models/ann_threshold.joblib` | deleted (superseded by `thresholds.joblib`) |

Net: `data/results/llm/` went from 17 files to 7 (plus auto-written
`llm_calls.csv` and the unchanged `archive/`).

---

## 4. Tooling

### Conda environment

Fresh `sabadell` env at
`/opt/homebrew/Caskroom/miniforge/base/envs/sabadell/`, Python 3.11.
Replaces the dead `pyclass` micromamba reference. Jupyter kernel registered
as "Python (sabadell)". `requirements.txt` updated with `shap` and
`statsmodels` for explainability work. `libomp` installed in `sabadell`,
`ds`, `dl`, and `geo` envs to fix XGBoost's macOS OpenMP runtime issue.

### Project-shared Claude Code skills

15 K-Dense-AI scientific skills installed at `.claude/skills/`:

`shap`, `scientific-writing`, `scientific-slides`, `pptx-posters`,
`latex-posters`, `scientific-visualization`, `infographics`, `pdf`,
`scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`,
`exploratory-data-analysis`, `statistical-analysis`, `peer-review`.

Pinned to commit `37a148ba51810e930f89551b5485c331f42171ca` of the
upstream repo. Available to anyone working on this project who has the
Claude Code CLI configured.

---

## 5. Optimization phase — first experiment scaffolded

### `05_reasoning_effort_runs.ipynb`

Pure data generation. Runs GPT-5 no-desc on `02_llm_sample.csv` at
`reasoning_effort ∈ {low, medium, high}`. Three runs in parallel via
`ThreadPoolExecutor(max_workers=3)`, each with its own API key
(`OPENAI_API_KEY`, `OPENAI_API_KEY2`, `OPENAI_API_KEY3`) so per-key rate
limits don't bottleneck. Outputs `05_predictions.csv` and
`05_metrics.csv` (default-threshold view). Estimated cost ~$4.

### `06_threshold_tune_and_test.ipynb`

Pure analysis + held-out validation. Loads 05's outputs, computes
F1-optimal threshold per variant on the tuning sample, presents a
decision table, then runs only the **winner** on the held-out resample
(`04c_new_batch_sample.csv` after refresh) and applies the tuned threshold
for final test-set metrics. Estimated cost $0.50–$3 depending on which
variant wins.

The 05/06 split exists because threshold tuning is cheap (no API spend)
and the LLM data generation is expensive — keeping them separate means
re-tuning with a different objective costs nothing.

### Workflow on next sit-down

1. Re-run `model_selection/00_Sample_New_Batch.ipynb` once to refresh the
   held-out batch under the new preprocessing.
2. Run `05_reasoning_effort_runs.ipynb` end-to-end (~5–10 min wall time).
3. Open `06`, run cells 1–4, look at the decision table.
4. Edit `WINNER_REASONING_EFFORT = "..."` in cell 5.
5. Run cells 6–9. Held-out test of the winner. Final headline metrics.

---

## 6. Open questions for the supervisor

- **Prompt-engineering scope.** Beyond `reasoning_effort`, the natural
  follow-ups are few-shot examples (with hard misclassified loans
  curated from the ML model's mistakes) and prompt-rewording variants.
  Both fit the supervisor's "improve no-desc" framing. Worth confirming
  priorities.
- **Cost vs accuracy frontier.** GPT-5-mini and GPT-5-nano exist at
  ~5× and ~25× lower cost than GPT-5. If the structured-features ceiling
  really is ~0.71, mini might match it. Worth a side experiment if
  budget is a stated criterion of the brief.
- **Re-running model_selection (04a/b/c).** Those experiments concluded
  GPT-5 wins, but with the *old* prompt (21 features, no FICO). For the
  June presentation's apples-to-apples cleanness, re-running them with
  the new 30-feature prompt is ~$5 and ~1 hour. Defer or do it now?
- **Ceiling story.** The headline finding — "every classical model
  converges on AUC ≈ 0.71, suggesting the structured features cap there"
  — is itself report-worthy. Worth confirming the supervisor wants this
  framed as a result, not a problem.

---

## Files changed this session

```
M  CLAUDE.md
M  notebooks/02_Preprocessing.ipynb
M  notebooks/03_Modeling.ipynb
M  notebooks/llm_models/llm_utils.py
   notebooks/llm_models/llm_pricing.py                       (new)
   notebooks/llm_models/model_selection/{00,04a,04b,04c}     (renamed in)
   notebooks/llm_models/optimization/05_reasoning_effort_runs.ipynb       (new)
   notebooks/llm_models/optimization/06_threshold_tune_and_test.ipynb     (new)
M  requirements.txt
M  .gitignore
   .claude/skills/                                           (15 skills, new)
   data/results/llm/                                         (consolidated layout)
   reports/Progress report 1.pdf                             (committed in)
   reports/session_progress_2026-05-06.md                    (this file)
   "P280 - Classical ML vs. LLM for Credit Scoring - Banc Sabadell.pdf"   (committed in)
D  reports/llm_evaluation_report.pdf                         (obsolete)
D  reports/llm_evaluation_report.py                          (obsolete)
D  models/ann_threshold.joblib                               (unused)
```
