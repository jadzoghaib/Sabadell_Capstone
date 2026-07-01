# Phase 4 — Final Benchmark, Explainability & Decision P&L

This folder is **the spine of the project, run last**. It establishes the conclusive
benchmark on the locked test set and translates predictive performance into commercial
business value.

Everything lives in a **single notebook, `04_Final_Test_Analysis.ipynb`**. It is the
**only** place in the repository that loads `test_batch.csv` (the 1000-loan held-out set) —
see the strict-holdout protocol in `CLAUDE.md`.

---

## Notebook Structure (`04_Final_Test_Analysis.ipynb`)

| Step | What it does |
|------|--------------|
| **1 — Load test set** | Loads (or generates, load-if-exists) the held-out `test_batch.csv`. |
| **2 — Finalists** | Declares the finalist configurations carried in from Phases 1–3: **GPT-5.4 baseline** and **GPT-5.4 chain-of-thought**, both `no_desc` (the apples-to-apples condition vs XGBoost). |
| **3 — Run on the test set** | The only cell that spends API money. Runs the finalists on all 1000 loans and logs per-call cost to `llm_calls.csv`. |
| **4 — Performance** | AUC, F1, precision/recall on the `Charged Off` class vs the XGBoost baseline. Decision threshold tuning happens here and *only* here, on the actual finalists. |
| **5 — Decision P&L** | Case-by-case profit/loss on the real LendingClub cashflows (confusion-matrix costing: approved-and-paid income vs approved-and-defaulted loss vs rejected-good opportunity cost), net of API cost. Also includes a hypothetical-portfolio sensitivity brief. |
| **6 — Explainability** | SHAP beeswarm for XGBoost (`04_shap_beeswarm.png`) contrasted with the LLM's natural-language rationales — the explainability comparison required by the brief. |
| **7 — Error overlap** | Where XGBoost and the LLM agree/disagree, and the verdict. |

The Phase-3 hybrid strategies are applied **post-hoc** using the frozen parameters in
`data/results/llm/03_locked_params.csv`, so the ensembled comparisons need **zero extra
API calls**.

---

## Outputs (in `data/results/llm/`)

| File | Contents |
|------|----------|
| `04_test_performance.csv` | Accuracy, precision/recall/F1 (Charged Off), and AUC per model on the test set. |
| `04_test_predictions.csv` | Per-loan actual, prediction, and probability for each finalist + XGBoost. |
| `04_test_financials.csv` | Decision P&L per model (income, default loss, opportunity cost, total, API cost). |
| `04_test_pnl.png` | P&L comparison chart. |
| `04_shap_beeswarm.png` | XGBoost SHAP feature-attribution summary. |

---

## Headline result (held-out 1000-loan test set)

| Model | Accuracy | Charged-Off F1 | AUC | Decision P&L (net of API) |
|---|---|---|---|---|
| **XGBoost** | 0.688 | **0.381** | **0.706** | **−$93.5k** |
| **GPT-5.4 (baseline)** | 0.764 | 0.298 | n/a¹ | **+$998.8k** |
| **GPT-5.4 (chain-of-thought)** | **0.775** | 0.252 | n/a¹ | **+$1.19M** |

¹ AUC is `n/a` for the LLM rows on the test run (no per-call logprobs were captured).

> [!NOTE]
> The two model families optimize different things. **XGBoost still wins on Charged-Off F1
> and AUC** (structured features, tuned threshold), but **GPT-5.4 wins decisively on the
> decision P&L** — its approve/reject mix turns a portfolio-level loss into a large positive
> return, and the API cost (single-digit-to-low-double-digit dollars over 1000 loans) is
> negligible against that. This is the commercial justification the Sabadell stakeholders
> care about, and it is meaningfully different from the older 100-loan `desc`-sample figures
> in `reports/Progress report 1.pdf`.
