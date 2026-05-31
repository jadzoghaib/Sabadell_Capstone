# Phase 4 — Final Benchmark & Executive Financial Dashboard

This folder contains the conclusive experiments of the Capstone project, establishing final benchmarks on the locked test set and evaluating commercial business value.

---

## Notebook Structure

### 1. Executive Financial Simulation (`04a_Financial_Simulation.ipynb`)
* **Objective:** Simulates actual bank credit portfolio metrics ($ yields, net cash flows, losses, and operating margins) by combining predictions with financial realities.
* **Key Focus Areas:**
  * Maps business cost-benefit curves across traditional classifiers, pure LLM prompts, and hybrid ensembles.
  * Formulates precise OpenAI prompt caching economics (piecewise 50% discount on inputs above 1,024 tokens) to prevent artificially inflated operating costs.
  * Measures the financial impact of False Positives (lost principal) vs. False Negatives (lost interest profit).

### 2. Final Benchmarking & Threshold Tuning (`04b_Final_Benchmark.ipynb`)
* **Objective:** The conclusive spine of the LLM pipeline. Run the final optimized prompting variants × `gpt-5.4` on the untouched `test_batch.csv`.
* **Key Features:**
  * Runs the final comparative assessment of pure LLM prompting vs. classical ML on the held-out test data.
  * Imports the locked parameters from Phase 3 (`03_locked_params.csv`) and applies the ensembled/hybrid predictions post-hoc to the test set without needing extra API calls.
  * Executes credit decision threshold tuning to balance risk vs. volume, maximizing the commercial return of the loan portfolio.
* **Outputs:** 
  * `data/results/llm/04b_final_benchmark.csv` — Comprehensive final ledger of test set metrics.
  * Yield curves and model benchmark visualizations.

---

## Executive Financial Reference

When comparing models on credit portfolios, traditional accuracy is highly misleading. A model with high default recall (like Llama) but high False Positives can cost the bank millions in lost loan interest, while a model with structured precision (like GPT-5.4) can preserve both principal and margin.

This phase provides the commercial justification to Banc Sabadell stakeholders for utilizing high-reasoning LLMs and hybrid ensembling in modern credit decisioning.
