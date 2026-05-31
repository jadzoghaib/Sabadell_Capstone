# Phase 3 — Hybrid Blended LLM/ML Approach

This folder contains Jad's experimental framework for blending traditional Machine Learning capabilities with high-performance Large Language Models.

---

## The Hybrid Concept

Traditional ML models (like XGBoost) are exceptionally fast and excel at parsing dense, structured financial metrics. On the other hand, LLMs can ingest unstructured contextual descriptors and perform nuanced semantic reasoning. 

Phase 3 tests whether a **collaborative hybrid system** can beat either model individually, focusing on improving the `Charged Off` minority class F1.

---

## Notebook Details

### Blended LLM & ML Modeling (`03_Blended_LLM_ML.ipynb`)
* **Model Configuration:** Evaluates `llama-3.3-70b-instruct` powered by **Groq** for high-throughput, low-latency execution.
* **Techniques Evaluated:**
  1. **Soft-Probability Ensembling:** Linearly blending XGBoost's numerical probabilities with the LLM's normalized logprobs (`prob_fully_paid`).
  2. **Confidence-Gated Routing:** Setting uncertainty boundaries. Clear, high-confidence credit applications are instantly routed to XGBoost (saving substantial API costs), while ambiguous boundary cases are routed to the LLM for deep rationalization.
  3. **Dual-Stage Risk Scoring:** Pipelining predictions where the classical model acts as a primary filter, and the LLM acts as a secondary verification gate.
* **Outputs:** 
  * `data/results/llm/03_blend_leaderboard.csv` — Comparative metrics of ensembling parameters.
  * `data/results/llm/03_locked_params.csv` — Serialized winning ensembling parameters.
  * Performance charts detailing F1 changes.

---

## Experimental Safety

> [!WARNING]  
> **Strict Test Set Holdout:** The test set (`test_batch.csv`) is strictly held out and never loaded in this directory to prevent label leakage. All parameters are tuned on the `tuning_sample` and strategy-selected on the `robustness_batch` only.
>
> The locked parameters in `03_locked_params.csv` are applied post-hoc to the test predictions during Phase 4 benchmarking.
