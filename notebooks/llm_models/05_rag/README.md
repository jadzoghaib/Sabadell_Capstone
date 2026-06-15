# 05 — RAG (Retrieval-Augmented) Credit Scoring

Ways to give the LLM **precedent loans** as evidence before it decides Fully Paid (1)
/ Charged Off (0), evaluated on the project's `robustness_batch` (the ~100-row validation
set). Per the **strict-holdout protocol**, the 1000-row `test_batch.csv` is reserved for
Phase 4 (`04_Final_Test_Analysis`) and is never loaded or evaluated here.

| Notebook | Approach | Inspiration |
|----------|----------|-------------|
| `00_Build_RAG_Dataset.ipynb` | Builds the retrieval corpus (the "RAG dataset") | — |
| `05a_RAG_Generative_SemanticID.ipynb` | **A** — Semantic IDs + multi-stage retrieval | the two papers |
| `05b_RAG_FullCorpus_Retrieval.ipynb`  | **B** — link everything except the eval batches | dense kNN RAG |
| `05c_RAG_Hybrid.ipynb`                | **C** — fuse A + B with RRF | hybrid (sparse/ID + dense) retrieval |

Run order: **`00` → then `05a` / `05b` / `05c`** (independent; run `05a` and `05b` before `05c`
if you want their rows to appear in `05c`'s leaderboard).

---

## The retrieval corpus — `data/processed/rag_corpus.csv`

The **full large dataset with every evaluation batch removed**, so no eval (or test) loan can ever
be retrieved (`rag_utils.assert_no_leakage` enforces this every run). The `robustness_batch` —
the set we evaluate on — is always excluded; the held-out `test_batch` is excluded too (read only
to keep it out of the corpus, never evaluated here).

`sample_generation.get_rag_corpus()` (centralized with the other batches — not in
`rag_utils`) picks its source automatically:

1. **Full corpus** — the full 2012–2014 LendingClub frame from
   `data/raw/accepted_2007_to_2018Q4.csv.gz` (via `sample_generation._build_frame`),
   minus the `test_batch` (1000 rows) **and** the `tuning`/`robustness` batches.
2. **Dev fallback** — if the raw `.csv.gz` is absent, it uses the committed
   `tuning_sample` (~100 rows, same schema, disjoint from the `robustness_batch` we evaluate on)
   so the whole pipeline runs today.

> **To get the real corpus:** drop `accepted_2007_to_2018Q4.csv.gz` into `data/raw/` and run
> `00` with `force=True`. The ~100-row dev corpus is only a stand-in for plumbing.

The `robustness_batch` rows (the eval set) and the `test_batch` rows are always excluded from the corpus.

---

## Approach A — Semantic IDs + multi-stage retrieval (`05a`)

- **TIGER** (NeurIPS 2023): each loan's content embedding is **residual-quantised into a
  hierarchical Semantic ID** (a tuple of codewords). Similar loans share codeword prefixes,
  so retrieval = match the prefix, then rank. `ResidualKMeansQuantizer` is a light,
  notebook-friendly stand-in for the paper's RQ-VAE.
- **RAG-FLARKO** (2025): **multi-stage retrieval** over two feature *views* —
  - *Stage 1 (behavioural, ≈ Personal KG):* borrower-view Semantic-ID retrieval → candidate pool.
  - *Stage 2 (loan/market, ≈ Market KG):* loan-view cosine re-rank **within that pool**.
  - Stage 2 is conditioned on Stage 1 (inter-stage context propagation); only a **compact
    precedent sub-context** is injected, not the whole corpus.

## Approach B — link everything except the eval batches (`05b`)

One flat dense index over the **entire corpus**; per evaluation loan, inject the top-`K` nearest
precedents by cosine similarity. No Semantic IDs, no staging — the simple counterpart to A.

## Approach C — hybrid: fuse A + B (`05c`)

A retrieves *behaviourally* similar precedents, B retrieves *globally* similar ones. `05c` runs
**both** and merges their rankings with **Reciprocal Rank Fusion** (`reciprocal_rank_fusion`,
constant `RRF_K=60`, optional `RRF_WEIGHTS`) before injecting the top-`K`. Mirrors how production
RAG combines an ID/structured retriever with a dense one.

**Section 4 is a built-in decision aid:** it measures how much A and B overlap. Low overlap (mean
Jaccard well under ~0.5) means they are complementary and the hybrid is the interesting bet; high
overlap means fusion adds little. The leaderboard pulls in the A/B rows from their summaries so all
five models (A, B, C, no-RAG, XGBoost) sit in one table.

---

## Config (top of each notebook)

```python
API_PROVIDER      = 'openai'                    # GPT-5.4 — the project's chosen model
MODEL_NAME        = 'gpt-5.4'
EMBEDDING_BACKEND = 'auto'                      # sentence-transformers if installed, else TF-IDF
K_PRECEDENTS      = 8        # precedents injected per loan
N_STAGE1          = 50       # stage-1 pool (Approach A only)
N_TEST            = None     # set e.g. 50 for a quick/cheap smoke run; None = all 100
```

> The first cell also sets `OMP_NUM_THREADS=1` / `KMP_DUPLICATE_LIB_OK=TRUE` before importing
> anything — torch (sentence-transformers) and XGBoost otherwise collide on macOS and segfault
> the kernel. Keep those lines at the very top.

- **Embeddings:** `EMBEDDING_BACKEND='auto'` uses `sentence-transformers` (faithful to the
  papers' content embeddings) when it's installed, and otherwise falls back to a TF-IDF + SVD
  embedding that needs no extra packages — so the notebook never hard-crashes. Run the install
  cell (`pip install sentence-transformers`, pulls torch) once to get the real backend; set
  `EMBEDDING_BACKEND='sentence-transformers'` to *require* it.
- **Model:** defaults to OpenAI GPT-5.4 (the project's chosen model), fanned out across every
  `OPENAI_API_KEY*` in `.env` via the `api_keys` + `max_workers` parallelism. Change the three
  config lines to use `nvidia`/`gemini` if you want a different provider.
- **Cost:** each notebook issues ~`len(test)` API calls for its RAG variant. The **no-RAG
  baseline is identical across 05a/05b/05c**, so it is computed **once** and recycled:
  `rag_utils.get_norag_baseline()` writes `05_norag_baseline_predictions.csv` on the first run
  and the other two notebooks load it with **no further API calls** (set `FORCE_RERUN=True` to
  recompute). Start with a small `N_TEST`; `use_cache=True` also makes per-notebook reruns cheap.

## Outputs (in `data/results/llm/`)

| File | Contents |
|------|----------|
| `05{a,b,c}_summary.csv` | accuracy / precision / recall / F1 (Charged Off) / AUC for RAG vs no-RAG LLM vs XGBoost (05c also lists A & B) |
| `05{a,b,c}_predictions.csv` | per-loan actual, RAG pred+prob, no-RAG pred+prob, XGBoost pred+prob |
| `05_norag_baseline_predictions.csv` | the shared no-RAG GPT-5.4 control, computed once and recycled by all three notebooks |
| `05{a,b,c}_comparison.png` | bar chart of the models |
| `05c_retriever_overlap.png` | histogram of A-vs-B precedent overlap (the hybrid decision aid) |
| `llm_calls.csv` | per-call tokens/cost, appended automatically by `llm_utils` |

## Files

- `rag_utils.py` — serialisation, `Embedder`, `ResidualKMeansQuantizer`
  (Semantic IDs), retrieval primitives (`cosine_topk`, `semantic_id_retrieve`,
  `multistage_retrieve`, `reciprocal_rank_fusion`), precedent/prompt builders, the leakage guard,
  and `get_norag_baseline()` (the shared no-RAG control — compute once, recycle).
