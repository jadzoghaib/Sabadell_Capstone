"""
Shared utilities for the two RAG (retrieval-augmented) credit-scoring notebooks.

Both notebooks predict Fully Paid (1) / Charged Off (0) on the project's
`robustness_batch` (the validation set), but instead of judging each applicant in
isolation they retrieve *precedent loans* (labelled past loans) and inject them as
evidence into the LLM. The held-out `test_batch` is reserved for Phase 4 only and is
never evaluated here (the strict-holdout protocol).

Two retrieval paradigms, inspired by the two papers:

  Approach A  (05a) — "papers" method
    * TIGER (NeurIPS 2023, Rajput et al.): content embeddings are quantised with a
      residual quantiser into a tuple of hierarchical codewords — a "Semantic ID".
      Similar loans share codeword prefixes, so retrieval = match the prefix then
      rank. We use residual k-means as a light, notebook-friendly stand-in for the
      paper's RQ-VAE.
    * RAG-FLARKO (Spadea & Seneviratne, 2025): multi-stage retrieval. A first
      "behavioural" stage (borrower view ≈ their Personal KG) conditions a second
      "loan/market" stage (loan view ≈ Market KG). Only a compact precedent
      sub-context is injected, not the whole corpus.

  Approach B  (05b) — "link everything except the test set"
    * One dense index over the ENTIRE corpus (all loans except the eval batches);
      per test loan, take the top-k nearest precedents and inject them.

Leakage safety
--------------
The retrieval corpus is the "RAG dataset" (`data/processed/rag_corpus.csv`), built
by `sample_generation.get_rag_corpus()` from the full 2012-2014 frame with every
evaluation batch removed. No eval (or test) loan can ever be retrieved.
`assert_no_leakage()` enforces this.

Paths/features are taken from `llm_utils` so the RAG notebooks see exactly the same
feature contract (`LLM_FEATURES`) and serialisation as the rest of the project.
"""

from __future__ import annotations

import os

# torch (sentence-transformers) and XGBoost each ship their own OpenMP runtime; on
# macOS loading both in one process segfaults. Pin to a single shared OpenMP thread
# before numpy/torch/xgboost initialise it. The notebooks set this too, but doing it
# here protects scripts that import rag_utils before llm_utils. setdefault = caller wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import the project's shared LLM utilities (one dir up) ───────────────────
_THIS = Path(__file__).resolve()
_LLM_MODELS_DIR = _THIS.parent.parent  # notebooks/llm_models
if str(_LLM_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_MODELS_DIR))

import llm_utils  # noqa: E402
from llm_utils import (  # noqa: E402
    LLM_FEATURES,
    FEATURE_DESCRIPTIONS,
    format_loan_features,
)

# Same content key the project uses to keep its batches mutually exclusive.
# The retrieval corpus itself is built by sample_generation.get_rag_corpus().
KEY_COLS = ["loan_amnt", "int_rate", "annual_inc"]

# Two "views" for the multi-stage (RAG-FLARKO) retrieval. Borrower view ≈ the
# Personal KG (who the borrower is); loan view ≈ the Market KG (the instrument).
# Only features present in LLM_FEATURES are kept, so this tracks the contract.
BORROWER_FEATURES = [
    f for f in [
        "annual_inc", "emp_length", "home_ownership", "dti", "credit_history_years",
        "earliest_cr_line", "delinq_2yrs", "has_past_delinq", "mths_since_last_delinq",
        "pub_rec", "pub_rec_bankruptcies", "open_acc", "total_acc",
        "acc_open_past_24mths", "revol_bal", "revol_util", "inq_last_6mths",
        "fico_range_low", "fico_range_high",
    ] if f in LLM_FEATURES
]
LOAN_FEATURES = [
    f for f in [
        "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
        "purpose", "verification_status", "application_type", "initial_list_status",
    ] if f in LLM_FEATURES
]


# ── Keys / leakage ──────────────────────────────────────────────────────────
def content_keys(df: pd.DataFrame) -> set:
    """Set of (loan_amnt, int_rate, annual_inc) tuples identifying loans."""
    return set(zip(*(df[c] for c in KEY_COLS)))


def assert_no_leakage(corpus: pd.DataFrame, test: pd.DataFrame) -> None:
    """Hard guarantee that no test loan sits in the retrieval corpus."""
    overlap = content_keys(corpus) & content_keys(test)
    if overlap:
        raise AssertionError(
            f"LEAKAGE: {len(overlap)} eval loan(s) found in the RAG corpus. "
            "Rebuild the corpus with sample_generation.get_rag_corpus(force=True)."
        )


# ── Serialisation: loan -> text (reuses the project's feature formatting) ───
def serialize_loan(row, features=None) -> str:
    """Human-readable text for a loan, restricted to `features` if given.
    Built on llm_utils.format_loan_features so the wording matches the prompts."""
    if features is None:
        return format_loan_features(row, include_desc=False)
    lines = []
    for feat in features:
        if feat in row and pd.notna(row[feat]):
            label = FEATURE_DESCRIPTIONS.get(feat, feat)
            lines.append(f"- {label}: {row[feat]}")
    return "\n".join(lines)


def serialize_frame(df: pd.DataFrame, features=None) -> list[str]:
    return [serialize_loan(r, features=features) for _, r in df.iterrows()]


# ── Embeddings (default: sentence-transformers; TF-IDF fallback) ────────────
class Embedder:
    """Uniform embed interface over a chosen backend.

    backend:
      "sentence-transformers" — content embeddings (faithful to TIGER's Sentence-T5).
      "tfidf"                 — TfidfVectorizer + TruncatedSVD (offline, light).
      "auto"                  — sentence-transformers, falling back to tfidf on
                                ImportError so notebooks never hard-crash.
    All outputs are L2-normalised, so a dot product is cosine similarity.
    """

    def __init__(self, backend: str = "auto",
                 st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 svd_dim: int = 128, random_state: int = 42):
        self.backend = backend
        self.st_model_name = st_model
        self.svd_dim = svd_dim
        self.random_state = random_state
        self._st = None
        self._tfidf = None
        self._svd = None
        self._resolved = None  # the backend actually used after fit

    # -- sentence-transformers --
    def _load_st(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.st_model_name)
        return self._st

    def fit(self, corpus_texts):
        corpus_texts = list(corpus_texts)
        if self.backend in ("auto", "sentence-transformers"):
            try:
                self._load_st()
                self._resolved = "sentence-transformers"
                return self
            except Exception as e:
                if self.backend == "sentence-transformers":
                    raise
                warnings.warn(
                    f"sentence-transformers unavailable ({type(e).__name__}); "
                    "falling back to TF-IDF + SVD embeddings.", stacklevel=2)
        # tfidf fallback / explicit
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self._tfidf.fit_transform(corpus_texts)
        dim = min(self.svd_dim, max(2, min(X.shape) - 1))
        self._svd = TruncatedSVD(n_components=dim, random_state=self.random_state)
        self._svd.fit(X)
        self._resolved = "tfidf"
        return self

    def transform(self, texts) -> np.ndarray:
        texts = list(texts)
        if self._resolved is None:
            raise RuntimeError("Embedder.fit must be called before transform.")
        if self._resolved == "sentence-transformers":
            emb = np.asarray(self._st.encode(
                texts, normalize_embeddings=True, show_progress_bar=False))
            return emb.astype(np.float32)
        X = self._tfidf.transform(texts)
        emb = self._svd.transform(X).astype(np.float32)
        return _l2_normalize(emb)

    def fit_transform(self, corpus_texts) -> np.ndarray:
        corpus_texts = list(corpus_texts)
        self.fit(corpus_texts)
        return self.transform(corpus_texts)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ── TIGER-style Semantic IDs via residual k-means quantisation ──────────────
class ResidualKMeansQuantizer:
    """A light residual quantiser standing in for the paper's RQ-VAE.

    At each level a k-means codebook quantises the current residual; the residual
    is then updated by subtracting the chosen centroid (coarse-to-fine). The tuple
    of per-level codeword indices is the item's hierarchical Semantic ID, so loans
    that are close in embedding space share a codeword prefix.
    """

    def __init__(self, n_levels: int = 3, codebook_size: int = 8,
                 random_state: int = 42):
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.codebooks_ = []  # list of fitted KMeans

    def fit(self, X: np.ndarray):
        from sklearn.cluster import KMeans
        residual = np.asarray(X, dtype=np.float32).copy()
        self.codebooks_ = []
        for level in range(self.n_levels):
            k = min(self.codebook_size, len(residual))
            km = KMeans(n_clusters=k, n_init=10,
                        random_state=self.random_state + level).fit(residual)
            self.codebooks_.append(km)
            residual = residual - km.cluster_centers_[km.labels_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        residual = np.asarray(X, dtype=np.float32).copy()
        codes = []
        for km in self.codebooks_:
            c = km.predict(residual)
            codes.append(c)
            residual = residual - km.cluster_centers_[c]
        return np.stack(codes, axis=1)

    def semantic_ids(self, X: np.ndarray) -> list[tuple]:
        """List of codeword tuples (the Semantic IDs)."""
        return [tuple(int(v) for v in row) for row in self.transform(X)]


# ── Retrieval primitives ────────────────────────────────────────────────────
def cosine_topk(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int,
                candidate_idx: np.ndarray | None = None):
    """Top-k by cosine similarity (embeddings assumed L2-normalised).
    Restrict the search to `candidate_idx` when given. Returns (idx, sims)."""
    if candidate_idx is None:
        sims = corpus_emb @ query_emb
        order = np.argsort(-sims)[:k]
        return order, sims[order]
    cand = np.asarray(candidate_idx)
    sims = corpus_emb[cand] @ query_emb
    order = np.argsort(-sims)[:k]
    return cand[order], sims[order]


def semantic_id_retrieve(query_emb, query_sid, corpus_emb, corpus_sids, k,
                         min_pool: int = 25):
    """TIGER-style hierarchical retrieval: take the longest Semantic-ID prefix that
    still yields a reasonable candidate pool, then cosine-rank inside it. Falls back
    to the whole corpus if even the coarsest prefix is too small. Returns (idx,sims).
    """
    cand = None
    for depth in range(len(query_sid), 0, -1):
        prefix = query_sid[:depth]
        c = [i for i, s in enumerate(corpus_sids) if s[:depth] == prefix]
        if len(c) >= max(k, min_pool):
            cand = np.asarray(c)
            break
    if cand is None or len(cand) < k:
        cand = np.arange(len(corpus_sids))
    return cosine_topk(query_emb, corpus_emb, k, candidate_idx=cand)


def reciprocal_rank_fusion(rankings, k: int = 60, weights=None, top_n: int | None = None):
    """Fuse several ranked candidate lists into one (Reciprocal Rank Fusion).

    Each item's fused score is sum over retrievers of w / (k + rank), with rank
    starting at 1, so high ranks in *either* list lift an item. The standard
    constant k=60 damps the contribution of deep ranks. This is the combination
    used in Approach C to merge Approach A (Semantic-ID multi-stage) with
    Approach B (full-corpus kNN).

    Args:
        rankings: list of sequences of corpus indices, each in descending order.
        k: RRF constant.
        weights: optional per-ranking weights (defaults to all 1.0).
        top_n: if set, return only the top-n fused indices.

    Returns:
        list of corpus indices (ints) sorted by fused score, descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, idx in enumerate(ranking, start=1):
            i = int(idx)
            scores[i] = scores.get(i, 0.0) + w / (k + rank)
    fused = sorted(scores, key=lambda i: -scores[i])
    return fused if top_n is None else fused[:top_n]


def multistage_retrieve(borrower_query, loan_query, borrower_corpus, loan_corpus,
                        k: int, n_stage1: int = 50, mode: str = "multistage"):
    """RAG-FLARKO-style two-stage retrieval over two feature "views".

    mode="multistage": Stage 1 retrieves the `n_stage1` behaviourally closest
      precedents (borrower view ≈ Personal KG); Stage 2 re-ranks *within that pool*
      by loan-view similarity (loan/market view ≈ Market KG). Stage 2 is thus
      conditioned on Stage 1 — the paper's inter-stage context propagation.
    mode="parallel": the ablation baseline — rank the loan view over the whole
      corpus independently of the borrower stage.

    Returns (idx, sims) of length k.
    """
    if mode == "parallel":
        return cosine_topk(loan_query, loan_corpus, k)
    s1_idx, _ = cosine_topk(borrower_query, borrower_corpus, min(n_stage1, len(borrower_corpus)))
    return cosine_topk(loan_query, loan_corpus, k, candidate_idx=s1_idx)


# ── Building the injected precedent context + the RAG prompts ───────────────
def build_precedent_block(neighbors: pd.DataFrame, sims=None,
                          features=None, header: str | None = None) -> str:
    """Compact, labelled precedent block for injection. Each precedent shows its
    features and its KNOWN outcome (the retrieval signal the LLM should weigh)."""
    head = header or (f"RETRIEVED PRECEDENTS — {len(neighbors)} similar past loans "
                      "with known outcomes:")
    lines = [head, ""]
    for rank, (_, r) in enumerate(neighbors.iterrows(), 1):
        outcome = "Fully Paid" if int(r["loan_status"]) == 1 else "Charged Off"
        sim_txt = f"  [similarity {sims[rank - 1]:.2f}]" if sims is not None else ""
        lines.append(f"Precedent {rank} — OUTCOME: {outcome}{sim_txt}")
        lines.append(serialize_loan(r, features=features))
        lines.append("")
    return "\n".join(lines).rstrip()


def build_rag_system_prompt() -> str:
    """The project's baseline system prompt, extended with precedent-usage guidance
    (and a guard against naively copying a precedent's label)."""
    base = llm_utils.build_system_prompt()
    addition = (
        "\n\nRETRIEVED EVIDENCE: You are given several PRECEDENT loans retrieved "
        "from a historical database, each with its known outcome (Fully Paid / "
        "Charged Off). Treat them as evidence: compare the applicant to the "
        "precedents and let the balance of similar outcomes inform — but not "
        "dictate — your decision. Judge the applicant on their own features; do "
        "NOT copy a precedent's outcome just because a few fields match."
    )
    return base + addition


def make_rag_prompt_fn(test_df: pd.DataFrame, precedent_blocks: list[str]):
    """Attach a precomputed precedent block to each test row and return a
    (modified_df, user_prompt_fn) pair ready for llm_utils.run_llm_experiment.

    The prompt is stored on the row (column `_rag_prompt`) so retrieval happens
    once, up front, and the experiment loop — which may run rows concurrently —
    just reads it back. Extra columns are ignored by run_ml_on_sample and the
    feature formatter, so this is safe to pass straight through.
    """
    df = test_df.reset_index(drop=True).copy()
    prompts = []
    for i, (_, row) in enumerate(df.iterrows()):
        base = llm_utils.build_user_prompt(row, include_desc=False)
        prompts.append(f"{precedent_blocks[i]}\n\n{'-' * 60}\n\n{base}")
    df["_rag_prompt"] = prompts

    def user_prompt_fn(row):
        return row["_rag_prompt"]

    return df, user_prompt_fn
