"""
Centralized, deterministic generation of the LLM sample batches.

Three role-based samples, one schema, mutually exclusive:

    tuning_sample.csv     development / tuning  (comparison, consistency,
                          reasoning effort, prompt variance, threshold tuning)
    robustness_batch.csv  robustness sanity-check (01c)
    test_batch.csv        FINAL test — touched once, in the benchmark (03)

Plus the Phase-5 RAG retrieval corpus (get_rag_corpus): the full frame minus every
evaluation batch — the pool of labelled precedent loans the 05_rag notebooks retrieve
from. Same content-key exclusion, so no eval/test loan can ever be retrieved.

Design:
- **Load-if-exists** by default: calling get_*() never clobbers a committed
  sample, so re-running any notebook is safe. Pass force=True to regenerate.
- **Deterministic**: every draw is seeded; given the same raw CSV the rows are
  identical.
- **One schema**: all three carry the same 35 columns (incl. FICO, delinquency,
  inquiries, emp_length, credit_history_years) so run_ml_on_sample and the LLM
  see the SAME features on every batch. (The old held-out batch was missing the
  9 newer features — XGBoost was scoring it with FICO=0, etc.)
- **Mutually exclusive**: robustness excludes tuning; test excludes both
  (content key = loan_amnt + int_rate + annual_inc).

Raw data (data/raw/accepted_2007_to_2018Q4.csv.gz) is only needed when actually
generating; the committed CSVs cover the load path.
"""

from pathlib import Path
import pandas as pd

from llm_utils import DATA_DIR, RAW_DATA_PATH

# ── Paths & seeds ────────────────────────────────────────────────────────────
TUNING_PATH     = Path(DATA_DIR) / "tuning_sample.csv"
ROBUSTNESS_PATH = Path(DATA_DIR) / "robustness_batch.csv"
TEST_PATH       = Path(DATA_DIR) / "test_batch.csv"
RAG_CORPUS_PATH = Path(DATA_DIR) / "rag_corpus.csv"  # Phase-5 RAG retrieval corpus

SEED_TUNING, SEED_ROBUSTNESS, SEED_TEST = 42, 99, 2024
N_DEFAULT = 100
N_TEST = 1000  # Phase-4 final test set (large, for stable held-out + real P&L)

# Raw columns to pull (the full set — including the 9 features the old keep-list
# was missing). emp_title is loaded only to mirror 02_Preprocessing, then dropped.
_RAW_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'issue_d', 'loan_status', 'purpose', 'dti', 'earliest_cr_line', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
    'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'zip_code',
    'addr_state', 'desc', 'fico_range_low', 'fico_range_high', 'delinq_2yrs',
    'inq_last_6mths', 'mths_since_last_delinq', 'acc_open_past_24mths',
    # Realized cashflow columns — NOT model features. Used only by the Phase 4
    # financial analysis to compute ACTUAL portfolio P&L (money in/out) on the
    # held-out test set, instead of made-up portfolio assumptions.
    'total_pymnt', 'total_rec_prncp', 'total_rec_int', 'recoveries',
    'collection_recovery_fee', 'funded_amnt',
]

# Realized-cashflow columns appended to the schema (test set only in practice;
# tuning/robustness keep their committed 35-col files because they load-if-exists).
_CASHFLOW_COLS = [
    'total_pymnt', 'total_rec_prncp', 'total_rec_int', 'recoveries',
    'collection_recovery_fee', 'funded_amnt',
]

# Final 35-column schema (matches the committed tuning_sample exactly).
_SCHEMA = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'issue_d', 'loan_status', 'purpose', 'dti', 'earliest_cr_line', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
    'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'zip_code',
    'addr_state', 'desc', 'fico_range_low', 'fico_range_high', 'delinq_2yrs',
    'inq_last_6mths', 'mths_since_last_delinq', 'acc_open_past_24mths',
    'credit_history_years', 'has_past_delinq',
]

_EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10,
}

_frame_cache = None


# ── Generation internals ─────────────────────────────────────────────────────
def _build_frame():
    """Load raw and reproduce ml_models/02_Preprocessing **exactly** up to (and
    including) the dropna, so the train_test_split below partitions the SAME rows
    XGBoost trained/tested on.

    CRITICAL (fixed May 2026): the dropna MUST run before the split. An earlier
    version split the un-dropna'd frame, which — because train_test_split's
    shuffle depends on row count — produced a *different* partition than
    02_Preprocessing's. The result was that ~64% of the held-out batches were
    actually in XGBoost's training set (verified), silently inflating every
    XGBoost-vs-LLM comparison run on robustness_batch / test_batch. Reproducing
    the dropna here yields the exact 126,845-row test set, so the drawn batches
    are a true subset of XGBoost's held-out test data.

    Cached for the process (the raw .csv.gz is ~1GB)."""
    global _frame_cache
    if _frame_cache is not None:
        return _frame_cache

    df = pd.read_csv(RAW_DATA_PATH, low_memory=False, usecols=_RAW_COLS)
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df = df[(df['issue_d'].dt.year >= 2012) & (df['issue_d'].dt.year <= 2014)].copy()
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

    if df['int_rate'].dtype == object:
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
    df.loc[df['home_ownership'].isin(['ANY', 'NONE']), 'home_ownership'] = 'OTHER'

    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    df['credit_history_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
    df['has_past_delinq'] = df['mths_since_last_delinq'].notna().astype(int)
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(999)
    df['emp_length'] = df['emp_length'].map(_EMP_LENGTH_MAP)

    # Impute mort_acc by total_acc-group mean (mirrors 02_Preprocessing), so those
    # rows are NOT dropped by the dropna below — matching the training pipeline.
    ta_mean = df.groupby('total_acc')['mort_acc'].mean()
    miss = df['mort_acc'].isna()
    df.loc[miss, 'mort_acc'] = df.loc[miss, 'total_acc'].map(ta_mean).round()
    df['mort_acc'] = df['mort_acc'].fillna(round(ta_mean.mean()))

    # Drop rows with any NaN in the modelling columns (desc/cashflow excluded) —
    # this is the step that defines the row universe the split partitions.
    model_cols = [c for c in _SCHEMA if c != 'desc']
    df = df.dropna(subset=model_cols)

    _frame_cache = df.reindex(columns=_SCHEMA + _CASHFLOW_COLS)
    return _frame_cache


def _test_pool(require_desc=True):
    """XGBoost's exact held-out test set (dropna'd frame, random_state=42,
    test_size=0.33) — the universe every batch is drawn from, guaranteeing drawn
    batches are a true subset of XGBoost's test data (no train leakage).

    require_desc=True (legacy) restricts to rows with a borrower description,
    needed when a batch will be evaluated in the ``with_desc`` condition. The
    Phase-4 final test runs ``no_desc`` only (apples-to-apples vs XGBoost), so it
    passes require_desc=False to draw from the full split and avoid the selection
    bias of "borrowers who wrote a blurb".
    """
    from sklearn.model_selection import train_test_split
    _, test = train_test_split(_build_frame(), test_size=0.33, random_state=42)
    if require_desc:
        return test[test['desc'].notna() & (test['desc'].astype(str).str.strip() != '')]
    return test


def _keys(df):
    return set(zip(df['loan_amnt'], df['int_rate'], df['annual_inc']))


def _draw(n, seed, exclude, require_desc=True):
    pool = _test_pool(require_desc=require_desc)
    if exclude:
        mask = ~pool.apply(
            lambda r: (r['loan_amnt'], r['int_rate'], r['annual_inc']) in exclude, axis=1
        )
        pool = pool[mask]
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} eligible rows left; need {n}.")
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def _cached(path, force, generate):
    """Load-if-exists; otherwise generate, save, and return."""
    if path.exists() and not force:
        return pd.read_csv(path)
    batch = generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(path, index=False)
    print(f"Generated {path.name}: {len(batch)} rows -> {path}")
    return batch


# ── Public API ───────────────────────────────────────────────────────────────
def get_tuning_sample(force=False, n=N_DEFAULT):
    """Development / tuning sample. Produced by ml_models/02_Preprocessing
    (stratified to the natural class balance); the committed CSV is the source
    of truth. force=True draws a fresh n-row sample from the test pool."""
    return _cached(TUNING_PATH, force,
                   lambda: _draw(n, SEED_TUNING, exclude=set()))


def get_robustness_batch(force=False, n=N_DEFAULT):
    """Robustness sanity-check batch (01c). Excludes the tuning sample."""
    return _cached(ROBUSTNESS_PATH, force,
                   lambda: _draw(n, SEED_ROBUSTNESS, exclude=_keys(get_tuning_sample())))


def get_test_batch(force=False, n=N_TEST, require_desc=False):
    """FINAL test batch — touched once, in Phase 4. Excludes both the tuning
    sample and the robustness batch, so it is genuinely unseen.

    Defaults (changed May 2026): n=1000 and require_desc=False. The Phase-4
    final benchmark runs the no_desc condition only, so the test set is drawn
    from the FULL held-out split (representative of the real portfolio) rather
    than the desc-having subsample. Carries realized-cashflow columns for the
    actual-P&L financial analysis. Pass require_desc=True / n=100 to reproduce
    the legacy 100-loan desc-restricted batch."""
    excl = _keys(get_tuning_sample()) | _keys(get_robustness_batch())
    return _cached(TEST_PATH, force,
                   lambda: _draw(n, SEED_TEST, exclude=excl, require_desc=require_desc))


def _rag_corpus_pool():
    """(pool_df, source) for the RAG corpus: the full 2012-2014 frame when the raw
    .csv.gz is present, else a DEV FALLBACK built from the committed tuning_sample
    (disjoint from the robustness eval set), so Phase 5 runs before the raw file."""
    try:
        return _build_frame().reset_index(drop=True), "full_frame"
    except Exception:
        if TUNING_PATH.exists():
            return pd.read_csv(TUNING_PATH), "dev_fallback"
        raise


def get_rag_corpus(force=False, exclude_all_eval=True):
    """Phase-5 RAG retrieval corpus — labelled *precedent* loans = the full
    2012-2014 frame with every evaluation batch removed, so no eval/test loan can
    ever be retrieved as a precedent.

    The robustness_batch (the Phase-5 eval set) and the held-out test_batch are
    ALWAYS excluded; with exclude_all_eval=True the tuning_sample is dropped too
    when the source is the full frame. Dev fallback (raw .csv.gz absent) = the
    committed tuning_sample (~100 rows, disjoint from robustness). Load-if-exists;
    force=True rebuilds (and, where the raw file is present, swaps the dev fallback
    for the full large corpus). Leakage vs the eval set is asserted on every build."""
    def _generate():
        exclude = set()
        if ROBUSTNESS_PATH.exists():
            exclude |= _keys(pd.read_csv(ROBUSTNESS_PATH))   # the eval set — never a precedent
        if TEST_PATH.exists():
            exclude |= _keys(pd.read_csv(TEST_PATH))          # held-out; kept out for safety
        pool, source = _rag_corpus_pool()
        if source == "full_frame" and exclude_all_eval and TUNING_PATH.exists():
            exclude |= _keys(pd.read_csv(TUNING_PATH))
        mask = ~pool.apply(
            lambda r: (r['loan_amnt'], r['int_rate'], r['annual_inc']) in exclude, axis=1
        )
        corpus = (pool[mask]
                  .drop_duplicates(subset=['loan_amnt', 'int_rate', 'annual_inc'])
                  .reset_index(drop=True))
        if ROBUSTNESS_PATH.exists() and (_keys(corpus) & _keys(pd.read_csv(ROBUSTNESS_PATH))):
            raise AssertionError("RAG corpus overlaps the robustness eval set — leakage.")
        return corpus
    return _cached(RAG_CORPUS_PATH, force, _generate)


if __name__ == "__main__":
    for name, fn in [("tuning", get_tuning_sample),
                     ("robustness", get_robustness_batch),
                     ("test", get_test_batch),
                     ("rag_corpus", get_rag_corpus)]:
        df = fn()
        co = int((df["loan_status"] == 0).sum())
        print(f"{name:11s} n={len(df)} cols={len(df.columns)} "
              f"ChargedOff={co} FullyPaid={len(df)-co}")
