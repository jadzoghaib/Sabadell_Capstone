"""
Centralized, deterministic generation of the LLM sample batches.

Three role-based samples, one schema, mutually exclusive:

    tuning_sample.csv     development / tuning  (comparison, consistency,
                          reasoning effort, prompt variance, threshold tuning)
    robustness_batch.csv  robustness sanity-check (01c)
    test_batch.csv        FINAL test — touched once, in the benchmark (03)

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

SEED_TUNING, SEED_ROBUSTNESS, SEED_TEST = 42, 99, 2024
N_DEFAULT = 100

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
    """Load raw, filter to 2012–2014 binary-status loans, and derive the same
    feature columns as ml_models/02_Preprocessing — yielding the 35-col schema.
    Cached for the process (the raw .csv.gz is ~1GB)."""
    global _frame_cache
    if _frame_cache is not None:
        return _frame_cache

    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)[_RAW_COLS].copy()
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df = df[(df['issue_d'].dt.year >= 2012) & (df['issue_d'].dt.year <= 2014)].copy()
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    df['credit_history_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
    df['has_past_delinq'] = df['mths_since_last_delinq'].notna().astype(int)
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(999)
    df['emp_length'] = df['emp_length'].map(_EMP_LENGTH_MAP)

    _frame_cache = df.reindex(columns=_SCHEMA)
    return _frame_cache


def _test_pool_with_desc():
    """The canonical test split (random_state=42, test_size=0.33), restricted to
    rows with a borrower description — the universe every batch is drawn from."""
    from sklearn.model_selection import train_test_split
    _, test = train_test_split(_build_frame(), test_size=0.33, random_state=42)
    return test[test['desc'].notna() & (test['desc'].astype(str).str.strip() != '')]


def _keys(df):
    return set(zip(df['loan_amnt'], df['int_rate'], df['annual_inc']))


def _draw(n, seed, exclude):
    pool = _test_pool_with_desc()
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


def get_test_batch(force=False, n=N_DEFAULT):
    """FINAL test batch — used once, in the benchmark (03). Excludes both the
    tuning sample and the robustness batch, so it is genuinely unseen."""
    excl = _keys(get_tuning_sample()) | _keys(get_robustness_batch())
    return _cached(TEST_PATH, force, lambda: _draw(n, SEED_TEST, exclude=excl))


if __name__ == "__main__":
    for name, fn in [("tuning", get_tuning_sample),
                     ("robustness", get_robustness_batch),
                     ("test", get_test_batch)]:
        df = fn()
        co = int((df["loan_status"] == 0).sum())
        print(f"{name:11s} n={len(df)} cols={len(df.columns)} "
              f"ChargedOff={co} FullyPaid={len(df)-co}")
