"""
Shared utilities for LLM evaluation notebooks.

Handles: data loading, ML re-encoding/prediction, prompt building,
LLM API calls, evaluation metrics, and full experiment loops.
"""

import json
import math
import os
import re
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score, roc_curve
)

from llm_pricing import get_price

def _get_cache_path(notebook_id, label, desc_tag):
    import re
    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', label)
    filename = f"{notebook_id}_{safe_label}_{desc_tag}.json"
    cache_dir = os.path.join(RESULTS_DIR, "cache")
    return os.path.join(cache_dir, filename)

# Anchor every project path to the repo root so notebooks can live at any depth
# under notebooks/ without breaking I/O. `__file__` is llm_models/llm_utils.py;
# parent.parent is notebooks/; parent.parent.parent is the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = str(_PROJECT_ROOT / "data" / "processed")
MODEL_DIR    = str(_PROJECT_ROOT / "models")
RESULTS_DIR  = str(_PROJECT_ROOT / "data" / "results" / "llm")
RAW_DATA_PATH = str(_PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz")

# ── Features the LLM sees (exclude target, desc, and non-predictive columns) ──
LLM_FEATURES = [
    # Original 21
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose',
    'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'initial_list_status', 'application_type',
    'mort_acc', 'pub_rec_bankruptcies',
    # Added for symmetry with the ML feature set (FICO, recent delinquency,
    # recent inquiries, credit-seeking behaviour, employment, history length).
    'fico_range_low', 'fico_range_high',
    'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq',
    'has_past_delinq', 'acc_open_past_24mths',
    'emp_length', 'credit_history_years',
]

FEATURE_DESCRIPTIONS = {
    'loan_amnt': 'Loan amount requested ($)',
    'term': 'Loan term',
    'int_rate': 'Interest rate (%)',
    'installment': 'Monthly payment ($)',
    'grade': 'LC assigned loan grade',
    'sub_grade': 'LC assigned loan sub-grade',
    'home_ownership': 'Home ownership status',
    'annual_inc': 'Annual income ($)',
    'verification_status': 'Income verification status',
    'purpose': 'Stated loan purpose',
    'dti': 'Debt-to-income ratio',
    'earliest_cr_line': 'Earliest credit line date',
    'open_acc': 'Number of open credit accounts',
    'pub_rec': 'Number of derogatory public records',
    'revol_bal': 'Revolving balance ($)',
    'revol_util': 'Revolving utilization rate (%)',
    'total_acc': 'Total number of credit accounts',
    'initial_list_status': 'Initial listing status (w=whole, f=fractional)',
    'application_type': 'Individual or joint application',
    'mort_acc': 'Number of mortgage accounts',
    'pub_rec_bankruptcies': 'Number of public-record bankruptcies',
    # Added (matches the ML feature set):
    'fico_range_low': 'FICO credit score (lower bound, 300-850)',
    'fico_range_high': 'FICO credit score (upper bound, 300-850)',
    'delinq_2yrs': 'Number of 30+ day delinquencies in the last 2 years',
    'inq_last_6mths': 'Number of credit inquiries in the last 6 months',
    'mths_since_last_delinq': 'Months since most recent delinquency',
    'has_past_delinq': 'Has any delinquency on record (1=yes, 0=no)',
    'acc_open_past_24mths': 'Number of accounts opened in the last 24 months',
    'emp_length': 'Employment length in years (0-10, where 10 = 10+ years)',
    'credit_history_years': 'Length of credit history at loan issue (years)',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_llm_sample():
    """Load the 100-row human-readable LLM evaluation sample."""
    df = pd.read_csv(f"{DATA_DIR}/tuning_sample.csv")
    return df


_LENDINGCLUB_KEEP_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
    'loan_status', 'purpose', 'dti', 'earliest_cr_line', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'initial_list_status', 'application_type', 'mort_acc',
    'pub_rec_bankruptcies', 'zip_code', 'addr_state', 'desc',
]
_lendingclub_cache = None


def _load_lendingclub_filtered():
    """Load 02_Preprocessing's source frame: 2012–2014, binary status only,
    `loan_status` mapped to 1/0. Cached for the process — `build_few_shot_examples`
    consumes this and the raw .csv.gz is ~1GB."""
    global _lendingclub_cache
    if _lendingclub_cache is not None:
        return _lendingclub_cache

    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    df = df[_LENDINGCLUB_KEEP_COLS].copy()
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df = df[(df['issue_d'].dt.year >= 2012) & (df['issue_d'].dt.year <= 2014)]
    df['loan_status'] = df.loan_status.map({'Fully Paid': 1, 'Charged Off': 0})
    _lendingclub_cache = df
    return df


def _train_test_split_canonical(df):
    """The same split 02_Preprocessing uses (random_state=42, test_size=0.33).
    Anything that needs to talk about train/test rows must use this split or
    risk leakage between the LLM sample and the train rows."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=0.33, random_state=42)


# NOTE: held-out batch generation moved to sample_generation.py
# (get_robustness_batch / get_test_batch). load_llm_sample() above still loads
# the tuning sample; _load_lendingclub_filtered / _train_test_split_canonical
# remain for build_few_shot_examples.


# ── XGBoost prediction on LLM sample rows ───────────────────────────────────

def encode_for_ml(llm_sample):
    """
    Re-encode and scale an LLM sample exactly as 02_Preprocessing did, and load
    the trained XGBoost artefacts. Single source of truth for the ML preprocessing
    so run_ml_on_sample and SHAP/explainability share identical encoding.

    Returns (X_scaled_df, model, threshold):
      X_scaled_df : DataFrame of scaled features, columns = feature_cols (named,
                    in training order) — ready for model.predict_proba or SHAP.
      model       : the loaded XGBoost classifier.
      threshold   : the tuned decision threshold (thresholds['xgb']).
    """
    import joblib

    scaler = joblib.load(f"{DATA_DIR}/02_scaler.joblib")
    feature_cols = joblib.load(f"{DATA_DIR}/02_feature_columns.joblib")
    model = joblib.load(f"{MODEL_DIR}/xgb_model.joblib")
    thresholds = joblib.load(f"{MODEL_DIR}/thresholds.joblib")
    threshold = thresholds['xgb']

    df = llm_sample.copy()

    # Apply same preprocessing as 02_Preprocessing.
    term_values = {' 36 months': 36, ' 60 months': 60}
    if not pd.api.types.is_numeric_dtype(df['term']):
        df['term'] = df['term'].map(term_values)

    # Ordinal-encode emp_length to match preprocessing.
    if 'emp_length' in df.columns and not pd.api.types.is_numeric_dtype(df['emp_length']):
        emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10,
        }
        df['emp_length'] = df['emp_length'].map(emp_length_map)

    # mths_since_last_delinq sentinel + flag (mirrors 02_Preprocessing).
    if 'mths_since_last_delinq' in df.columns:
        df['has_past_delinq'] = df['mths_since_last_delinq'].notna().astype(int)
        df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(999)

    # Compute credit_history_years (years between earliest credit line and
    # loan issue) before dropping the date columns.
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        issue_d = pd.to_datetime(df['issue_d'])
        ecl     = pd.to_datetime(df['earliest_cr_line'])
        df['credit_history_years'] = (issue_d - ecl).dt.days / 365.25

    df.drop(columns=['grade', 'zip_code', 'addr_state', 'issue_d',
                     'earliest_cr_line', 'desc'],
            errors='ignore', inplace=True)

    dummies = ['sub_grade', 'verification_status', 'purpose',
               'initial_list_status', 'application_type', 'home_ownership']
    df = pd.get_dummies(df, columns=dummies, drop_first=True)

    X = df.drop(columns=['loan_status'], errors='ignore')
    X = X.reindex(columns=feature_cols, fill_value=0)

    X_scaled = scaler.transform(X).astype(np.float32)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=llm_sample.index)
    return X_scaled_df, model, threshold


def run_ml_on_sample(llm_sample):
    """
    Re-encode and scale the LLM sample, then run the XGBoost model.
    Returns (xgb_probs, xgb_preds) as arrays.
    """
    X_scaled_df, model, threshold = encode_for_ml(llm_sample)
    probs = model.predict_proba(X_scaled_df.values)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds


def top_xgb_features(n=8):
    """Return the `n` most important raw LLM features by XGBoost importance.

    The model is trained on one-hot-encoded columns (sub_grade_B2, purpose_car,
    …); this aggregates each dummy's importance back onto its source feature so
    the result is a subset of LLM_FEATURES suitable for prompt construction.
    Derived live from models/xgb_model.joblib — never hard-code the list, since
    the feature set has changed (FICO etc. were added)."""
    import joblib

    model = joblib.load(f"{MODEL_DIR}/xgb_model.joblib")
    feature_cols = joblib.load(f"{DATA_DIR}/02_feature_columns.joblib")
    importances = model.feature_importances_

    agg = {f: 0.0 for f in LLM_FEATURES}
    for col, w in zip(feature_cols, importances):
        if col in agg:                      # numeric feature — exact match
            agg[col] += float(w)
            continue
        # one-hot dummy: attribute to its source feature. Longest prefix wins so
        # e.g. a 'pub_rec_bankruptcies' dummy never gets folded into 'pub_rec'.
        cands = [f for f in LLM_FEATURES if col.startswith(f + "_")]
        if cands:
            agg[max(cands, key=len)] += float(w)

    return sorted(agg, key=agg.get, reverse=True)[:n]


# ── Prompt building ───────────────────────────────────────────────────────────

def format_loan_features(row, include_desc=False):
    """Format a single loan's features as a readable string for the LLM."""
    lines = []
    for feat in LLM_FEATURES:
        if feat not in row or pd.isna(row[feat]):
            continue
        label = FEATURE_DESCRIPTIONS.get(feat, feat)
        val = row[feat]

        # `mths_since_last_delinq == 999` is the sentinel encoding for "no
        # delinquency on record" (set during preprocessing). Show that
        # explicitly so the LLM doesn't try to interpret 999 as a real count.
        if feat == 'mths_since_last_delinq' and val == 999:
            val = 'no delinquency on record'

        lines.append(f"- {label}: {val}")

    if include_desc and 'desc' in row and pd.notna(row['desc']):
        lines.append(f"- Borrower description: {row['desc']}")

    return "\n".join(lines)


def build_system_prompt():
    """System prompt for LLM loan default prediction."""
    return (
        "You are a credit risk analyst. Given a loan application's features, "
        "predict whether the borrower will fully repay the loan or default "
        "(charge off).\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"prediction": <1 or 0>, "reasoning": "<brief explanation>"}\n\n'
        "Where:\n"
        "- prediction: 1 = Fully Paid, 0 = Charged Off\n"
        "- reasoning: 1-2 sentence explanation of your prediction"
    )


def build_user_prompt(row, include_desc=False):
    """Build the user prompt for a single loan prediction."""
    features = format_loan_features(row, include_desc=include_desc)
    return f"Predict the outcome for this loan application:\n\n{features}"


def build_few_shot_examples(n_examples=5, random_state=42):
    """
    Build few-shot examples from the training set (not the LLM sample).
    Returns a string with labeled examples.
    """
    train_data, _ = _train_test_split_canonical(_load_lendingclub_filtered())

    n_per_class = n_examples // 2
    paid = train_data[train_data.loan_status == 1].sample(n=n_per_class, random_state=random_state)
    default = train_data[train_data.loan_status == 0].sample(
        n=n_examples - n_per_class, random_state=random_state
    )
    examples = pd.concat([paid, default]).sample(frac=1, random_state=random_state)

    lines = []
    for i, (_, row) in enumerate(examples.iterrows(), 1):
        features = format_loan_features(row, include_desc=True)
        outcome = "Fully Paid" if row['loan_status'] == 1 else "Charged Off"
        lines.append(f"Example {i}:\n{features}\nOutcome: {outcome}\n")

    return "\n".join(lines)


# ── LLM API call ──────────────────────────────────────────────────────────────

def _load_env():
    """Load .env files into os.environ. Checks both the llm_models folder and
    the repo root so provider keys can live in either location.
    override=True so keys in .env win over any empty env-var shells inherit
    (e.g. ANTHROPIC_API_KEY='' from system config)."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
    load_dotenv(_PROJECT_ROOT / ".env", override=True)


def load_api_key(api_provider, model=None):
    """
    Load API key from .env. Supports per-model keys for Gemini.

    Lookup order for gemini:
      1. GEMINI_API_KEY_FLASH / GEMINI_API_KEY_PRO (if model contains 'flash'/'pro')
      2. GEMINI_API_KEY (fallback)
    """
    import os
    _load_env()

    if api_provider == "gemini" and model:
        model_lower = model.lower()
        if "flash" in model_lower:
            key = os.environ.get("GEMINI_API_KEY_FLASH")
            if key:
                return key
        elif "pro" in model_lower:
            key = os.environ.get("GEMINI_API_KEY_PRO")
            if key:
                return key
        # Fallback
        return os.environ.get("GEMINI_API_KEY")

    key_map = {
        "gemini":    "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "nvidia":    "NVIDIA_API_KEY",
        "groq":      "GROQ_API_KEY",
    }
    env_var = key_map.get(api_provider)
    return os.environ.get(env_var) if env_var else None


def load_all_api_keys(api_provider):
    """Discover ALL numbered API keys for a provider from .env.

    Scans for the base key (e.g. OPENAI_API_KEY) plus every numbered variant
    (OPENAI_API_KEY_2, OPENAI_API_KEY_3, …, up to _50) and returns a
    deduplicated list preserving discovery order. Empty/missing keys are
    silently skipped.

    Example .env layout::

        OPENAI_API_KEY=sk-abc
        OPENAI_API_KEY_2=sk-def
        OPENAI_API_KEY_3=sk-ghi

    >>> load_all_api_keys("openai")
    ['sk-abc', 'sk-def', 'sk-ghi']

    Falls back to [load_api_key(provider)] if no numbered keys exist, so
    callers always get a non-empty list (or a list with one None if nothing
    is configured at all — same behaviour as before).
    """
    import os
    _load_env()

    _BASE_MAP = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "nvidia":    "NVIDIA_API_KEY",
        "groq":      "GROQ_API_KEY",
        "gemini":    "GEMINI_API_KEY",
    }
    base_var = _BASE_MAP.get(api_provider)
    if base_var is None:
        return [load_api_key(api_provider)]

    seen = set()
    keys = []

    def _add(val):
        if val and val not in seen:
            seen.add(val)
            keys.append(val)

    # Base key (no suffix)
    _add(os.environ.get(base_var))

    # Numbered variants: _2, _3, …, _50
    for i in range(2, 51):
        _add(os.environ.get(f"{base_var}_{i}"))

    # Gemini has model-specific keys too — include those
    if api_provider == "gemini":
        _add(os.environ.get("GEMINI_API_KEY_PRO"))
        _add(os.environ.get("GEMINI_API_KEY_FLASH"))

    return keys or [load_api_key(api_provider)]


_USAGE_ATTRS = {
    "gemini":    ("usage_metadata", "prompt_token_count", "candidates_token_count"),
    "openai":    ("usage",          "prompt_tokens",      "completion_tokens"),
    "nvidia":    ("usage",          "prompt_tokens",      "completion_tokens"),
    "anthropic": ("usage",          "input_tokens",       "output_tokens"),
    "groq":      ("usage",          "prompt_tokens",      "completion_tokens"),
}
_USAGE_WARNED = set()


def _extract_usage(api_provider, response):
    """Pull (input_tokens, output_tokens) off a provider's response object.
    Returns (0, 0) and warns once per provider if the shape is unexpected,
    so cost rows are still written but the user knows the data is missing."""
    spec = _USAGE_ATTRS.get(api_provider)
    if spec is None:
        return 0, 0
    container, in_attr, out_attr = spec
    try:
        u = getattr(response, container, None)
        if u is None:
            raise AttributeError(f"{container} is None")
        return int(getattr(u, in_attr, 0) or 0), int(getattr(u, out_attr, 0) or 0)
    except Exception as e:
        if api_provider not in _USAGE_WARNED:
            _USAGE_WARNED.add(api_provider)
            warnings.warn(
                f"Could not extract token usage from {api_provider} response ({e}); "
                f"cost rows for this provider will record 0 tokens.",
                stacklevel=2,
            )
        return 0, 0


_LOGPROBS_WARNED = set()
_REASONING_EFFORT_WARNED = set()
_PREDICTION_RE = re.compile(r'"prediction"\s*:\s*([01])')


def find_best_threshold(y_true, probs):
    """Find the threshold that maximises F1 for the minority class
    (Charged Off = 0). Generic — works on any (y_true, probs) pair from any
    model. Returns (threshold, precision, recall, f1) for the minority class."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    recall_co = 1 - fpr
    n_co = (y_true == 0).sum()
    n_fp = (y_true == 1).sum()
    tp_co = recall_co * n_co
    fp_co = (1 - tpr) * n_fp
    precision_co = tp_co / (tp_co + fp_co + 1e-8)
    f1_co = 2 * (precision_co * recall_co) / (precision_co + recall_co + 1e-8)
    best_idx = int(np.argmax(f1_co))
    return (float(thresholds[best_idx]), float(precision_co[best_idx]),
            float(recall_co[best_idx]),  float(f1_co[best_idx]))


def _extract_token_logprobs(api_provider, response):
    """Return [(token_text, [(alt_token, alt_logprob), ...]), ...] for the
    output, or None if logprobs aren't available. Anthropic doesn't expose them
    so this always returns None for that provider."""
    try:
        if api_provider == "openai":
            content = getattr(getattr(response.choices[0], "logprobs", None), "content", None)
            if not content:
                return None
            return [(t.token, [(tl.token, tl.logprob) for tl in (t.top_logprobs or [])])
                    for t in content]
        if api_provider == "gemini":
            lp = getattr(response.candidates[0], "logprobs_result", None)
            if lp is None:
                return None
            chosen = list(getattr(lp, "chosen_candidates", []) or [])
            tops = list(getattr(lp, "top_candidates", []) or [])
            out = []
            for i, c in enumerate(chosen):
                tok_text = getattr(c, "token", "")
                alts = []
                if i < len(tops):
                    for x in (getattr(tops[i], "candidates", []) or []):
                        alts.append((getattr(x, "token", ""),
                                     getattr(x, "log_probability", 0.0)))
                out.append((tok_text, alts))
            return out
    except Exception:
        return None
    return None


def _extract_prediction_prob(api_provider, raw_text, token_data):
    """Find the `"prediction": <0|1>` token in the response and return the
    normalised P(prediction=1), i.e. P("1") / (P("1") + P("0")). Returns None
    if logprobs are missing or the prediction token can't be located."""
    if not token_data:
        return None
    m = _PREDICTION_RE.search(raw_text or "")
    if not m:
        return None
    digit_pos = m.start(1)

    pos = 0
    target = None
    for tok, alts in token_data:
        next_pos = pos + len(tok)
        if pos <= digit_pos < next_pos:
            target = (tok, alts)
            break
        pos = next_pos

    if target is None:
        return None
    _, alts = target
    p_one = 0.0
    p_zero = 0.0
    for cand_tok, cand_lp in alts:
        cs = (cand_tok or "").lstrip()
        if not cs:
            continue
        if cs[0] == "1":
            p_one += math.exp(cand_lp)
        elif cs[0] == "0":
            p_zero += math.exp(cand_lp)

    total = p_one + p_zero
    if total <= 0:
        if api_provider not in _LOGPROBS_WARNED:
            _LOGPROBS_WARNED.add(api_provider)
            warnings.warn(
                f"Logprobs for {api_provider} located the prediction token but "
                "found no '0' or '1' alternatives in top_logprobs. Increase "
                "top_logprobs (currently set to 5) or check provider response.",
                stacklevel=2,
            )
        return None
    return p_one / total


def call_llm(system_prompt, user_prompt, api_provider="gemini", model=None,
             api_key=None, return_usage=False, with_logprobs=False, max_tokens=None,
             reasoning_effort=None, temperature=None, seed=None):
    """
    Call the LLM API. Supports gemini, anthropic, openai, and groq providers.

    Args:
        system_prompt: System message
        user_prompt: User message
        api_provider: "gemini", "anthropic", or "openai"
        model: Model name (if None, uses default for provider)
        api_key: API key (if None, reads from .env / environment)
        return_usage: if True, return (text, meta) where `meta` includes
                      input_tokens, output_tokens, and (when with_logprobs=True)
                      prob_fully_paid in [0, 1] or None.
        with_logprobs: ask the provider for top-K token logprobs and use them
                       to compute P(prediction=1). Anthropic doesn't expose
                       logprobs and is silently skipped (prob_fully_paid=None).

    Returns:
        Raw response text, or (text, meta_dict) if return_usage=True.
    """
    if api_key is None:
        api_key = load_api_key(api_provider, model=model)

    # reasoning_effort is OpenAI-specific (gpt-5 family). Warn once if a caller
    # passes it for another provider; don't raise — keeps the API uniform.
    if reasoning_effort is not None and api_provider != "openai":
        if api_provider not in _REASONING_EFFORT_WARNED:
            _REASONING_EFFORT_WARNED.add(api_provider)
            warnings.warn(
                f"reasoning_effort is OpenAI-specific; ignored for {api_provider}.",
                stacklevel=2,
            )

    if api_provider == "gemini":
        # google.genai (Jan 2026) does not retry 503/429 internally — wrap manually.
        # The openai and anthropic SDKs already retry these on their own.
        import time
        import os
        from google import genai
        gcp_project = os.environ.get("GCP_PROJECT_ID")
        gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        if gcp_project and "3.5" not in model:
            client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        else:
            client = genai.Client(api_key=api_key)
        model = model or "gemini-3.5-flash"

        def _gemini_config(use_logprobs):
            cfg_kwargs = {}
            if use_logprobs:
                cfg_kwargs.update(response_logprobs=True, logprobs=5)
            if temperature is not None:
                cfg_kwargs["temperature"] = temperature
            if seed is not None:
                cfg_kwargs["seed"] = seed
            if not cfg_kwargs:
                return None
            try:
                from google.genai import types
                return types.GenerateContentConfig(**cfg_kwargs)
            except Exception as e:
                if "gemini" not in _LOGPROBS_WARNED:
                    _LOGPROBS_WARNED.add("gemini")
                    warnings.warn(f"Could not build Gemini config: {e}", stacklevel=2)
                return None

        config = _gemini_config(with_logprobs)
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                    config=config,
                )
                text = response.text
                if return_usage:
                    in_t, out_t = _extract_usage("gemini", response)
                    meta = {"input_tokens": in_t, "output_tokens": out_t}
                    if with_logprobs:
                        meta["prob_fully_paid"] = _extract_prediction_prob(
                            "gemini", text, _extract_token_logprobs("gemini", response)
                        )
                    return text, meta
                return text
            except Exception as e:
                err_str = str(e).lower()
                if with_logprobs and "logprobs" in err_str and ("not enabled" in err_str or "invalid" in err_str):
                    if "gemini" not in _LOGPROBS_WARNED:
                        _LOGPROBS_WARNED.add("gemini")
                        warnings.warn(
                            f"Gemini model {model} does not support logprobs. "
                            "Falling back to calling without logprobs.",
                            stacklevel=2,
                        )
                    with_logprobs = False
                    config = _gemini_config(False)
                    # Retry immediately without waiting
                    try:
                        response = client.models.generate_content(
                            model=model,
                            contents=f"{system_prompt}\n\n{user_prompt}",
                            config=config,
                        )
                        text = response.text
                        if return_usage:
                            in_t, out_t = _extract_usage("gemini", response)
                            meta = {"input_tokens": in_t, "output_tokens": out_t, "prob_fully_paid": None}
                            return text, meta
                        return text
                    except Exception as fallback_err:
                        e = fallback_err
                
                if "503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e):
                    wait = 2 ** attempt * 5
                    print(f"  Retry {attempt+1}/5 in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Gemini API failed after 5 retries")

    elif api_provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        model = model or "claude-sonnet-4-6"
        anth_kwargs = {}
        if temperature is not None:
            anth_kwargs["temperature"] = temperature  # Anthropic has no seed param
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **anth_kwargs,
        )
        text = response.content[0].text
        if return_usage:
            in_t, out_t = _extract_usage("anthropic", response)
            meta = {"input_tokens": in_t, "output_tokens": out_t}
            if with_logprobs:
                # Anthropic API does not expose logprobs.
                if "anthropic" not in _LOGPROBS_WARNED:
                    _LOGPROBS_WARNED.add("anthropic")
                    warnings.warn(
                        "Anthropic API does not expose token logprobs; "
                        "prob_fully_paid will be None for this provider.",
                        stacklevel=2,
                    )
                meta["prob_fully_paid"] = None
            return text, meta
        return text

    elif api_provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        model = model or "gpt-4o"
        kwargs = {}
        if with_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        # temperature/seed: reasoning models (gpt-5 family) reject temperature, so
        # only pass when a caller explicitly sets it. seed is best-effort determinism.
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens or 2048,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
        except Exception as e:
            err_str = str(e).lower()
            if with_logprobs and "logprobs" in err_str:
                if "openai" not in _LOGPROBS_WARNED:
                    _LOGPROBS_WARNED.add("openai")
                    warnings.warn(
                        f"OpenAI model {model} does not support logprobs under current settings. "
                        "Falling back to calling without logprobs.",
                        stacklevel=2,
                    )
                with_logprobs = False
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                response = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=max_tokens or 2048,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )
            else:
                raise
        text = response.choices[0].message.content
        if return_usage:
            in_t, out_t = _extract_usage("openai", response)
            meta = {"input_tokens": in_t, "output_tokens": out_t}
            if with_logprobs:
                meta["prob_fully_paid"] = _extract_prediction_prob(
                    "openai", text, _extract_token_logprobs("openai", response)
                )
            return text, meta
        return text

    elif api_provider == "nvidia":
        import time
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        model = model or "meta/llama-3.3-70b-instruct"
        kwargs = {}
        if with_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens or 2048,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )
                text = response.choices[0].message.content
                if return_usage:
                    in_t, out_t = _extract_usage("nvidia", response)
                    meta = {"input_tokens": in_t, "output_tokens": out_t}
                    if with_logprobs:
                        meta["prob_fully_paid"] = _extract_prediction_prob(
                            "openai", text, _extract_token_logprobs("openai", response)
                        )
                    return text, meta
                return text
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = "429" in str(e) or "too many requests" in err or "rate" in err
                is_connection = "connection" in err or "timeout" in err or "503" in str(e) or "502" in str(e) or "504" in str(e)
                if is_rate_limit:
                    wait = 2 ** attempt * 10  # 10s, 20s, 40s …
                    print(f"  [nvidia] Rate limited — retry {attempt+1}/8 in {wait}s...")
                    time.sleep(wait)
                elif is_connection:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s …
                    print(f"  [nvidia] Connection error — retry {attempt+1}/8 in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("NVIDIA NIM API failed after 8 retries")

    elif api_provider == "groq":
        # Groq is OpenAI-compatible (https://api.groq.com/openai/v1). Same retry
        # shape as nvidia. Logprobs are not reliably exposed by Groq for the
        # Llama models, so with_logprobs is accepted but prob_fully_paid will be
        # None unless the provider returns usable token data.
        import time
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = model or "llama-3.3-70b-versatile"
        kwargs = {}
        if with_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens or 2048,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )
                text = response.choices[0].message.content
                if return_usage:
                    in_t, out_t = _extract_usage("groq", response)
                    meta = {"input_tokens": in_t, "output_tokens": out_t}
                    if with_logprobs:
                        meta["prob_fully_paid"] = _extract_prediction_prob(
                            "openai", text, _extract_token_logprobs("openai", response)
                        )
                    return text, meta
                return text
            except Exception as e:
                estr = str(e)
                err = estr.lower()
                is_rate_limit = "429" in estr or "too many requests" in err or "rate" in err
                is_connection = "connection" in err or "timeout" in err or "503" in estr or "502" in estr
                if is_rate_limit:
                    # Surface Groq's exact reason + retry-after so we can tell which
                    # limit hit: per-minute (RPM/TPM, resets in seconds) vs per-day
                    # (RPD/TPD, resets at midnight UTC). Groq names it in the body.
                    retry_after = None
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        try:
                            ra = resp.headers.get("retry-after")
                            retry_after = float(ra) if ra is not None else None
                        except Exception:
                            retry_after = None
                    reason = ""
                    body = getattr(e, "body", None)
                    if isinstance(body, dict):
                        reason = str(body.get("error", {}).get("message", ""))[:300]
                    if not reason:
                        reason = estr[:300]
                    # A long retry-after means a daily cap — won't clear by waiting,
                    # so fail fast with a clear message instead of sleeping for ~1h.
                    if retry_after is not None and retry_after > 120:
                        raise RuntimeError(
                            f"[groq] day-scale rate limit hit (retry-after={retry_after:.0f}s, "
                            f"~={retry_after/60:.0f} min). This is a per-DAY quota (RPD/TPD); "
                            f"it resets at midnight UTC. Reason: {reason}"
                        )
                    wait = retry_after if retry_after is not None else min(2 ** attempt * 10, 60)
                    print(f"  [groq] 429 [{reason}] — retry {attempt+1}/8 in {wait:.0f}s")
                    time.sleep(wait)
                elif is_connection:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s …
                    print(f"  [groq] Connection error — retry {attempt+1}/8 in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq API failed after 8 retries")

    else:
        raise ValueError(f"Unknown api_provider: {api_provider}")


def parse_llm_response(text):
    """
    Parse the LLM JSON response.
    Returns dict with 'prediction' (int) and 'reasoning' (str).
    """
    text = text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        result = json.loads(text)
        result['prediction'] = int(result['prediction'])
        return result
    except (json.JSONDecodeError, KeyError, ValueError):
        return {'prediction': None, 'reasoning': f'PARSE_ERROR: {text[:200]}'}


def format_loans_batch(df, user_prompt_fn=None, include_desc=False):
    """
    Format a DataFrame of loans as a compact numbered table for batch prediction.
    TOON-inspired: field headers once, then pipe-separated rows — ~40% fewer tokens
    than repeating field names per loan.

    When user_prompt_fn is provided (e.g. top_features_only, chain_of_thought),
    each loan section uses the custom prompt but is wrapped with an index marker
    so the model knows which index to return.
    """
    n = len(df)
    lines = []

    if user_prompt_fn is not None:
        lines.append(f"Predict outcomes for the {n} loans below.")
        lines.append("Return a JSON array: "
                     '[{"i": 0, "prediction": 1, "reasoning": "..."}, ...]')
        lines.append("")
        for i, (_, row) in enumerate(df.iterrows()):
            lines.append(f"--- LOAN #{i} ---")
            lines.append(user_prompt_fn(row))
    else:
        features = [c for c in LLM_FEATURES if c in df.columns]
        lines.append(f"Predict outcomes for the {n} loans below.")
        lines.append("Return a JSON array: "
                     '[{"i": 0, "prediction": 1, "reasoning": "..."}, ...]')
        lines.append("")
        lines.append("Fields: " + " | ".join(features))
        for i, (_, row) in enumerate(df.iterrows()):
            vals = []
            for f in features:
                v = row.get(f, "N/A")
                vals.append("N/A" if pd.isna(v) else str(v))
            lines.append(f"#{i}: " + " | ".join(vals))

    return "\n".join(lines)


def _batch_system_prompt(system_prompt, n):
    """
    Rewrite the single-prediction JSON instruction in system_prompt for batch use.
    Replaces the per-loan output spec with an array output spec.
    """
    # Remove the existing single-prediction JSON block
    import re as _re
    cleaned = _re.sub(
        r"Respond ONLY with valid JSON.*?reasoning.*?\}.*?$",
        "",
        system_prompt,
        flags=_re.DOTALL | _re.IGNORECASE,
    ).rstrip()
    batch_instruction = (
        f"\n\nYou will receive {n} loans. "
        "Respond ONLY with a JSON array — one object per loan:\n"
        '[{"i": 0, "prediction": 1, "reasoning": "brief"}, '
        '{"i": 1, "prediction": 0, "reasoning": "brief"}, ...]\n'
        "prediction: 1 = Fully Paid, 0 = Charged Off. "
        "reasoning: 1-2 sentences. Include ALL loans — do not skip any index."
    )
    return cleaned + batch_instruction


def parse_batch_llm_response(text, n):
    """
    Parse a batch response containing a JSON array of {i, prediction, reasoning}.
    Returns (predictions, reasonings) — lists of length n, None/MISSING for gaps.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    predictions = [None] * n
    reasonings  = ["MISSING"] * n

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("predictions", "results", "loans", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        for item in data:
            idx = item.get("i", item.get("index", item.get("loan_index")))
            if idx is None:
                continue
            idx = int(idx)
            if 0 <= idx < n:
                raw_pred = item.get("prediction")
                predictions[idx] = int(raw_pred) if raw_pred is not None else None
                reasonings[idx]  = str(item.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    return predictions, reasonings


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true, y_pred, label="Model", probabilities=None):
    """Print classification metrics for a set of predictions.

    probabilities: optional list/array of P(class=1=Fully Paid) per row, same
    length as y_pred. When supplied, AUC is reported (computed only over rows
    where both prediction and probability are non-None)."""
    valid = [i for i in range(len(y_pred)) if y_pred[i] is not None]
    if len(valid) < len(y_pred):
        print(f"Warning: {len(y_pred) - len(valid)} unparseable predictions excluded")

    print(f"\n{'=' * 50}")
    print(f"{label} Results ({len(valid)} samples)")
    print(f"{'=' * 50}")

    # All calls failed (e.g. persistent 504) — return a null metrics dict rather
    # than crashing the whole experiment loop.
    if len(valid) == 0:
        print("No valid predictions — all API calls failed.")
        return {
            'accuracy': None, 'auc': None,
            'precision_charged_off': None, 'recall_charged_off': None,
            'f1_charged_off': None, 'n_valid': 0,
        }

    y_true_v = np.array(y_true)[valid].astype(int)
    y_pred_v = np.array(y_pred)[valid].astype(int)
    print(f"Accuracy: {accuracy_score(y_true_v, y_pred_v) * 100:.1f}%")

    auc = None
    if probabilities is not None:
        prob_valid = [(i, probabilities[i]) for i in valid if probabilities[i] is not None]
        if len(prob_valid) >= 2 and len(set(int(y_true[i]) for i, _ in prob_valid)) == 2:
            idxs = [i for i, _ in prob_valid]
            scores = np.array([p for _, p in prob_valid])
            auc = roc_auc_score(np.array(y_true)[idxs], scores)
            print(f"AUC:      {auc:.3f}  (over {len(prob_valid)} rows with logprobs)")
        else:
            print(f"AUC:      n/a  (only {len(prob_valid)} rows had usable probabilities)")

    print(f"\nClassification Report:")
    print(classification_report(y_true_v, y_pred_v,
                                target_names=['Charged Off', 'Fully Paid']))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true_v, y_pred_v))

    return {
        'accuracy': accuracy_score(y_true_v, y_pred_v),
        'precision_charged_off': precision_score(y_true_v, y_pred_v, pos_label=0),
        'recall_charged_off': recall_score(y_true_v, y_pred_v, pos_label=0),
        'f1_charged_off': f1_score(y_true_v, y_pred_v, pos_label=0),
        'auc': auc,
        'n_valid': len(valid),
    }


def compare_results(y_true, llm_preds, ml_preds, llm_reasonings=None,
                    input_tokens=None, output_tokens=None, cost_usd=None):
    """
    Build a comparison DataFrame of LLM vs XGBoost predictions.

    Pass `input_tokens`/`output_tokens`/`cost_usd` (per-loan lists, e.g. from a
    run_llm_experiment result) to embed cost data directly in the predictions
    file. This makes cost derivable from predictions alone, so the predictions
    file and the per-call cost log (llm_calls.csv) cannot silently drift.
    """
    df = pd.DataFrame({
        'actual': y_true,
        'llm_pred': llm_preds,
        'xgb_pred': ml_preds,
        'llm_correct': [1 if p == a else 0 for p, a in zip(llm_preds, y_true)],
        'xgb_correct': [1 if p == a else 0 for p, a in zip(ml_preds, y_true)],
    })
    if llm_reasonings:
        df['llm_reasoning'] = llm_reasonings
    if input_tokens is not None:
        df['input_tokens'] = input_tokens
    if output_tokens is not None:
        df['output_tokens'] = output_tokens
    if cost_usd is not None:
        df['cost_usd'] = cost_usd
    return df


# ── Per-call LLM log (tokens, cost, prediction probability) ─────────────────

_LLM_CALLS_LOCK = threading.Lock()
LLM_CALLS_PATH = Path(RESULTS_DIR) / "llm_calls.csv"

# Frozen 14-column schema — pandas pivots in the analysis notebooks depend on it.
# Do NOT add or remove columns here (per CLAUDE.md).
LLM_CALLS_COLUMNS = [
    "timestamp", "notebook_id", "label", "desc_tag", "provider", "model",
    "row_index", "input_tokens", "output_tokens", "input_price_per_1k_usd",
    "output_price_per_1k_usd", "cost_usd", "prob_fully_paid", "reasoning_effort",
]
# (notebook_id, label, desc_tag) uniquely identifies one run_llm_experiment call:
# every notebook uses a distinct label per logical run (e.g. "GPT-5.4 Run 1
# (no_desc)", "... | run1", "reasoning=high"), so replacing rows on this key on
# re-run can never clobber a sibling run.
_LLM_CALLS_KEY = ["notebook_id", "label", "desc_tag"]


def _atomic_write_csv(df, path):
    """Write `df` to `path` via a temp file in the same directory + atomic
    os.replace, so an interrupted write can never truncate or corrupt the
    existing file (it is replaced in one atomic step or not at all)."""
    import os
    import tempfile
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _append_llm_calls(rows, replace=True):
    """
    Persist per-call rows to data/results/llm/llm_calls.csv. One row per
    successful API call: tokens, cost, and (when logprobs are available)
    P(prediction=1).

    Idempotent and crash-safe (fixes the silent cost/metrics drift):
    - replace=True (default): before writing, any existing rows whose
      (notebook_id, label, desc_tag) match the incoming rows are dropped, so
      re-running an experiment REPLACES its own rows instead of appending
      duplicates (which would double-count cost). Distinct logical runs use
      distinct labels, so a sibling run is never affected.
    - the write goes through a temp file + atomic rename, so a crash mid-write
      cannot corrupt the shared, git-tracked log.

    Thread-safe. Only invoked on successful completion; interrupted runs drop
    their buffer and never reach this.
    """
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    with _LLM_CALLS_LOCK:
        LLM_CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LLM_CALLS_PATH.exists():
            try:
                existing = pd.read_csv(LLM_CALLS_PATH)
            except Exception:
                existing = pd.DataFrame(columns=LLM_CALLS_COLUMNS)
            if replace and len(existing) and set(_LLM_CALLS_KEY).issubset(existing.columns):
                ex_keys = existing[_LLM_CALLS_KEY].apply(tuple, axis=1)
                new_keys = set(new_df[_LLM_CALLS_KEY].apply(tuple, axis=1))
                existing = existing[~ex_keys.isin(new_keys)]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        _atomic_write_csv(combined, LLM_CALLS_PATH)


def dedup_llm_calls(keep="last"):
    """One-shot maintenance: collapse duplicate per-call rows in llm_calls.csv
    on (notebook_id, label, desc_tag, row_index), keeping the most recent by
    timestamp. Legitimate multi-run experiments use distinct labels, so they're
    left intact — this only removes accidental double-logging from re-runs that
    predate the idempotent writer. Returns (rows_before, rows_after)."""
    with _LLM_CALLS_LOCK:
        if not LLM_CALLS_PATH.exists():
            return (0, 0)
        df = pd.read_csv(LLM_CALLS_PATH)
        before = len(df)
        subset = [c for c in ["notebook_id", "label", "desc_tag", "row_index"]
                  if c in df.columns]
        df = (df.sort_values("timestamp")
                .drop_duplicates(subset=subset, keep=keep)
                .reset_index(drop=True))
        _atomic_write_csv(df, LLM_CALLS_PATH)
        return (before, len(df))

def _get_notebook_name():
    """
    Dynamically retrieve the current notebook filename across multiple environments:
    1. VS Code Jupyter Extension (`__vsc_ipynb_file__` or session metadata in shell namespace)
    2. Jupyter Web / Browser (`JPY_SESSION_NAME` environment variable)
    3. Jupyter/nbconvert execution parent command line parsing (via JPY_PARENT_PID)
    Falls back to the current directory name if not running within a notebook context.
    """
    import os
    import re
    # 1. Inspect IPython namespace
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            # Check for VS Code ipynb file reference
            vsc_file = ip.user_ns.get("__vsc_ipynb_file__")
            if vsc_file:
                return os.path.basename(vsc_file)
            
            # Check for standard ipykernel session reference
            session = ip.user_ns.get("__session__")
            if session and isinstance(session, str) and session.endswith(".ipynb"):
                return os.path.basename(session)
    except Exception:
        pass

    # 2. Check Jupyter session env vars
    jpy_session = os.environ.get("JPY_SESSION_NAME")
    if jpy_session:
        return os.path.basename(jpy_session)

    # 3. Check JPY_PARENT_PID for nbconvert/jupyter runner process command line
    ppid = os.environ.get("JPY_PARENT_PID")
    if ppid:
        try:
            import subprocess
            import shlex
            res = subprocess.run(["ps", "-p", str(ppid), "-o", "command="], capture_output=True, text=True)
            parent_cmd = res.stdout.strip()
            if parent_cmd:
                try:
                    tokens = shlex.split(parent_cmd)
                except Exception:
                    tokens = parent_cmd.split()
                for token in tokens:
                    token = token.strip("'\"")
                    if ".ipynb" in token:
                        base = token.split("?")[0]
                        if base.endswith(".ipynb"):
                            return os.path.basename(base)
                        # Extract the notebook file if it's embedded in an arg
                        match = re.search(r'([^/\\]+\.ipynb)', token)
                        if match:
                            return match.group(1)
        except Exception:
            pass

    # Fallback to the current directory name
    return os.path.basename(os.getcwd())


# ── Experiment runner ────────────────────────────────────────────────────────

def run_llm_experiment(llm_sample, api_provider, model_name, include_desc=False,
                       api_key=None, label=None, with_logprobs=True,
                       system_prompt=None, user_prompt_fn=None,
                       batch_size=None, reasoning_effort=None,
                       notebook_id=None, temperature=None, seed=None,
                       strict=False, max_fail_frac=0.5,
                       api_keys=None, max_workers=None, use_cache=False):
    """
    Run the full LLM prediction loop on a sample.

    Args:
        llm_sample: DataFrame with loan features
        api_provider: "gemini", "anthropic", or "openai"
        model_name: model identifier string
        include_desc: whether to include borrower description
        api_key: optional API key override
        label: display label (defaults to model_name)
        with_logprobs: request token logprobs and compute P(prediction=1) per
                       call so AUC is reportable. Default True. Anthropic
                       silently skips (probabilities will be None).
                       Ignored in batch mode (logprobs are per-call, not per-loan).
        system_prompt: optional system prompt string override. If None, uses
                       build_system_prompt().
        user_prompt_fn: optional callable(row) -> str override for building
                        the user prompt. If None, uses build_user_prompt(row,
                        include_desc=include_desc).
        batch_size: if 0, send all loans in a single API call (TOON-style compact
                    table). If N > 0, send chunks of N loans per call. If None,
                    use the original one-call-per-loan loop.
        notebook_id: explicit value for the cost log's notebook_id column. If
                     None, falls back to _get_notebook_name() (which can mis-resolve
                     outside Jupyter); pass it explicitly for deterministic tagging.
        temperature/seed: determinism controls forwarded to call_llm. Default
                          None = provider default. (Reasoning models ignore
                          temperature; use reasoning_effort for those.)
        strict: if True, raise RuntimeError when the fraction of failed/unparsed
                predictions exceeds max_fail_frac — AFTER logging what succeeded —
                so a half-broken run can't silently masquerade as complete.
        max_fail_frac: failure-fraction threshold for `strict` (default 0.5).
        api_keys: optional list of API keys to rotate across when running the
                  per-loan loop in parallel (round-robin by loan index). More
                  keys = more aggregate throughput under per-key rate limits.
                  Ignored in batch mode. Falls back to [api_key] if omitted.
        max_workers: thread-pool size for the per-loan loop. None/1 = the
                  original sequential loop (default, fully backward compatible);
                  >1 fans calls out concurrently (results stay in loan order;
                  cost logging is index-keyed and lock-guarded). For 1000-loan
                  runs, len(api_keys) * ~4 is a good starting point.
        use_cache: if True and notebook_id is provided, check the JSON cache in
                   data/results/llm/cache/ for an identical prior run, and instantly
                   return it to bypass all API calls. A successful new run will
                   automatically save its result to this cache.

    Returns:
        dict with 'predictions', 'probabilities', 'reasonings', 'raw_responses',
        'metrics', 'label', 'desc_tag', plus per-loan 'input_tokens',
        'output_tokens', 'cost_usd' (lists; None where a call failed or in batch
        mode), and 'n_failed' / 'failed_indices'.
    """
    import time as _time
    import threading
    _log_lock = threading.Lock()

    label = label or model_name
    desc_tag = "with_desc" if include_desc else "no_desc"
    tag = f"[{label} | {desc_tag}]"
    nb_id = notebook_id or _get_notebook_name()

    if use_cache and nb_id:
        c_path = _get_cache_path(nb_id, label, desc_tag)
        if os.path.exists(c_path):
            print(f"{tag} Loading cached results from {os.path.basename(c_path)}...")
            with open(c_path, 'r') as f:
                import json
                return json.load(f)

    # Resolve pricing once, fail-fast on unknown model before burning API calls.
    in_price_per_1k, out_price_per_1k = get_price(model_name)

    system_prompt = system_prompt or build_system_prompt()
    y_true = llm_sample['loan_status'].values
    n = len(llm_sample)

    call_rows = []
    # Per-loan cost arrays, returned so the predictions file can embed cost and
    # never drift from llm_calls.csv. None where a call failed / in batch gaps.
    tok_in_arr = [None] * n
    tok_out_arr = [None] * n
    cost_arr = [None] * n
    failed_indices = []

    def _record(row_index, meta):
        in_t = meta.get("input_tokens", 0)
        out_t = meta.get("output_tokens", 0)
        cost_usd = (in_t / 1000.0) * in_price_per_1k + (out_t / 1000.0) * out_price_per_1k
        with _log_lock:
            call_rows.append({
                "timestamp":             datetime.now(timezone.utc).isoformat(),
                "notebook_id":           nb_id,
                "label":                 label,
                "desc_tag":              desc_tag,
                "provider":              api_provider,
                "model":                 model_name,
                "row_index":             row_index,
                "input_tokens":          in_t,
                "output_tokens":         out_t,
                "input_price_per_1k_usd":  in_price_per_1k,
                "output_price_per_1k_usd": out_price_per_1k,
                "cost_usd":              cost_usd,
                "prob_fully_paid":       meta.get("prob_fully_paid"),
                "reasoning_effort":      reasoning_effort,
            })
        if 0 <= row_index < n:
            tok_in_arr[row_index] = in_t
            tok_out_arr[row_index] = out_t
            cost_arr[row_index] = cost_usd

    def _build_user_prompt(row):
        if user_prompt_fn is not None:
            return user_prompt_fn(row)
        return build_user_prompt(row, include_desc=include_desc)

    def _do_call(prompt, key=None):
        return call_llm(
            system_prompt, prompt,
            api_provider=api_provider, model=model_name, api_key=(key or api_key),
            return_usage=True, with_logprobs=with_logprobs,
            reasoning_effort=reasoning_effort, temperature=temperature, seed=seed,
        )

    # ── Batch mode ────────────────────────────────────────────────────────────
    if batch_size is not None:
        n = len(llm_sample)
        chunk = n if batch_size == 0 else batch_size
        chunks = [llm_sample.iloc[i:i+chunk] for i in range(0, n, chunk)]
        n_calls = len(chunks)

        predictions   = [None] * n
        probabilities = [None] * n
        reasonings    = ["MISSING"] * n
        raw_responses = []

        start = _time.time()
        for call_i, sub_df in enumerate(chunks):
            offset = call_i * chunk

            batch_sys = _batch_system_prompt(system_prompt, len(sub_df))
            batch_usr = format_loans_batch(sub_df, user_prompt_fn=user_prompt_fn,
                                           include_desc=include_desc)
            print(f"{tag} Batch call {call_i+1}/{n_calls} "
                  f"(loans {offset}–{offset+len(sub_df)-1}) ...")
            try:
                raw, meta = call_llm(
                    batch_sys, batch_usr,
                    api_provider=api_provider, model=model_name, api_key=api_key,
                    return_usage=True, with_logprobs=False,
                    max_tokens=len(sub_df) * 120,  # ~120 tokens per prediction
                    temperature=temperature, seed=seed,
                )
                raw_responses.append(raw)
                _record(offset, meta)

                preds, reasns = parse_batch_llm_response(raw, len(sub_df))
                for j, (pred, rsn) in enumerate(zip(preds, reasns)):
                    predictions[offset + j]   = pred
                    reasonings[offset + j]    = rsn
                    probabilities[offset + j] = None

                n_ok = sum(1 for p in preds if p is not None)
                print(f"{tag}  -> {n_ok}/{len(sub_df)} parsed OK "
                      f"({_time.time()-start:.0f}s elapsed)")
            except Exception as e:
                print(f"{tag} Batch call {call_i+1} FAILED: {e}")
                raw_responses.append(None)

        n_errors = sum(1 for p in predictions if p is None)
        total_time = _time.time() - start
        print(f"{tag} COMPLETE — {n_calls} batch calls, "
              f"{n - n_errors}/{n} predictions parsed, {total_time:.0f}s total")

    # ── Per-loan loop (sequential, or parallel across keys) ───────────────────
    else:
        keys = [k for k in (api_keys or [api_key]) if k] or [None]
        workers = max_workers or (len(keys) if len(keys) > 1 else 1)

        # Index-keyed result arrays so parallel completions stay in loan order.
        predictions   = [None] * n
        probabilities = [None] * n
        reasonings    = ["MISSING"] * n
        raw_responses = [None] * n
        rows = [row for _, row in llm_sample.iterrows()]

        def _process_one(i, row, key):
            # Resilient: a transient failure on one loan (after call_llm's own
            # retries) records a gap and continues, rather than discarding the
            # whole run's buffer. Failures are counted and surfaced loudly below.
            try:
                raw, meta = _do_call(_build_user_prompt(row), key)
                parsed = parse_llm_response(raw)
                _record(i, meta)
                raw_responses[i]  = raw
                predictions[i]    = parsed['prediction']
                probabilities[i]  = meta.get('prob_fully_paid')
                reasonings[i]     = parsed['reasoning']
            except Exception as e:
                print(f"{tag}   loan {i} FAILED after retries: {e}")
                reasonings[i] = f"CALL_ERROR: {e}"

        # Fail-fast: run loan 0 first so an auth/route error aborts before we
        # fan out n calls (and before any worker pool spins up).
        try:
            _process_one(0, rows[0], keys[0])
            if predictions[0] is None and str(reasonings[0]).startswith("CALL_ERROR"):
                raise RuntimeError(reasonings[0])
            print(f"{tag} First call OK (pred={predictions[0]}, prob={probabilities[0]}). "
                  f"Running {n} loans on {len(keys)} key(s) × {workers} worker(s)...")
        except Exception as e:
            print(f"{tag} FAILED on first call: {e}")
            raise RuntimeError(f"{tag} cannot reach API: {e}")

        start = _time.time()
        if workers <= 1:
            for i in range(1, n):
                _process_one(i, rows[i], keys[0])
                if (i + 1) % 10 == 0:
                    elapsed = _time.time() - start
                    rate = (i + 1) / max(elapsed, 1e-9)
                    eta = (n - i - 1) / max(rate, 1e-9)
                    print(f"{tag} {i+1}/{n} done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_process_one, i, rows[i], keys[i % len(keys)]): i
                        for i in range(1, n)}
                done = 0
                for fut in as_completed(futs):
                    fut.result()
                    done += 1
                    if done % 25 == 0 or done == len(futs):
                        elapsed = _time.time() - start
                        rate = done / max(elapsed, 1e-9)
                        eta = (len(futs) - done) / max(rate, 1e-9)
                        print(f"{tag} {done+1}/{n} done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        n_errors = sum(1 for p in predictions if p is None)
        n_probs = sum(1 for p in probabilities if p is not None)
        total_time = _time.time() - start
        print(f"{tag} COMPLETE — {len(predictions)} predictions, {n_errors} parse errors, "
              f"{n_probs} with logprobs, {total_time:.0f}s total")

    try:
        _append_llm_calls(call_rows)
    except Exception as e:
        print(f"Warning: could not write call log: {e}")

    metrics = evaluate_predictions(
        y_true, predictions, label=f"{label} ({desc_tag})",
        probabilities=probabilities,
    )

    # Uniform failure accounting across per-loan and batch modes.
    failed_indices = [i for i, p in enumerate(predictions) if p is None]
    n_failed = len(failed_indices)
    if n_failed:
        print(f"{tag} ⚠️  {n_failed}/{n} predictions failed or did not parse "
              f"(indices: {failed_indices[:20]}{'…' if n_failed > 20 else ''})")

    # Loud guard: don't let a half-broken run masquerade as a complete result.
    # The cost rows are still written above (so nothing is wasted), then we raise.
    if strict and n > 0 and (n_failed / n) > max_fail_frac:
        raise RuntimeError(
            f"{tag} {n_failed}/{n} predictions failed (> {max_fail_frac:.0%} "
            f"threshold). Cost rows were logged; aborting before this partial "
            f"result is saved. Set strict=False to keep partial results."
        )

    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'reasonings': reasonings,
        'raw_responses': raw_responses,
        'metrics': metrics,
        'label': label,
        'desc_tag': desc_tag,
        'input_tokens': tok_in_arr,
        'output_tokens': tok_out_arr,
        'cost_usd': cost_arr,
        'n_failed': n_failed,
        'failed_indices': failed_indices,
    }
