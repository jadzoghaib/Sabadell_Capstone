"""
Shared utilities for LLM evaluation notebooks.

Handles: data loading, ML re-encoding/prediction, prompt building,
LLM API calls, evaluation metrics, and full experiment loops.
"""

import json
import math
import re
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)

from llm_pricing import get_price

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
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose',
    'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'initial_list_status', 'application_type',
    'mort_acc', 'pub_rec_bankruptcies',
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
    'pub_rec': 'Has derogatory public records (0/1)',
    'revol_bal': 'Revolving balance ($)',
    'revol_util': 'Revolving utilization rate (%)',
    'total_acc': 'Total number of credit accounts',
    'initial_list_status': 'Initial listing status (w=whole, f=fractional)',
    'application_type': 'Individual or joint application',
    'mort_acc': 'Has mortgage accounts (0/1)',
    'pub_rec_bankruptcies': 'Has public record bankruptcies (0/1)',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_llm_sample():
    """Load the 100-row human-readable LLM evaluation sample."""
    df = pd.read_csv(f"{DATA_DIR}/02_llm_sample.csv")
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
    `loan_status` mapped to 1/0. Cached for the process — both `sample_new_batch`
    and `build_few_shot_examples` consume this and the raw .csv.gz is ~1GB."""
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


def sample_new_batch(n=100, random_state=99):
    """Sample a new batch of n loans from the test set (with descriptions),
    excluding rows already in `02_llm_sample.csv`."""
    _, test_data = _train_test_split_canonical(_load_lendingclub_filtered())

    test_with_desc = test_data[test_data['desc'].notna() & (test_data['desc'].str.strip() != '')]

    original = load_llm_sample()
    orig_keys = set(zip(original['loan_amnt'], original['int_rate'], original['annual_inc']))
    mask = ~test_with_desc.apply(
        lambda r: (r['loan_amnt'], r['int_rate'], r['annual_inc']) in orig_keys, axis=1
    )
    available = test_with_desc[mask]

    return available.sample(n=n, random_state=random_state).reset_index(drop=True)


# ── XGBoost prediction on LLM sample rows ───────────────────────────────────

def run_ml_on_sample(llm_sample):
    """
    Re-encode and scale the LLM sample, then run the XGBoost model.
    Returns (xgb_probs, xgb_preds) as arrays.
    """
    import joblib

    scaler = joblib.load(f"{DATA_DIR}/02_scaler.joblib")
    feature_cols = joblib.load(f"{DATA_DIR}/02_feature_columns.joblib")
    model = joblib.load(f"{MODEL_DIR}/xgb_model.joblib")
    thresholds = joblib.load(f"{MODEL_DIR}/thresholds.joblib")
    threshold = thresholds['xgb']

    df = llm_sample.copy()

    # Apply same preprocessing as 02_Preprocessing
    term_values = {' 36 months': 36, ' 60 months': 60}
    if df['term'].dtype == object:
        df['term'] = df.term.map(term_values)

    df.drop(columns=['grade', 'zip_code', 'addr_state', 'issue_d', 'desc'],
            errors='ignore', inplace=True)
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line']).dt.year

    dummies = ['sub_grade', 'verification_status', 'purpose',
               'initial_list_status', 'application_type', 'home_ownership']
    df = pd.get_dummies(df, columns=dummies, drop_first=True)

    X = df.drop(columns=['loan_status'], errors='ignore')
    X = X.reindex(columns=feature_cols, fill_value=0)

    X_scaled = scaler.transform(X).astype(np.float32)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    return probs, preds


# ── Prompt building ───────────────────────────────────────────────────────────

def format_loan_features(row, include_desc=False):
    """Format a single loan's features as a readable string for the LLM."""
    lines = []
    for feat in LLM_FEATURES:
        if feat in row and pd.notna(row[feat]):
            label = FEATURE_DESCRIPTIONS.get(feat, feat)
            lines.append(f"- {label}: {row[feat]}")

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
    """Load .env into os.environ. `override=False` matches setdefault semantics:
    an exported shell var beats a stale .env entry."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)


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
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_var = key_map.get(api_provider)
    return os.environ.get(env_var) if env_var else None


_USAGE_ATTRS = {
    "gemini":    ("usage_metadata", "prompt_token_count", "candidates_token_count"),
    "openai":    ("usage",          "prompt_tokens",      "completion_tokens"),
    "anthropic": ("usage",          "input_tokens",       "output_tokens"),
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
_PREDICTION_RE = re.compile(r'"prediction"\s*:\s*([01])')


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
             api_key=None, return_usage=False, with_logprobs=False):
    """
    Call the LLM API. Supports gemini, anthropic, and openai providers.

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

    if api_provider == "gemini":
        # google.genai (Jan 2026) does not retry 503/429 internally — wrap manually.
        # The openai and anthropic SDKs already retry these on their own.
        import time
        from google import genai
        client = genai.Client(api_key=api_key)
        model = model or "gemini-2.5-flash"
        config = None
        if with_logprobs:
            try:
                from google.genai import types
                config = types.GenerateContentConfig(response_logprobs=True, logprobs=5)
            except Exception as e:
                if "gemini" not in _LOGPROBS_WARNED:
                    _LOGPROBS_WARNED.add("gemini")
                    warnings.warn(f"Could not build Gemini logprobs config: {e}", stacklevel=2)
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
        model = model or "claude-sonnet-4-20250514"
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
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
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true, y_pred, label="Model", probabilities=None):
    """Print classification metrics for a set of predictions.

    probabilities: optional list/array of P(class=1=Fully Paid) per row, same
    length as y_pred. When supplied, AUC is reported (computed only over rows
    where both prediction and probability are non-None)."""
    valid = [i for i in range(len(y_pred)) if y_pred[i] is not None]
    if len(valid) < len(y_pred):
        print(f"Warning: {len(y_pred) - len(valid)} unparseable predictions excluded")

    y_true_v = np.array(y_true)[valid]
    y_pred_v = np.array(y_pred)[valid]

    print(f"\n{'=' * 50}")
    print(f"{label} Results ({len(valid)} samples)")
    print(f"{'=' * 50}")
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


def compare_results(y_true, llm_preds, ml_preds, llm_reasonings=None):
    """
    Build a comparison DataFrame of LLM vs XGBoost predictions.
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
    return df


# ── Per-call LLM log (tokens, cost, prediction probability) ─────────────────

_LLM_CALLS_LOCK = threading.Lock()
LLM_CALLS_PATH = Path(RESULTS_DIR) / "llm_calls.csv"


def _append_llm_calls(rows):
    """
    Append per-call rows to data/results/llm/llm_calls.csv. One row per
    successful API call: tokens, cost, and (when logprobs are available)
    P(prediction=1). Thread-safe — 04b runs experiments concurrently and
    each calls this once when its loop returns. Only invoked on successful
    completion; interrupted runs drop their buffer and never reach this.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    with _LLM_CALLS_LOCK:
        LLM_CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not LLM_CALLS_PATH.exists()
        df.to_csv(LLM_CALLS_PATH, mode="a", header=write_header, index=False)


# ── Experiment runner ────────────────────────────────────────────────────────

def run_llm_experiment(llm_sample, api_provider, model_name, include_desc=False,
                       api_key=None, label=None, with_logprobs=True,
                       system_prompt=None, user_prompt_fn=None):
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
        system_prompt: optional system prompt string override. If None, uses
                       build_system_prompt().
        user_prompt_fn: optional callable(row) -> str override for building
                        the user prompt. If None, uses build_user_prompt(row,
                        include_desc=include_desc).

    Returns:
        dict with 'predictions', 'probabilities', 'reasonings', 'raw_responses',
        'metrics', 'label', 'desc_tag'
    """
    import time as _time

    label = label or model_name
    desc_tag = "with_desc" if include_desc else "no_desc"
    tag = f"[{label} | {desc_tag}]"
    experiment_id = f"{label}|{desc_tag}"

    # Resolve pricing once, fail-fast on unknown model before burning API calls.
    in_price_per_1k, out_price_per_1k = get_price(model_name)

    system_prompt = system_prompt or build_system_prompt()
    y_true = llm_sample['loan_status'].values

    call_rows = []

    def _record(row_index, meta):
        in_t = meta.get("input_tokens", 0)
        out_t = meta.get("output_tokens", 0)
        cost_usd = (in_t / 1000.0) * in_price_per_1k + (out_t / 1000.0) * out_price_per_1k
        call_rows.append({
            "timestamp":             datetime.now(timezone.utc).isoformat(),
            "experiment_id":         experiment_id,
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
        })

    def _build_user_prompt(row):
        if user_prompt_fn is not None:
            return user_prompt_fn(row)
        return build_user_prompt(row, include_desc=include_desc)

    def _do_call(prompt):
        return call_llm(
            system_prompt, prompt,
            api_provider=api_provider, model=model_name, api_key=api_key,
            return_usage=True, with_logprobs=with_logprobs,
        )

    first_row = llm_sample.iloc[0]
    try:
        test_raw, test_meta = _do_call(_build_user_prompt(first_row))
        test_parsed = parse_llm_response(test_raw)
        print(f"{tag} First call OK (pred={test_parsed['prediction']}, "
              f"prob={test_meta.get('prob_fully_paid')}). Starting full run...")
    except Exception as e:
        print(f"{tag} FAILED on first call: {e}")
        raise RuntimeError(f"{tag} cannot reach API: {e}")

    _record(0, test_meta)

    predictions   = [test_parsed['prediction']]
    probabilities = [test_meta.get('prob_fully_paid')]
    reasonings    = [test_parsed['reasoning']]
    raw_responses = [test_raw]

    start = _time.time()
    for i, (_, row) in enumerate(llm_sample.iterrows()):
        if i == 0:
            continue

        raw, meta = _do_call(_build_user_prompt(row))
        raw_responses.append(raw)
        _record(i, meta)

        parsed = parse_llm_response(raw)
        predictions.append(parsed['prediction'])
        probabilities.append(meta.get('prob_fully_paid'))
        reasonings.append(parsed['reasoning'])

        if (i + 1) % 10 == 0:
            elapsed = _time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(llm_sample) - i - 1) / rate
            print(f"{tag} {i+1}/100 done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    n_errors = sum(1 for p in predictions if p is None)
    n_probs = sum(1 for p in probabilities if p is not None)
    total_time = _time.time() - start
    print(f"{tag} COMPLETE — {len(predictions)} predictions, {n_errors} parse errors, "
          f"{n_probs} with logprobs, {total_time:.0f}s total")

    _append_llm_calls(call_rows)

    metrics = evaluate_predictions(
        y_true, predictions, label=f"{label} ({desc_tag})",
        probabilities=probabilities,
    )

    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'reasonings': reasonings,
        'raw_responses': raw_responses,
        'metrics': metrics,
        'label': label,
        'desc_tag': desc_tag,
    }
