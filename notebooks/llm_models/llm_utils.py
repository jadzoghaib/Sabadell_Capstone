"""
Shared utilities for LLM evaluation notebooks.

Handles: data loading, ML re-encoding/prediction, prompt building,
LLM API calls, evaluation metrics, and full experiment loops.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)

DATA_DIR = "../../data/processed"
MODEL_DIR = "../../models"
RESULTS_DIR = "../../data/results/llm"

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


def sample_new_batch(n=100, random_state=99):
    """
    Sample a different batch of n loans from the test set (with descriptions).
    Uses the same preprocessing logic as 02_Preprocessing to select from test rows.
    """
    from sklearn.model_selection import train_test_split

    full_data = pd.read_csv("../../data/raw/accepted_2007_to_2018Q4.csv.gz", low_memory=False)
    keep_cols = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
        'loan_status', 'purpose', 'dti', 'earliest_cr_line', 'open_acc',
        'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'initial_list_status', 'application_type', 'mort_acc',
        'pub_rec_bankruptcies', 'zip_code', 'addr_state', 'desc'
    ]
    full_data = full_data[keep_cols].copy()
    full_data = full_data[full_data['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    full_data['issue_d'] = pd.to_datetime(full_data['issue_d'], format='%b-%Y')
    full_data = full_data[
        (full_data['issue_d'].dt.year >= 2012) &
        (full_data['issue_d'].dt.year <= 2014)
    ]
    full_data['loan_status'] = full_data.loan_status.map({'Fully Paid': 1, 'Charged Off': 0})

    # Same split as preprocessing (random_state=42)
    _, test_data = train_test_split(full_data, test_size=0.33, random_state=42)

    # Only rows with descriptions
    test_with_desc = test_data[test_data['desc'].notna() & (test_data['desc'].str.strip() != '')]

    # Exclude the original 100 sample rows
    original = load_llm_sample()
    original_idx = set(original.index) if 'index' not in original.columns else set()
    # Use a content-based dedup: match on loan_amnt + int_rate + annual_inc
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
    from sklearn.model_selection import train_test_split

    # Reload the full processed data to get training rows
    full_data = pd.read_csv("../../data/raw/accepted_2007_to_2018Q4.csv.gz",
                            low_memory=False)
    keep_cols = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
        'loan_status', 'purpose', 'dti', 'earliest_cr_line', 'open_acc',
        'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'initial_list_status', 'application_type', 'mort_acc',
        'pub_rec_bankruptcies', 'desc'
    ]
    full_data = full_data[keep_cols].copy()
    full_data = full_data[full_data['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    full_data['issue_d'] = pd.to_datetime(full_data['issue_d'], format='%b-%Y')
    full_data = full_data[
        (full_data['issue_d'].dt.year >= 2012) &
        (full_data['issue_d'].dt.year <= 2014)
    ]
    full_data['loan_status'] = full_data.loan_status.map({'Fully Paid': 1, 'Charged Off': 0})

    # Use the same split as preprocessing to get training rows only
    train_data, _ = train_test_split(full_data, test_size=0.33, random_state=42)

    # Sample balanced examples
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
    """Load .env file into os.environ (once)."""
    import os
    from pathlib import Path

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


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


def call_llm(system_prompt, user_prompt, api_provider="gemini", model=None, api_key=None):
    """
    Call the LLM API. Supports gemini, anthropic, and openai providers.

    Args:
        system_prompt: System message
        user_prompt: User message
        api_provider: "gemini", "anthropic", or "openai"
        model: Model name (if None, uses default for provider)
        api_key: API key (if None, reads from .env / environment)

    Returns:
        Raw response text from the LLM.
    """
    if api_key is None:
        api_key = load_api_key(api_provider, model=model)

    if api_provider == "gemini":
        import time
        from google import genai
        client = genai.Client(api_key=api_key)
        model = model or "gemini-2.5-flash"
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                )
                return response.text
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
        return response.content[0].text

    elif api_provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        model = model or "gpt-4o"
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

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

def evaluate_predictions(y_true, y_pred, label="Model"):
    """Print classification metrics for a set of predictions."""
    valid = [i for i in range(len(y_pred)) if y_pred[i] is not None]
    if len(valid) < len(y_pred):
        print(f"Warning: {len(y_pred) - len(valid)} unparseable predictions excluded")

    y_true_v = np.array(y_true)[valid]
    y_pred_v = np.array(y_pred)[valid]

    print(f"\n{'=' * 50}")
    print(f"{label} Results ({len(valid)} samples)")
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy_score(y_true_v, y_pred_v) * 100:.1f}%")
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


# ── Experiment runner ────────────────────────────────────────────────────────

def run_llm_experiment(llm_sample, api_provider, model_name, include_desc=False,
                       api_key=None, label=None):
    """
    Run the full LLM prediction loop on a sample.

    Args:
        llm_sample: DataFrame with loan features
        api_provider: "gemini", "anthropic", or "openai"
        model_name: model identifier string
        include_desc: whether to include borrower description
        api_key: optional API key override
        label: display label (defaults to model_name)

    Returns:
        dict with 'predictions', 'reasonings', 'raw_responses', 'metrics', 'label'
    """
    import time as _time

    label = label or model_name
    desc_tag = "with_desc" if include_desc else "no_desc"
    tag = f"[{label} | {desc_tag}]"

    system_prompt = build_system_prompt()
    y_true = llm_sample['loan_status'].values

    # Fail-fast: test first call before committing to the full loop
    first_row = llm_sample.iloc[0]
    test_prompt = build_user_prompt(first_row, include_desc=include_desc)
    try:
        test_raw = call_llm(system_prompt, test_prompt,
                            api_provider=api_provider, model=model_name, api_key=api_key)
        test_parsed = parse_llm_response(test_raw)
        print(f"{tag} First call OK (pred={test_parsed['prediction']}). Starting full run...")
    except Exception as e:
        print(f"{tag} FAILED on first call: {e}")
        raise RuntimeError(f"{tag} cannot reach API: {e}")

    predictions = [test_parsed['prediction']]
    reasonings = [test_parsed['reasoning']]
    raw_responses = [test_raw]

    start = _time.time()
    for i, (_, row) in enumerate(llm_sample.iterrows()):
        if i == 0:
            continue  # already done above

        user_prompt = build_user_prompt(row, include_desc=include_desc)
        raw = call_llm(system_prompt, user_prompt,
                       api_provider=api_provider, model=model_name, api_key=api_key)
        raw_responses.append(raw)

        parsed = parse_llm_response(raw)
        predictions.append(parsed['prediction'])
        reasonings.append(parsed['reasoning'])

        # Progress every 10 rows
        if (i + 1) % 10 == 0:
            elapsed = _time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(llm_sample) - i - 1) / rate
            print(f"{tag} {i+1}/100 done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    n_errors = sum(1 for p in predictions if p is None)
    total_time = _time.time() - start
    print(f"{tag} COMPLETE — {len(predictions)} predictions, {n_errors} parse errors, {total_time:.0f}s total")

    metrics = evaluate_predictions(y_true, predictions, label=f"{label} ({desc_tag})")

    return {
        'predictions': predictions,
        'reasonings': reasonings,
        'raw_responses': raw_responses,
        'metrics': metrics,
        'label': label,
        'desc_tag': desc_tag,
    }
