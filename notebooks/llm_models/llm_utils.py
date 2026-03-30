"""
Shared utilities for LLM evaluation notebooks.

Handles: data loading, ANN re-encoding/prediction, prompt building,
LLM API calls, and evaluation metrics.
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


# ── ANN prediction on LLM sample rows ────────────────────────────────────────

def run_ann_on_sample(llm_sample):
    """
    Re-encode and scale the LLM sample, then run the ANN model.
    Returns (ann_probs, ann_preds) as arrays.
    """
    import joblib
    from tensorflow.keras.models import load_model

    scaler = joblib.load(f"{DATA_DIR}/02_scaler.joblib")
    feature_cols = joblib.load(f"{DATA_DIR}/02_feature_columns.joblib")
    model = load_model(f"{MODEL_DIR}/ann_model.keras")

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
    probs = model.predict(X_scaled, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

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

def call_llm(system_prompt, user_prompt, api_provider="anthropic", model=None, api_key=None):
    """
    Call the LLM API. Swap provider/model as needed.

    Args:
        system_prompt: System message
        user_prompt: User message
        api_provider: "anthropic" or "openai"
        model: Model name (e.g. "claude-sonnet-4-20250514", "gpt-4o")
        api_key: API key (if None, reads from environment)

    Returns:
        Raw response text from the LLM.
    """
    if api_provider == "anthropic":
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
            max_tokens=256,
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


def compare_results(y_true, llm_preds, ann_preds, llm_reasonings=None):
    """
    Build a comparison DataFrame of LLM vs ANN predictions.
    """
    df = pd.DataFrame({
        'actual': y_true,
        'llm_pred': llm_preds,
        'ann_pred': ann_preds,
        'llm_correct': [1 if p == a else 0 for p, a in zip(llm_preds, y_true)],
        'ann_correct': [1 if p == a else 0 for p, a in zip(ann_preds, y_true)],
    })
    if llm_reasonings:
        df['llm_reasoning'] = llm_reasonings
    return df
