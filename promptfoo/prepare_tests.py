"""
Reads data/results/llm/05_reasonings.jsonl (produced by 05_Prompt_Variance.ipynb)
and generates promptfoo/tests.yaml — one test case per Phase 1 prompt variant.

Run from the repo root:
    python promptfoo/prepare_tests.py

Each test case feeds the judge 10 reasoning samples from that variant
(a mix of correct and incorrect predictions, to surface both strengths and blind spots).
"""

import json
import random
from pathlib import Path

import yaml

REPO_ROOT     = Path(__file__).resolve().parent.parent
JSONL_PATH    = REPO_ROOT / "data" / "results" / "llm" / "05_reasonings.jsonl"
TESTS_OUT     = REPO_ROOT / "promptfoo" / "tests.yaml"
N_SAMPLES     = 10
RANDOM_SEED   = 42

VARIANT_DESCRIPTIONS = {
    "baseline":         "Standard credit analyst role, no special framing.",
    "conservative":     "Risk-averse underwriter told to err toward Charged Off when uncertain.",
    "chain_of_thought": "Asked to reason step-by-step through risk signals before predicting.",
    "few_shot":         "Given 4 labeled training examples before each prediction.",
    "top_features_only":"Only the 8 most important features (by XGBoost importance) provided.",
    "structured_4factor":"Told to evaluate using an explicit 4-factor framework: income capacity, debt burden, credit history, loan characteristics.",
}

def load_phase1_reasonings(jsonl_path):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("phase") == 1:
                records.append(rec)
    return records


def sample_reasonings(records, variant_name, n, seed):
    variant_records = [r for r in records if r["variant"] == variant_name]
    if not variant_records:
        return []
    rng = random.Random(seed)
    # Balance: half correct, half incorrect where possible
    correct   = [r for r in variant_records if r.get("correct") == 1]
    incorrect = [r for r in variant_records if r.get("correct") == 0]
    n_inc = min(n // 2, len(incorrect))
    n_cor = min(n - n_inc, len(correct))
    chosen = rng.sample(correct, n_cor) + rng.sample(incorrect, n_inc)
    rng.shuffle(chosen)
    return chosen


def format_reasonings(samples):
    lines = []
    for i, s in enumerate(samples, 1):
        outcome = "Fully Paid" if s["actual"] == 1 else "Charged Off"
        pred    = "Fully Paid" if s["prediction"] == 1 else "Charged Off"
        correct = "correct" if s.get("correct") == 1 else "WRONG"
        lines.append(
            f"[{i}] Actual: {outcome} | Predicted: {pred} ({correct})\n"
            f"    Reasoning: {s['reasoning']}"
        )
    return "\n\n".join(lines)


def main():
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found.")
        print("Run 05_Prompt_Variance.ipynb first to generate the reasonings file.")
        return

    records  = load_phase1_reasonings(JSONL_PATH)
    variants = list(dict.fromkeys(r["variant"] for r in records))  # preserve order

    test_cases = []
    for variant in variants:
        if variant not in VARIANT_DESCRIPTIONS:
            continue
        samples = sample_reasonings(records, variant, N_SAMPLES, RANDOM_SEED)
        if not samples:
            print(f"  WARNING: no Phase 1 records found for variant '{variant}', skipping.")
            continue
        test_cases.append({
            "vars": {
                "variant_name":        variant,
                "variant_description": VARIANT_DESCRIPTIONS[variant],
                "reasonings":          format_reasonings(samples),
            },
            "description": f"Qualitative characterisation of [{variant}]",
        })
        print(f"  [{variant}] — {len(samples)} samples prepared")

    TESTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TESTS_OUT, "w", encoding="utf-8") as f:
        yaml.dump(test_cases, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {len(test_cases)} test cases to {TESTS_OUT}")
    print("Next: npx promptfoo@latest eval --config promptfoo/promptfooconfig.yaml")


if __name__ == "__main__":
    main()
