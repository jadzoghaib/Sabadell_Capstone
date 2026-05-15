# Promptfoo — Qualitative Prompt Characterisation

Uses a Groq-hosted LLM to describe what is *distinctive* about each prompt variant's reasoning style — not scores, just qualitative fingerprints.

## Prerequisites

- Node.js installed (already on your machine)
- `05_Prompt_Variance.ipynb` has been run and produced `data/results/llm/05_reasonings.jsonl`
- Groq API key

## Steps

### 1. Add your Groq API key

In `notebooks/llm_models/.env`, add:
```
GROQ_API_KEY=your_key_here
```

Or export it in your terminal:
```
$env:GROQ_API_KEY = "your_key_here"
```

### 2. Prepare the test cases

From the repo root:
```
python promptfoo/prepare_tests.py
```

This reads `05_reasonings.jsonl`, samples 10 reasonings per variant (balanced correct/incorrect), and writes `promptfoo/tests.yaml`.

### 3. Run the evaluation

```
npx promptfoo@latest eval --config promptfoo/promptfooconfig.yaml
```

### 4. View results

```
npx promptfoo@latest view
```

Or check `promptfoo/results/qualitative_characterisations.json` directly.

## What the judge produces

For each of the 6 prompt variants, the Groq model writes a 4-6 sentence description of that variant's reasoning fingerprint — what features it anchors on, its risk posture, any systematic patterns or blind spots. No scores.

## Model

Default judge: `llama-3.3-70b-versatile` via Groq.  
To change, edit the `providers` section in `promptfooconfig.yaml`.
