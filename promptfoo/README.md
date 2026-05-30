# Promptfoo — Qualitative Prompt Characterisation

Uses NVIDIA NIM (llama-3.3-70b-instruct) as the judge to describe what is *distinctive* about each prompt variant's reasoning style — not scores, just qualitative fingerprints.

## Prerequisites

- Node.js installed (already on your machine)
- `05_Prompt_Variance.ipynb` has been run and produced `data/results/llm/05_reasonings.jsonl`
- `NVIDIA_API_KEY` set in `.env` at repo root (already done)

## Run order

### 1. Run the notebook first

Open and run `notebooks/llm_models/prompt_variance/05_Prompt_Variance.ipynb` top to bottom.
This produces `data/results/llm/05_reasonings.jsonl`.

### 2. Prepare test cases

From the repo root:
```powershell
python promptfoo/prepare_tests.py
```

Reads the JSONL, samples 10 reasonings per variant (balanced correct/incorrect), writes `promptfoo/tests.yaml`.

### 3. Load the API key into your shell

```powershell
$env:NVIDIA_API_KEY = (Get-Content .env | Select-String "NVIDIA_API_KEY").ToString().Split("=")[1]
```

### 4. Run the evaluation

```powershell
npx promptfoo@latest eval --config promptfoo/promptfooconfig.yaml
```

### 5. View results

```powershell
npx promptfoo@latest view
```

Or inspect `promptfoo/results/qualitative_characterisations.json` directly.

## What the judge produces

For each of the 6 prompt variants, Llama-3.3-70b writes a 4-6 sentence description of that variant's reasoning fingerprint — what features it anchors on, its risk posture, any systematic patterns or blind spots. No scores.

## Model

Judge: `meta/llama-3.3-70b-instruct` via NVIDIA NIM (`https://integrate.api.nvidia.com/v1`).  
To change, edit the `providers` section in `promptfooconfig.yaml`.
