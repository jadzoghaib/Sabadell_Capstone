"""
Per-model token pricing for cost tracking.

Prices are USD per 1,000 tokens (input and output priced separately).
The exact prices used at call time are recorded in
data/results/llm/llm_calls.csv, so updating values here only affects future
runs — historical cost rows stay correct against whatever price was logged.

VERIFY against current provider pricing pages before relying on totals:
  - OpenAI:  https://openai.com/api/pricing/
  - Gemini:  https://ai.google.dev/pricing
  - Claude:  https://www.anthropic.com/pricing#api
"""

PRICING_DATE = "2026-05"

# (input_price_per_1k_usd, output_price_per_1k_usd)
PRICES = {
    # ── Current models (May 2026) ────────────────────────────────────────────

    # OpenAI — gpt-5.4 released 2026-03-05; alias resolves to gpt-5.4-2026-03-05
    "gpt-5.4":          (0.00250, 0.01500),

    # Google Gemini — 3.1 Pro Preview is the current stable Pro tier (≤200k ctx)
    "gemini-3.1-pro-preview": (0.00200, 0.01200),
    "gemini-3.5-flash":        (0.00150, 0.00900),

    # Anthropic — Sonnet 4.6 and Opus 4.8 (latest as of May 2026)
    "claude-sonnet-4-6": (0.00300, 0.01500),
    "claude-opus-4-8":   (0.00500, 0.02500),

    # NVIDIA NIM (phase 02) — verify at https://build.nvidia.com/pricing
    "meta/llama-3.3-70b-instruct": (0.00077, 0.00077),

    # Groq (legacy — early Phase 3 exploration; the committed Phase 3 uses GPT-5.4.
    # Kept so any historical llm_calls.csv rows stay accurate) — https://groq.com/pricing
    "llama-3.3-70b-versatile":     (0.00059, 0.00079),

    # ── Legacy entries — kept so historical llm_calls.csv rows stay accurate ─
    "gpt-5":                   (0.00125, 0.01000),
    "gemini-2.5-pro":          (0.00125, 0.01000),
    "gemini-2.5-flash":        (0.00030, 0.00250),
    "claude-sonnet-4-20250514": (0.00300, 0.01500),
    "claude-opus-4-7":         (0.00500, 0.02500),
}


def get_price(model):
    """Return (input_price_per_1k, output_price_per_1k) in USD for `model`."""
    if model not in PRICES:
        raise KeyError(
            f"No pricing entry for model {model!r}. "
            f"Add it to PRICES in llm_pricing.py. "
            f"Known models: {sorted(PRICES.keys())}"
        )
    return PRICES[model]


def compute_cost(model, input_tokens, output_tokens):
    """
    Cost in USD for one call. Returns
    (cost_usd, input_price_per_1k_usd, output_price_per_1k_usd) so the caller
    can log the prices used alongside the computed cost.
    """
    in_p, out_p = get_price(model)
    cost = (input_tokens / 1000.0) * in_p + (output_tokens / 1000.0) * out_p
    return cost, in_p, out_p
