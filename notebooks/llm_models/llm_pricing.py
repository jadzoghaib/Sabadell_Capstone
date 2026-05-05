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

PRICING_DATE = "2026-01"

# (input_price_per_1k_usd, output_price_per_1k_usd)
PRICES = {
    # OpenAI
    "gpt-5":            (0.00125, 0.01000),

    # Google Gemini (standard ≤200k-token context tier)
    "gemini-2.5-pro":   (0.00125, 0.01000),
    "gemini-2.5-flash": (0.00030, 0.00250),

    # Anthropic (not used in current experiments — placeholder)
    "claude-sonnet-4-20250514": (0.00300, 0.01500),
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
