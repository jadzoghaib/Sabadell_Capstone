import sys
sys.path.insert(0, r"C:\Users\Jad Zoghaib\OneDrive\Desktop\Sabadell_Capstone\notebooks\llm_models")

from llm_utils import load_llm_sample, run_llm_experiment, build_system_prompt, build_user_prompt

sample = load_llm_sample().head(10)

print("Testing batch_size=0 (10 loans in 1 call)...")
result = run_llm_experiment(
    sample,
    api_provider="nvidia",
    model_name="meta/llama-3.3-70b-instruct",
    include_desc=False,
    label="batch_test",
    with_logprobs=False,
    system_prompt=build_system_prompt(),
    user_prompt_fn=lambda row: build_user_prompt(row, include_desc=False),
    batch_size=0,
)
print(f"\nPredictions: {result['predictions']}")
print(f"Parse errors: {sum(1 for p in result['predictions'] if p is None)}/10")
print(f"Metrics: acc={result['metrics']['accuracy']:.3f}, f1_charged_off={result['metrics']['f1_charged_off']:.3f}")
print("\nSample reasonings:")
for i, r in enumerate(result["reasonings"][:3]):
    print(f"  [{i}] {r[:100]}")
