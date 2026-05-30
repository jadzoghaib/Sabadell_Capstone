"""Patch 05_Prompt_Variance.ipynb to use batch_size=0 on all run_llm_experiment calls."""
import json, re

NB = r"C:\Users\Jad Zoghaib\OneDrive\Desktop\Sabadell_Capstone\notebooks\llm_models\prompt_variance\05_Prompt_Variance.ipynb"

nb = json.load(open(NB, encoding="utf-8"))

def get_src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s

def patch_source(src):
    # Replace with_logprobs=True or False with False and add batch_size=0 after it
    patched = re.sub(
        r"([ \t]*with_logprobs\s*=\s*(?:True|False),?\n)",
        r"            with_logprobs=False,\n            batch_size=0,\n",
        src,
    )
    return patched

changed = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = get_src(cell)
    if "run_llm_experiment" in src and "with_logprobs" in src:
        new_src = patch_source(src)
        if new_src != src:
            cell["source"] = new_src
            cell["outputs"] = []
            cell["execution_count"] = None
            changed += 1
            cid = cell.get("id", "?")
            print(f"Patched cell: {cid}")
            print("--- preview ---")
            for line in new_src.splitlines():
                if "with_logprobs" in line or "batch_size" in line:
                    print(f"  {line}")
            print()

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1)
print(f"\n{changed} cell(s) patched.")
