import json
nb = json.load(open(r"C:\Users\Jad Zoghaib\OneDrive\Desktop\Sabadell_Capstone\notebooks\llm_models\prompt_variance\05_Prompt_Variance.ipynb", encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell["source"]
    if "run_llm_experiment" in src:
        print(f"=== {cell.get('id', '?')} ===")
        print(src[:600])
        print()
