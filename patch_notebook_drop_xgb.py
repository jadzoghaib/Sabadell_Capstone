"""Remove all XGBoost baseline references from 05_Prompt_Variance.ipynb."""
import json

NB = r"C:\Users\Jad Zoghaib\OneDrive\Desktop\Sabadell_Capstone\notebooks\llm_models\prompt_variance\05_Prompt_Variance.ipynb"

nb = json.load(open(NB, encoding="utf-8"))

patches = {

"cell-load-data": (
    "llm_sample = load_llm_sample()\n"
    "y_true = llm_sample['loan_status'].values\n"
    "\n"
    "print(f'Sample: {len(llm_sample)} loans | '\n"
    "      f'Fully Paid: {(y_true==1).sum()} | Charged Off: {(y_true==0).sum()}')\n"
),

"cell-phase1-table": (
    "p1_rows = []\n"
    "for v in PROMPT_VARIANTS:\n"
    "    name = v['name']\n"
    "    if name in phase1_results:\n"
    "        p1_rows.append({'variant': name, 'description': v['description'], **phase1_results[name]['metrics']})\n"
    "\n"
    "phase1_summary = pd.DataFrame(p1_rows).set_index('variant')\n"
    "display_cols = ['accuracy', 'auc', 'precision_charged_off', 'recall_charged_off', 'f1_charged_off', 'n_valid']\n"
    "print(phase1_summary[display_cols].to_string(float_format='{:.3f}'.format))\n"
),

"cell-phase1-plot": (
    "plot_df = phase1_summary.reset_index()\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
    "fig.suptitle('Phase 1 - Prompt Variant Comparison (Llama-3.3-70b, no desc)', fontsize=13)\n"
    "colors = plt.cm.tab10.colors\n"
    "\n"
    "for ax, (metric, title) in zip(axes, [\n"
    "    ('accuracy',       'Overall Accuracy'),\n"
    "    ('f1_charged_off', 'Charged Off F1'),\n"
    "    ('auc',            'AUC (logprobs)'),\n"
    "]):\n"
    "    bars = ax.barh(plot_df['variant'], plot_df[metric], color=colors[:len(plot_df)])\n"
    "    for bar, val in zip(bars, plot_df[metric]):\n"
    "        if val is not None and not (isinstance(val, float) and np.isnan(val)):\n"
    "            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,\n"
    "                    f'{val:.3f}', va='center', fontsize=8)\n"
    "    ax.set_title(title)\n"
    "    ax.set_xlim(0, 1.1)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
),

"cell-phase2-plot": (
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "fig.suptitle(f'Phase 2 - Consistency ({N_CONSISTENCY_RUNS} runs, winner: {winner_name})', fontsize=12)\n"
    "\n"
    "runs = [f'Run {i}' for i in p2_metrics.index]\n"
    "\n"
    "axes[0].bar(runs, p2_metrics['accuracy'], color='steelblue', alpha=0.8)\n"
    "axes[0].set_ylim(0, 1)\n"
    "axes[0].set_title('Accuracy per run')\n"
    "for i, v in enumerate(p2_metrics['accuracy']):\n"
    "    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)\n"
    "\n"
    "axes[1].bar(runs, p2_metrics['f1_charged_off'], color='coral', alpha=0.8)\n"
    "axes[1].set_ylim(0, 1)\n"
    "axes[1].set_title('Charged Off F1 per run')\n"
    "for i, v in enumerate(p2_metrics['f1_charged_off']):\n"
    "    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
),

"cell-phase3-sample": (
    "import os as _os\n"
    "\n"
    "from llm_utils import DATA_DIR as _DATA_DIR\n"
    "_raw_csv = _os.path.join(_DATA_DIR, '..', 'raw', 'accepted_2007_to_2018Q4.csv.gz')\n"
    "_presampled = _os.path.join(_DATA_DIR, '04c_new_batch_sample.csv')\n"
    "\n"
    "if _os.path.exists(_raw_csv):\n"
    "    print('Sampling new held-out batch from raw CSV...')\n"
    "    new_batch = sample_new_batch(n=100, random_state=99)\n"
    "else:\n"
    "    print(f'Raw CSV not found - loading pre-sampled held-out batch from {_presampled}')\n"
    "    new_batch = pd.read_csv(_presampled)\n"
    "\n"
    "y_true_new = new_batch['loan_status'].values\n"
    "print(f'New batch: {len(new_batch)} loans | '\n"
    "      f'Fully Paid: {(y_true_new==1).sum()} | Charged Off: {(y_true_new==0).sum()}')\n"
),

"cell-phase3-comparison": (
    "p1_winner_metrics = phase1_results[winner_name]['metrics']\n"
    "p3_metrics = phase3_result['metrics']\n"
    "\n"
    "robustness_df = pd.DataFrame([\n"
    "    {'split': 'Original sample (Phase 1)', **p1_winner_metrics},\n"
    "    {'split': 'New batch (Phase 3)',       **p3_metrics},\n"
    "]).set_index('split')\n"
    "\n"
    "display_cols = ['accuracy', 'auc', 'precision_charged_off', 'recall_charged_off', 'f1_charged_off']\n"
    "print('Robustness comparison:')\n"
    "print(robustness_df[display_cols].to_string(float_format='{:.3f}'.format))\n"
    "\n"
    "acc_drop = p1_winner_metrics['accuracy'] - p3_metrics['accuracy']\n"
    "f1_drop  = p1_winner_metrics['f1_charged_off'] - p3_metrics['f1_charged_off']\n"
    "print(f'\\nPerformance drop (original -> new batch):')\n"
    "print(f'  Accuracy:        {acc_drop:+.3f}')\n"
    "print(f'  Charged Off F1:  {f1_drop:+.3f}')\n"
),

"cell-phase3-plot": (
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "fig.suptitle(f'Phase 3 - Robustness (winner: {winner_name})', fontsize=12)\n"
    "\n"
    "groups = ['Original\\nsample', 'New batch']\n"
    "llm_acc = [p1_winner_metrics['accuracy'], p3_metrics['accuracy']]\n"
    "llm_f1  = [p1_winner_metrics['f1_charged_off'], p3_metrics['f1_charged_off']]\n"
    "\n"
    "x = np.arange(len(groups))\n"
    "w = 0.5\n"
    "\n"
    "for ax, (vals, title) in zip(axes, [\n"
    "    (llm_acc, 'Overall Accuracy'),\n"
    "    (llm_f1,  'Charged Off F1'),\n"
    "]):\n"
    "    bars = ax.bar(x, vals, w, label=f'Llama | {winner_name}', color='steelblue', alpha=0.85)\n"
    "    for bar, val in zip(bars, vals):\n"
    "        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center', fontsize=9)\n"
    "    ax.set_xticks(x)\n"
    "    ax.set_xticklabels(groups)\n"
    "    ax.set_ylim(0, 1.1)\n"
    "    ax.set_title(title)\n"
    "    ax.legend(fontsize=8)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
),

# Also remove run_ml_on_sample from imports
"cell-imports": (
    "import sys\n"
    "sys.path.insert(0, '..')\n"
    "\n"
    "import os\n"
    "import json\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from llm_utils import (\n"
    "    load_llm_sample,\n"
    "    sample_new_batch,\n"
    "    run_llm_experiment,\n"
    "    evaluate_predictions,\n"
    "    build_system_prompt,\n"
    "    build_user_prompt,\n"
    "    build_few_shot_examples,\n"
    "    format_loan_features,\n"
    "    FEATURE_DESCRIPTIONS,\n"
    "    RESULTS_DIR,\n"
    ")\n"
    "\n"
    "API_PROVIDER       = 'nvidia'\n"
    "MODEL_NAME         = 'meta/llama-3.3-70b-instruct'\n"
    "MODEL_LABEL        = 'Llama-3.3-70b'\n"
    "N_CONSISTENCY_RUNS = 3\n"
),

# Update export cell — robustness_df already correct, but remove xgb from phase1 table
"cell-export": (
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n"
    "\n"
    "# Phase 1 - metrics\n"
    "phase1_summary.to_csv(f'{RESULTS_DIR}/05_phase1_metrics.csv')\n"
    "\n"
    "# Per-loan predictions + reasonings\n"
    "pred_rows = []\n"
    "for v in PROMPT_VARIANTS:\n"
    "    name = v['name']\n"
    "    if name not in phase1_results:\n"
    "        continue\n"
    "    result = phase1_results[name]\n"
    "    for i, (pred, prob, reasoning) in enumerate(\n"
    "        zip(result['predictions'], result['probabilities'], result['reasonings'])\n"
    "    ):\n"
    "        pred_rows.append({\n"
    "            'phase': 1, 'variant': name, 'loan_index': i,\n"
    "            'actual': int(y_true[i]), 'prediction': pred,\n"
    "            'correct': int(pred == y_true[i]) if pred is not None else None,\n"
    "            'prob_fully_paid': prob, 'reasoning': reasoning,\n"
    "        })\n"
    "\n"
    "# Phase 2 - per-run predictions\n"
    "for run_i, result in enumerate(phase2_runs, 1):\n"
    "    for i, (pred, prob, reasoning) in enumerate(\n"
    "        zip(result['predictions'], result['probabilities'], result['reasonings'])\n"
    "    ):\n"
    "        pred_rows.append({\n"
    "            'phase': 2, 'variant': f'{winner_name}_run{run_i}', 'loan_index': i,\n"
    "            'actual': int(y_true[i]), 'prediction': pred,\n"
    "            'correct': int(pred == y_true[i]) if pred is not None else None,\n"
    "            'prob_fully_paid': prob, 'reasoning': reasoning,\n"
    "        })\n"
    "\n"
    "# Phase 3 - new batch predictions\n"
    "for i, (pred, prob, reasoning) in enumerate(\n"
    "    zip(phase3_result['predictions'], phase3_result['probabilities'], phase3_result['reasonings'])\n"
    "):\n"
    "    pred_rows.append({\n"
    "        'phase': 3, 'variant': f'{winner_name}_new_batch', 'loan_index': i,\n"
    "        'actual': int(y_true_new[i]), 'prediction': pred,\n"
    "        'correct': int(pred == y_true_new[i]) if pred is not None else None,\n"
    "        'prob_fully_paid': prob, 'reasoning': reasoning,\n"
    "    })\n"
    "\n"
    "pd.DataFrame(pred_rows).to_csv(f'{RESULTS_DIR}/05_predictions.csv', index=False)\n"
    "p2_metrics.to_csv(f'{RESULTS_DIR}/05_phase2_consistency.csv')\n"
    "robustness_df[display_cols].to_csv(f'{RESULTS_DIR}/05_phase3_robustness.csv')\n"
    "\n"
    "with open(f'{RESULTS_DIR}/05_reasonings.jsonl', 'w', encoding='utf-8') as f:\n"
    "    for row in pred_rows:\n"
    "        f.write(json.dumps(row) + '\\n')\n"
    "\n"
    "print('Saved:')\n"
    "print(f'  05_phase1_metrics.csv')\n"
    "print(f'  05_predictions.csv ({len(pred_rows)} rows)')\n"
    "print(f'  05_phase2_consistency.csv')\n"
    "print(f'  05_phase3_robustness.csv')\n"
    "print(f'  05_reasonings.jsonl  <- for promptfoo')\n"
    "print(f'\\nAll in: {RESULTS_DIR}')\n"
),
}

cell_index = {cell["id"]: i for i, cell in enumerate(nb["cells"])}

for cell_id, new_src in patches.items():
    if cell_id not in cell_index:
        print(f"WARNING: cell {cell_id} not found, skipping")
        continue
    idx = cell_index[cell_id]
    nb["cells"][idx]["source"] = new_src
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1)
print("XGBoost baseline permanently removed from notebook.")
