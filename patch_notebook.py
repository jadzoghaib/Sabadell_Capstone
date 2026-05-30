import json

nb = json.load(open(
    r"C:\Users\Jad Zoghaib\OneDrive\Desktop\Sabadell_Capstone\notebooks\llm_models\prompt_variance\05_Prompt_Variance.ipynb",
    encoding="utf-8"
))

patches = {

"cell-load-data": (
    "llm_sample = load_llm_sample()\n"
    "y_true = llm_sample['loan_status'].values\n"
    "\n"
    "_xgb_model_path = os.path.join(os.path.dirname(os.path.dirname(RESULTS_DIR)), 'models', 'xgb_model.joblib')\n"
    "_HAS_XGB = os.path.exists(_xgb_model_path)\n"
    "\n"
    "if _HAS_XGB:\n"
    "    xgb_probs, xgb_preds = run_ml_on_sample(llm_sample)\n"
    "    xgb_metrics = evaluate_predictions(y_true, xgb_preds.tolist(), label='XGBoost (baseline)')\n"
    "    print('XGBoost baseline loaded.')\n"
    "else:\n"
    "    xgb_probs, xgb_preds, xgb_metrics = None, None, {}\n"
    "    print('XGBoost model not found — skipping baseline (run 03_Modeling.ipynb to generate it).')\n"
    "\n"
    "print(f'\\nSample: {len(llm_sample)} loans | '\n"
    "      f'Fully Paid: {(y_true==1).sum()} | Charged Off: {(y_true==0).sum()}')\n"
),

"cell-phase1-table": (
    "p1_rows = []\n"
    "if xgb_metrics:\n"
    "    p1_rows.append({'variant': 'XGBoost', 'description': 'Gradient boosting on 66 features', **xgb_metrics})\n"
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
    "llm_df  = plot_df[plot_df['variant'] != 'XGBoost']\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
    "fig.suptitle('Phase 1 - Prompt Variant Comparison (Llama-3.3-70b, no desc)', fontsize=13)\n"
    "colors = plt.cm.tab10.colors\n"
    "\n"
    "for ax, (metric, title, xgb_val) in zip(axes, [\n"
    "    ('accuracy',       'Overall Accuracy',  xgb_metrics.get('accuracy')),\n"
    "    ('f1_charged_off', 'Charged Off F1',     xgb_metrics.get('f1_charged_off')),\n"
    "    ('auc',            'AUC (logprobs)',      xgb_metrics.get('auc')),\n"
    "]):\n"
    "    bars = ax.barh(llm_df['variant'], llm_df[metric], color=colors[:len(llm_df)])\n"
    "    if xgb_val is not None:\n"
    "        ax.axvline(xgb_val, color='black', linestyle='--', linewidth=1.2,\n"
    "                   label=f'XGBoost ({xgb_val:.3f})')\n"
    "        ax.legend(fontsize=8)\n"
    "    for bar, val in zip(bars, llm_df[metric]):\n"
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
    "if xgb_metrics.get('accuracy') is not None:\n"
    "    axes[0].axhline(xgb_metrics['accuracy'], color='black', linestyle='--',\n"
    "                    label=f\"XGBoost ({xgb_metrics['accuracy']:.3f})\")\n"
    "    axes[0].legend(fontsize=8)\n"
    "axes[0].set_ylim(0, 1)\n"
    "axes[0].set_title('Accuracy per run')\n"
    "for i, v in enumerate(p2_metrics['accuracy']):\n"
    "    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)\n"
    "\n"
    "axes[1].bar(runs, p2_metrics['f1_charged_off'], color='coral', alpha=0.8)\n"
    "if xgb_metrics.get('f1_charged_off') is not None:\n"
    "    axes[1].axhline(xgb_metrics['f1_charged_off'], color='black', linestyle='--',\n"
    "                    label=f\"XGBoost ({xgb_metrics['f1_charged_off']:.3f})\")\n"
    "    axes[1].legend(fontsize=8)\n"
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
    "_raw_csv = _os.path.join(_os.path.dirname(RESULTS_DIR), '..', 'raw', 'accepted_2007_to_2018Q4.csv.gz')\n"
    "_presampled = _os.path.join(_os.path.dirname(RESULTS_DIR), 'processed', '04c_new_batch_sample.csv')\n"
    "\n"
    "if _os.path.exists(_raw_csv):\n"
    "    print('Sampling new held-out batch from raw CSV...')\n"
    "    new_batch = sample_new_batch(n=100, random_state=99)\n"
    "else:\n"
    "    print(f'Raw CSV not found - loading pre-sampled held-out batch from {_presampled}')\n"
    "    new_batch = pd.read_csv(_presampled)\n"
    "\n"
    "y_true_new = new_batch['loan_status'].values\n"
    "\n"
    "if _HAS_XGB:\n"
    "    xgb_probs_new, xgb_preds_new = run_ml_on_sample(new_batch)\n"
    "    xgb_new_metrics = evaluate_predictions(y_true_new, xgb_preds_new.tolist(), label='XGBoost (new batch)')\n"
    "else:\n"
    "    xgb_new_metrics = {}\n"
    "\n"
    "print(f'\\nNew batch: {len(new_batch)} loans | '\n"
    "      f'Fully Paid: {(y_true_new==1).sum()} | Charged Off: {(y_true_new==0).sum()}')\n"
),

"cell-phase3-comparison": (
    "p1_winner_metrics = phase1_results[winner_name]['metrics']\n"
    "p3_metrics = phase3_result['metrics']\n"
    "\n"
    "comparison_rows = [\n"
    "    {'split': 'Original sample (Phase 1)', **p1_winner_metrics},\n"
    "    {'split': 'New batch (Phase 3)',       **p3_metrics},\n"
    "]\n"
    "if xgb_metrics:\n"
    "    comparison_rows.append({'split': 'XGBoost - original', **xgb_metrics})\n"
    "if xgb_new_metrics:\n"
    "    comparison_rows.append({'split': 'XGBoost - new batch', **xgb_new_metrics})\n"
    "\n"
    "robustness_df = pd.DataFrame(comparison_rows).set_index('split')\n"
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
    "w = 0.35 if xgb_new_metrics else 0.6\n"
    "\n"
    "for ax, (llm_vals, title, xgb_key) in zip(axes, [\n"
    "    (llm_acc, 'Overall Accuracy', 'accuracy'),\n"
    "    (llm_f1,  'Charged Off F1',   'f1_charged_off'),\n"
    "]):\n"
    "    if xgb_new_metrics:\n"
    "        xgb_vals = [xgb_metrics.get(xgb_key), xgb_new_metrics.get(xgb_key)]\n"
    "        b1 = ax.bar(x - w/2, llm_vals, w, label=f'Llama | {winner_name}', color='steelblue', alpha=0.85)\n"
    "        b2 = ax.bar(x + w/2, xgb_vals, w, label='XGBoost', color='coral', alpha=0.85)\n"
    "        for bar in list(b1) + list(b2):\n"
    "            h = bar.get_height()\n"
    "            if h is not None:\n"
    "                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', fontsize=8)\n"
    "        ax.legend(fontsize=8)\n"
    "    else:\n"
    "        b1 = ax.bar(x, llm_vals, w, label=f'Llama | {winner_name}', color='steelblue', alpha=0.85)\n"
    "        for bar, val in zip(b1, llm_vals):\n"
    "            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center', fontsize=8)\n"
    "        ax.legend(fontsize=8)\n"
    "    ax.set_xticks(x)\n"
    "    ax.set_xticklabels(groups)\n"
    "    ax.set_ylim(0, 1.1)\n"
    "    ax.set_title(title)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
),
}

cell_index = {cell["id"]: i for i, cell in enumerate(nb["cells"])}

for cell_id, new_src in patches.items():
    idx = cell_index[cell_id]
    nb["cells"][idx]["source"] = new_src
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

json.dump(nb, open(
    r"C:\Users\Jad Zoghaib\OneDrive\Desktop\Sabadell_Capstone\notebooks\llm_models\prompt_variance\05_Prompt_Variance.ipynb",
    "w", encoding="utf-8"
), indent=1)
print("Notebook patched successfully.")
