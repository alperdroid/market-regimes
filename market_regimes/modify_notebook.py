import json

notebook_path = "/Users/alper/market_regimes.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find the summary cell index
summary_idx = len(nb["cells"])
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and any("Summary" in line for line in cell.get("source", [])):
        summary_idx = i
        break

md_cell = {
   "cell_type": "markdown",
   "id": "model_quality_md",
   "metadata": {},
   "source": [
    "---\n",
    "## Step 8 — Model Quality Metrics\n",
    "\n",
    "Statistical validation of both regime identification models:\n",
    "\n",
    "| Model | Metrics Reported |\n",
    "|-------|-----------------|\n",
    "| **HMM** | BIC/AIC model selection curves · Transition matrix persistence & half-lives · Stationary (ergodic) distribution · Regime return/volatility statistics · Residual ACF diagnostics |\n",
    "| **GMM + RF** | BIC/AIC/Silhouette model selection curves · Out-of-sample confusion matrix & F1-scores · Feature importance breakdown |"
   ]
}

code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "model_quality_code",
   "metadata": {},
   "outputs": [],
   "source": [
    "from model_quality_metrics import run_all_quality_metrics\n",
    "\n",
    "quality_results = run_all_quality_metrics(\n",
    "    hmm_feats=hmm_feats,\n",
    "    hmm_last_model=hmm_last_model,\n",
    "    hmm_regimes=hmm_regimes,\n",
    "    gmm_feats=gmm_feats,\n",
    "    ml_regimes=ml_regimes,\n",
    "    ml_last_pipeline=ml_last_pipeline,\n",
    "    rf_feature_names=rf_feature_names,\n",
    "    vix_regimes=vix_regimes,\n",
    "    excess_spy=excess_spy,\n",
    "    results_dir=RESULTS_ABS,\n",
    ")\n",
    "\n",
    "print(\"\\n✓ All model quality metrics computed and saved.\")"
   ]
}

nb["cells"].insert(summary_idx, md_cell)
nb["cells"].insert(summary_idx + 1, code_cell)

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)
    
print("Successfully modified notebook.")
