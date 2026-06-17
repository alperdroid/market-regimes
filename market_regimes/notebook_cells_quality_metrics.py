# ============================================================================
# CELLS TO ADD TO market_regimes1.ipynb
# ============================================================================
#
# Copy the following cells into the notebook BEFORE the "Summary" section.
# Each "# ── CELL ──" block corresponds to one Jupyter notebook cell.
#
# ============================================================================


# ── CELL 1: Markdown ──────────────────────────────────────────────────────
# Cell type: markdown
# Source:
"""
---
## Step 8 — Model Quality Metrics

Statistical validation of both regime identification models:

| Model | Metrics Reported |
|-------|-----------------|
| **HMM** | BIC/AIC model selection curves · Transition matrix persistence & half-lives · Stationary (ergodic) distribution · Regime return/volatility statistics · Residual ACF diagnostics |
| **GMM + RF** | BIC/AIC/Silhouette model selection curves · Out-of-sample confusion matrix & F1-scores · Feature importance breakdown |
"""


# ── CELL 2: Code ──────────────────────────────────────────────────────────
# Import the quality metrics module
from model_quality_metrics import run_all_quality_metrics

# Run all metrics in one call
quality_results = run_all_quality_metrics(
    hmm_feats=hmm_feats,
    hmm_last_model=hmm_last_model,
    hmm_regimes=hmm_regimes,
    gmm_feats=gmm_feats,
    ml_regimes=ml_regimes,
    ml_last_pipeline=ml_last_pipeline,
    rf_feature_names=rf_feature_names,
    vix_regimes=vix_regimes,
    excess_spy=excess_spy,
    results_dir=RESULTS_ABS,
)

print("\n✓ All model quality metrics computed and saved.")
