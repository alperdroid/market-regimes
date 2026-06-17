"""
regimes/vix_classifier.py
--------------------------
Rule-based VIX regime classification.
Literature-standard fixed thresholds (Whaley 2009 JPM; Bloom 2009 NBER;
Ang & Bekaert 2004 RFS; CBOE; S&P Global):
  Calm         VIX < 20
  Transitional 20 ≤ VIX < 30
  Crisis       VIX ≥ 30
"""

import numpy as np
import pandas as pd


def classify_vix_regimes(
    vix: pd.Series,
    calm_threshold: float = 20.0,
    transitional_threshold: float = 30.0,
) -> pd.Series:
    """
    Assign integer regime label to each day based on VIX level.

    Returns
    -------
    pd.Series[int]
        0 = Calm, 1 = Transitional, 2 = Crisis
    """
    labels = pd.Series(index=vix.index, dtype=int)
    labels[vix < calm_threshold]                           = 0   # Calm
    labels[(vix >= calm_threshold) & (vix < transitional_threshold)] = 1   # Transitional
    labels[vix >= transitional_threshold]                  = 2   # Crisis
    labels.name = "VIX_Regime"
    return labels


def regime_statistics(
    vix: pd.Series,
    labels: pd.Series,
    log_returns: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each VIX-defined regime.

    Returns
    -------
    pd.DataFrame
        Rows: regimes. Columns: count, pct_days, mean_vix, std_vix,
        plus mean/std returns per ETF if log_returns provided.
    """
    regime_names = {0: "Calm", 1: "Transitional", 2: "Crisis"}
    rows = []
    for code, name in regime_names.items():
        mask = labels == code
        row = {
            "Regime": name,
            "Days": mask.sum(),
            "Pct_Days": mask.mean() * 100,
            "Mean_VIX": vix[mask].mean(),
            "Std_VIX":  vix[mask].std(),
        }
        if log_returns is not None:
            # Align mask to the log_returns index to avoid shape mismatches
            mask_aligned = mask.reindex(log_returns.index).fillna(False)
            ret_sub = log_returns.loc[mask_aligned]
            if len(ret_sub):
                row["Mean_EqWt_Ret"] = ret_sub.mean(axis=1).mean() * 252
                row["Std_EqWt_Ret"]  = ret_sub.mean(axis=1).std() * np.sqrt(252)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Regime")
