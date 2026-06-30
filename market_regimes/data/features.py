"""
data/features.py
----------------
Compute log-returns, excess returns, and the macro-financial feature matrix
used by regime classifiers and portfolio optimizers.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    r_{i,t} = ln(P_{i,t} / P_{i,t-1})
    Returns a DataFrame aligned to prices.index[1:].
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret


def compute_excess_returns(log_returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """
    e_{i,t} = r_{i,t} - rf_t
    rf is daily decimal risk-free rate.
    """
    rf_aligned = rf.reindex(log_returns.index).ffill().bfill()
    excess = log_returns.sub(rf_aligned, axis=0)
    return excess


def compute_vix_change(vix: pd.Series) -> pd.Series:
    """
    ΔVIX_t = VIX_t - VIX_{t-1}
    Used as a feature for regime classifiers.
    """
    dvix = vix.diff().dropna()
    dvix.name = "DVIX"
    return dvix


def build_hmm_features(
    log_returns: pd.DataFrame,
    vix: pd.Series,
) -> tuple[pd.DataFrame, pd.Index]:
    """
    Feature matrix for HMM training.
    Uses ΔVIX + daily ETF log-returns (as per Section 4 of the paper).
    Returns (feature_df, aligned_index).
    """
    dvix = compute_vix_change(vix)
    combined = pd.concat([dvix, log_returns], axis=1).dropna()
    return combined, combined.index


def build_gmm_features(
    log_returns: pd.DataFrame,
    vix: pd.Series,
) -> tuple[pd.DataFrame, pd.Index]:
    """
    Feature matrix for GMM regime clustering.
    Uses ΔVIX + daily ETF log-returns (identical to the HMM feature set).
    """
    return build_hmm_features(log_returns, vix)


def get_common_index(*frames) -> pd.Index:
    """Return the intersection of all DataFrame/Series indices."""
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    return idx.sort_values()
