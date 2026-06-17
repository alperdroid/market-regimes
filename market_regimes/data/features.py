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


def build_feature_matrix(
    vix: pd.Series,
    ted: pd.Series,
    term: pd.Series,
    log_returns: pd.DataFrame,
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Assemble the macro-financial feature matrix for ML regime classifiers.

    Core features (aligned daily):
      - VIX level
      - ΔVIX (daily change in VIX)
      - TED Spread
      - Term Spread (10Y-2Y)

    Optional lagged return features (for RF regressor):
      - Lagged ETF log-returns for each lag in `lags`

    Returns a DataFrame indexed to the intersection of all series.
    """
    dvix = compute_vix_change(vix)

    # Align all macro features
    macro = pd.DataFrame({
        "VIX":  vix,
        "DVIX": dvix,
        "TED":  ted,
        "TERM": term,
    }).dropna()

    if lags:
        lag_frames = []
        for lag in lags:
            shifted = log_returns.shift(lag)
            shifted.columns = [f"{c}_lag{lag}" for c in shifted.columns]
            lag_frames.append(shifted)
        lagged = pd.concat(lag_frames, axis=1)
        features = pd.concat([macro, lagged], axis=1).dropna()
    else:
        features = macro

    return features


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
    Feature matrix for GMM clustering (Step 1 of ML pipeline).
    Uses ΔVIX + daily ETF log-returns.
    """
    return build_hmm_features(log_returns, vix)


def build_rf_features(
    vix: pd.Series,
    ted: pd.Series,
    term: pd.Series,
    log_returns: pd.DataFrame,
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Feature matrix for Random Forest regressors (Step 2 of ML pipeline).
    Includes VIX, TED, Term, and lagged returns.
    """
    if lags is None:
        lags = [1, 2, 3, 5]
    return build_feature_matrix(vix, ted, term, log_returns, lags=lags)


def get_common_index(*frames) -> pd.Index:
    """Return the intersection of all DataFrame/Series indices."""
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    return idx.sort_values()
