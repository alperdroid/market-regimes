"""
regimes/ensemble.py
-------------------
Utilities for combining regime labels from multiple classifiers.
"""

import numpy as np
import pandas as pd


def majority_vote_regimes(
    vix_regimes: pd.Series,
    hmm_regimes: pd.Series,
    ml_regimes: pd.Series,
    name: str = "Ensemble_Regime",
) -> pd.Series:
    """
    Combine VIX, HMM, and ML regime labels by daily majority vote.

    Labels are expected to be integer encoded as 0=Calm, 1=Transitional,
    2=Crisis. When all three classifiers disagree, the median label is used,
    which maps the 0/1/2 split to Transitional.
    """
    common_idx = (
        vix_regimes.index
        .intersection(hmm_regimes.index)
        .intersection(ml_regimes.index)
        .sort_values()
    )
    labels = pd.DataFrame(
        {
            "vix": vix_regimes.reindex(common_idx).ffill().bfill().astype(int),
            "hmm": hmm_regimes.reindex(common_idx).ffill().bfill().astype(int),
            "ml": ml_regimes.reindex(common_idx).ffill().bfill().astype(int),
        },
        index=common_idx,
    )

    def vote(row: pd.Series) -> int:
        values = row.to_numpy(dtype=int)
        counts = np.bincount(values, minlength=3)
        if counts.max() >= 2:
            return int(counts.argmax())
        return int(np.median(values))

    return labels.apply(vote, axis=1).astype(int).rename(name)
