"""
regimes/gmm_pipeline.py
-----------------------
Unsupervised Gaussian Mixture Model (GMM) regime labelling.

A 3-component Gaussian Mixture is fitted on [ΔVIX, ETF log-returns] to cluster
trading days into Calm / Transitional / Crisis regimes. Mixture components are
sorted by their conditional ΔVIX mean so that 0 = Calm and 2 = Crisis.

The model is refitted every ``refit_freq`` days on an expanding window; all labels
are produced strictly out-of-sample. Unlike the Hidden Markov Model, the mixture is
memoryless (it has no transition matrix), which isolates the value of modelling
regime persistence when the two are compared.

The GMM produces *regime labels only*. Portfolio moments (mean vector and covariance
matrix) are estimated downstream from the historical days assigned to the prevailing
regime — exactly as for the VIX rule and the HMM. This keeps performance differences
attributable to the regime-labelling method itself, rather than to any auxiliary
return-forecasting model.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


# ─────────────────────────────────────────────────────────────────────────────
#  GMM REGIME LABELER
# ─────────────────────────────────────────────────────────────────────────────

def _sort_gmm_states_by_vix(gmm: GaussianMixture, dvix_col_idx: int = 0) -> np.ndarray:
    """Sort GMM components by ascending ΔVIX mean → Calm=0, Crisis=2."""
    means = gmm.means_[:, dvix_col_idx]
    return np.argsort(means)


def fit_gmm(
    X: np.ndarray,
    n_components: int = 3,
    n_init: int = 20,
    covariance_type: str = "full",
    random_state: int = 42,
) -> tuple[GaussianMixture, np.ndarray]:
    """Fit a GMM and return (fitted_model, volatility-sort order)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=random_state,
        )
        gmm.fit(X)
    order = _sort_gmm_states_by_vix(gmm, dvix_col_idx=0)
    return gmm, order


def predict_gmm_labels(gmm: GaussianMixture, order: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predict GMM hard cluster labels, remapped to 0=Calm, 1=Trans, 2=Crisis."""
    raw = gmm.predict(X)
    inverse = np.argsort(order)
    return inverse[raw]


def predict_gmm_proba(gmm: GaussianMixture, order: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Posterior GMM membership probabilities, sorted by volatility."""
    posteriors = gmm.predict_proba(X)
    return posteriors[:, order]


# ─────────────────────────────────────────────────────────────────────────────
#  GMM REGIME PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class GMMRegimePipeline:
    """GMM-only regime labeller with walk-forward refit support."""

    def __init__(
        self,
        gmm_n_components: int = 3,
        gmm_n_init: int = 20,
        gmm_covariance_type: str = "full",
        gmm_random_state: int = 42,
    ):
        self.gmm_kwargs = dict(
            n_components=gmm_n_components,
            n_init=gmm_n_init,
            covariance_type=gmm_covariance_type,
            random_state=gmm_random_state,
        )
        self.gmm_ = None
        self.gmm_order_ = None

    def fit(self, X_gmm: np.ndarray) -> "GMMRegimePipeline":
        gmm, order = fit_gmm(X_gmm, **self.gmm_kwargs)
        self.gmm_ = gmm
        self.gmm_order_ = order
        return self

    def predict_regime(self, X_gmm: np.ndarray) -> np.ndarray:
        """GMM hard-label regime assignment for new data."""
        return predict_gmm_labels(self.gmm_, self.gmm_order_, X_gmm)

    def predict_regime_proba(self, X_gmm: np.ndarray) -> np.ndarray:
        """GMM soft posterior membership probabilities."""
        return predict_gmm_proba(self.gmm_, self.gmm_order_, X_gmm)


def fit_gmm_walkforward(
    gmm_features: pd.DataFrame,    # aligned DataFrame (ΔVIX + returns)
    min_train_days: int = 504,
    refit_freq:     int = 252,
    gmm_kwargs:     dict = None,
) -> tuple[pd.Series, "GMMRegimePipeline"]:
    """
    Walk-forward expanding-window GMM regime labelling.

    At each refit point t the GMM is fitted on all data up to t and used to label
    the out-of-sample window [prev_t : t]. This mirrors the HMM walk-forward exactly,
    so the two classifiers are compared on an identical footing.

    Returns
    -------
    regime_labels : pd.Series[int]   (OOS regime labels from the GMM)
    last_pipeline : GMMRegimePipeline (last retrained pipeline)
    """
    if gmm_kwargs is None:
        gmm_kwargs = {}

    gmm_arr = gmm_features.values.astype(float)
    dates = gmm_features.index
    T = len(dates)

    regime_labels = pd.Series(index=dates, dtype=float, name="GMM_Regime")

    refit_points = list(range(min_train_days, T, refit_freq))
    if T not in refit_points:
        refit_points.append(T)

    prev_t  = min_train_days
    pipeline = None

    for t in refit_points:
        pipeline = GMMRegimePipeline(**gmm_kwargs)
        try:
            pipeline.fit(gmm_arr[:t])
        except Exception as e:
            print(f"  [GMM] Fit failed at t={t}: {e}")
            if pipeline.gmm_ is None:
                prev_t = t
                continue

        X_pred = gmm_arr[prev_t:t]
        if len(X_pred) == 0:
            prev_t = t
            continue

        regime_labels.iloc[prev_t:t] = pipeline.predict_regime(X_pred)
        prev_t = t

    regime_labels = regime_labels.dropna().astype(int)
    return regime_labels, pipeline
