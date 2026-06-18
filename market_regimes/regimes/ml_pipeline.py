"""
regimes/ml_pipeline.py
-----------------------
Decoupled Machine Learning regime pipeline (Section 2.2):

Step 1 — Unsupervised Regime Labeling:
    GaussianMixture (GMM) fitted on [ΔVIX, ETF log-returns] to cluster
    historical data into 3 regime states.

Step 2 — Supervised Return Forecasting:
    Three independent supervised regressors (one per regime) trained on
    regime-specific data subsets to forecast next-day ETF log-returns.

Walk-Forward:
    The entire pipeline is retrained every `refit_freq` days using an
    expanding window of historical data.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


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
    """
    Fit GMM and return (fitted_model, sort_order).
    """
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


def predict_gmm_labels(
    gmm: GaussianMixture,
    order: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Predict GMM hard cluster labels, remapped to 0=Calm, 1=Trans, 2=Crisis."""
    raw = gmm.predict(X)
    inverse = np.argsort(order)
    return inverse[raw]


def predict_gmm_proba(
    gmm: GaussianMixture,
    order: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Posterior GMM membership probabilities, sorted by volatility."""
    posteriors = gmm.predict_proba(X)
    return posteriors[:, order]


# ─────────────────────────────────────────────────────────────────────────────
#  REGIME-SPECIALIST SUPERVISED FORECASTERS
# ─────────────────────────────────────────────────────────────────────────────

class RegimeForecastEnsemble:
    """
    Container for regime-specific supervised return forecasters.
    Each model is trained independently on its regime's data.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        model_type: str = "random_forest",
        n_estimators: int = 300,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        max_features: float | str | None = 1.0,
        learning_rate: float = 0.05,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_regimes = n_regimes
        self.model_type = model_type
        self.models = [
            self._make_model(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            for _ in range(n_regimes)
        ]
        self._fitted = [False] * n_regimes

    def _make_model(
        self,
        n_estimators: int,
        max_depth: int | None,
        min_samples_leaf: int,
        max_features: float | str | None,
        learning_rate: float,
        l2_regularization: float,
        random_state: int,
        n_jobs: int,
    ):
        model_type = self.model_type.lower()
        if model_type in {"rf", "random_forest"}:
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        if model_type in {"et", "extra_trees", "extratrees"}:
            return ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        if model_type in {"hgb", "hist_gradient_boosting"}:
            return MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=n_estimators,
                    max_leaf_nodes=31,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    random_state=random_state,
                ),
                n_jobs=n_jobs,
            )
        raise ValueError(
            "Unsupported supervised ML model. "
            "Use 'random_forest', 'extra_trees', or 'hist_gradient_boosting'."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
    ) -> "RegimeForestEnsemble":
        """
        Train each specialist model on its regime's data slice.

        Parameters
        ----------
        X : (T, n_features) feature matrix
        y : (T, n_assets)  next-day log-return targets
        labels : (T,) integer regime labels
        """
        for r in range(self.n_regimes):
            mask = labels == r
            if mask.sum() < 10:
                # insufficient data for this regime — leave unfitted
                continue
            X_r = X[mask]
            y_r = y[mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models[r].fit(X_r, y_r)
            self._fitted[r] = True
        return self

    def predict(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Predict next-day returns using the specialist model for the given regime.

        Parameters
        ----------
        X      : (T, n_features)
        labels : (T,) predicted regime for each row

        Returns
        -------
        np.ndarray (T, n_assets)
        """
        preds = np.zeros((len(X), self.models[0].n_outputs_
                          if self._fitted[0] else 1))
        # Determine output shape from first fitted model
        for r in range(self.n_regimes):
            if self._fitted[r]:
                n_out = self.models[r].n_outputs_
                preds = np.zeros((len(X), n_out))
                break

        for r in range(self.n_regimes):
            mask = labels == r
            if not mask.any():
                continue
            if self._fitted[r]:
                preds[mask] = self.models[r].predict(X[mask])
            else:
                # Fallback: use any fitted model
                for fallback in range(self.n_regimes):
                    if self._fitted[fallback]:
                        preds[mask] = self.models[fallback].predict(X[mask])
                        break
        return preds

    def feature_importances(self) -> dict:
        """Return feature importances per regime (for interpretation)."""
        importances = {}
        for r in range(self.n_regimes):
            if self._fitted[r] and hasattr(self.models[r], "feature_importances_"):
                importances[r] = self.models[r].feature_importances_
        return importances


# Backwards-compatible name used by older notebooks.
RegimeForestEnsemble = RegimeForecastEnsemble


# ─────────────────────────────────────────────────────────────────────────────
#  FULL ML PIPELINE — WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────

class MLRegimePipeline:
    """
    Full decoupled ML pipeline:
      1. GMM clustering on [ΔVIX, log-returns] → regime labels
      2. Regime-specific supervised regressors → next-day return forecasts

    Supports walk-forward expanding-window retraining.
    """

    def __init__(
        self,
        gmm_n_components: int = 3,
        gmm_n_init: int = 20,
        gmm_covariance_type: str = "full",
        gmm_random_state: int = 42,
        forecast_model: str = "random_forest",
        rf_n_estimators: int = 300,
        rf_max_depth: int = 5,
        rf_min_samples: int = 20,
        rf_max_features: float | str | None = 1.0,
        gb_learning_rate: float = 0.05,
        gb_l2_regularization: float = 0.0,
        rf_random_state: int = 42,
        rf_n_jobs: int = -1,
    ):
        self.gmm_kwargs = dict(
            n_components=gmm_n_components,
            n_init=gmm_n_init,
            covariance_type=gmm_covariance_type,
            random_state=gmm_random_state,
        )
        self.rf_kwargs = dict(
            model_type=forecast_model,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples,
            max_features=rf_max_features,
            learning_rate=gb_learning_rate,
            l2_regularization=gb_l2_regularization,
            random_state=rf_random_state,
            n_jobs=rf_n_jobs,
        )
        self.gmm_      = None
        self.gmm_order_ = None
        self.ensemble_ = None

    def fit(
        self,
        X_gmm: np.ndarray,     # Features for GMM (ΔVIX + returns)
        X_rf: np.ndarray,      # Forecasting features (macro + lagged returns)
        y_rf: np.ndarray,      # Next-day return targets (T, n_assets)
    ) -> "MLRegimePipeline":
        """
        Full in-sample fit.
          1. Fit GMM on X_gmm
          2. Label training data
          3. Fit regime-specific supervised forecasting ensemble
        """
        # Step 1: GMM clustering
        gmm, order = fit_gmm(X_gmm, **self.gmm_kwargs)
        self.gmm_       = gmm
        self.gmm_order_ = order

        # Step 2: Label training data with GMM
        labels = predict_gmm_labels(gmm, order, X_gmm)

        # Step 3: Align RF features with GMM labels (shapes may differ slightly)
        n = min(len(X_rf), len(y_rf), len(labels))
        X_rf_   = X_rf[-n:]
        y_rf_   = y_rf[-n:]
        labels_ = labels[-n:]

        # Step 4: Train regime-specific supervised forecasters
        ensemble = RegimeForecastEnsemble(
            n_regimes=self.gmm_kwargs["n_components"],
            **self.rf_kwargs,
        )
        ensemble.fit(X_rf_, y_rf_, labels_)
        self.ensemble_ = ensemble
        return self

    def predict_regime(self, X_gmm: np.ndarray) -> np.ndarray:
        """GMM hard-label regime assignment for new data."""
        return predict_gmm_labels(self.gmm_, self.gmm_order_, X_gmm)

    def predict_regime_proba(self, X_gmm: np.ndarray) -> np.ndarray:
        """GMM soft posterior membership probabilities."""
        return predict_gmm_proba(self.gmm_, self.gmm_order_, X_gmm)

    def predict_returns(
        self,
        X_gmm: np.ndarray,
        X_rf: np.ndarray,
    ) -> np.ndarray:
        """
        Two-step prediction:
          1. Predict regime from X_gmm
          2. Query corresponding regime specialist for return forecasts
        """
        labels = self.predict_regime(X_gmm)
        return self.ensemble_.predict(X_rf, labels)


def fit_ml_walkforward(
    gmm_features: pd.DataFrame,    # aligned DataFrame (ΔVIX + returns)
    rf_features:  pd.DataFrame,    # aligned forecasting features (macro + lags)
    log_returns:  pd.DataFrame,    # next-day targets (sector ETFs only)
    min_train_days: int = 504,
    refit_freq:     int = 252,
    ml_kwargs:      dict = None,
) -> tuple[pd.Series, pd.DataFrame, MLRegimePipeline]:
    """
    Walk-forward expanding-window ML pipeline.

    Returns
    -------
    ml_regime_labels : pd.Series[int]  (OOS regime labels from GMM)
    ml_return_preds  : pd.DataFrame    (OOS return forecasts)
    last_pipeline    : MLRegimePipeline (last retrained pipeline)
    """
    if ml_kwargs is None:
        ml_kwargs = {}

    # Align all inputs on common dates
    common_idx = (
        gmm_features.index
        .intersection(rf_features.index)
        .intersection(log_returns.index)
        .sort_values()
    )
    gmm_arr = gmm_features.loc[common_idx].values.astype(float)
    rf_arr  = rf_features.loc[common_idx].values.astype(float)
    ret_arr = log_returns.loc[common_idx].values.astype(float)
    dates   = common_idx

    T = len(dates)
    regime_labels = pd.Series(index=dates, dtype=float, name="ML_Regime")
    return_preds  = pd.DataFrame(
        index=dates, columns=log_returns.columns, dtype=float
    )

    refit_points = list(range(min_train_days, T, refit_freq))
    if T not in refit_points:
        refit_points.append(T)

    prev_t  = min_train_days
    pipeline = None

    for t in refit_points:
        # Build targets: next-day returns (shift by 1)
        y_train = ret_arr[1:t]          # targets for days 1..t-1
        X_gmm_train = gmm_arr[:t-1]     # features for days 0..t-2
        X_rf_train  = rf_arr[:t-1]

        pipeline = MLRegimePipeline(**ml_kwargs)
        try:
            pipeline.fit(X_gmm_train, X_rf_train, y_train)
        except Exception as e:
            print(f"  [ML] Fit failed at t={t}: {e}")
            if pipeline.gmm_ is None:
                prev_t = t
                continue

        # Predict on window [prev_t : t]
        X_gmm_pred = gmm_arr[prev_t:t]
        X_rf_pred  = rf_arr[prev_t:t]

        if len(X_gmm_pred) == 0:
            prev_t = t
            continue

        reg = pipeline.predict_regime(X_gmm_pred)
        ret = pipeline.predict_returns(X_gmm_pred, X_rf_pred)

        regime_labels.iloc[prev_t:t] = reg
        return_preds.iloc[prev_t:t] = ret
        prev_t = t

    regime_labels = regime_labels.dropna().astype(int)
    return_preds  = return_preds.dropna(how="all")
    return regime_labels, return_preds, pipeline
