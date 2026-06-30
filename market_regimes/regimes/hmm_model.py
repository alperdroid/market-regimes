"""
regimes/hmm_model.py
--------------------
3-state Gaussian HMM for latent regime detection.

Uses hmmlearn.hmm.GaussianHMM with full covariance matrices.
- Training: Baum-Welch EM algorithm with multiple random restarts.
- Decoding: Viterbi algorithm for most probable state sequence.
- Model selection: Best log-likelihood across random starts.
- State relabeling: states are aligned so 0=Calm, 1=Transitional, 2=Crisis
  based on ascending state-conditional VIX mean.
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def _sort_states_by_vix_mean(model: GaussianHMM, dvix_col_idx: int = 0) -> np.ndarray:
    """
    Return a permutation array that reorders HMM states so that
    state 0 has the lowest ΔVIX mean (Calm) and state 2 the highest (Crisis).
    """
    means = model.means_[:, dvix_col_idx]
    order = np.argsort(means)  # ascending: calm → crisis
    return order


def _remap_states(states: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Remap decoded states using the ordering permutation."""
    inverse = np.argsort(order)
    return inverse[states]


class HMMRegimeModel:
    """
    Wrapper around hmmlearn GaussianHMM with:
      - multi-start random initialisation
      - state sorting by volatility level
      - incremental (walk-forward) refit support
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        n_init: int = 20,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model_ = None
        self._order = None

    def fit(self, X: np.ndarray) -> "HMMRegimeModel":
        """
        Fit HMM with multiple random starts; keep model with highest log-likelihood.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
            Feature matrix. Column 0 should be ΔVIX.
        """
        best_model = None
        best_score = -np.inf

        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 10_000, size=self.n_init)

        for seed in seeds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=int(seed),
                )
                try:
                    model.fit(X)
                    score = model.score(X)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception:
                    continue

        if best_model is None:
            raise RuntimeError("HMM failed to converge in all random restarts.")

        self.model_ = best_model
        self._order = _sort_states_by_vix_mean(best_model, dvix_col_idx=0)
        return self

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding → sorted integer state labels {0, 1, 2}.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        _, raw_states = self.model_.decode(X, algorithm="viterbi")
        return _remap_states(raw_states, self._order)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior state probabilities (forward-backward smoothing).
        Returns shape (T, n_states), columns sorted by volatility.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        posteriors = self.model_.predict_proba(X)
        return posteriors[:, self._order]

    @property
    def transition_matrix(self) -> np.ndarray:
        """Transition probability matrix A, shape (n_states, n_states), sorted."""
        A = self.model_.transmat_
        # Reorder rows and columns
        A_sorted = A[self._order][:, self._order]
        return A_sorted

    @property
    def state_means(self) -> np.ndarray:
        """State-conditional means, sorted by volatility. Shape (n_states, n_features)."""
        return self.model_.means_[self._order]

    @property
    def state_covars(self) -> np.ndarray:
        """State-conditional covariance matrices, sorted. Shape (n_states, n_features, n_features)."""
        return self.model_.covars_[self._order]

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion for model selection."""
        n_params = (
            self.n_states ** 2 - self.n_states          # transition matrix
            + self.n_states * X.shape[1]                # means
            + self.n_states * X.shape[1] ** 2           # full covariances
        )
        log_lik = self.model_.score(X) * len(X)
        return -2 * log_lik + n_params * np.log(len(X))


def fit_hmm_walkforward(
    feature_df: pd.DataFrame,
    min_train_days: int = 504,
    refit_freq: int = 252,
    hmm_kwargs: dict = None,
) -> pd.Series:
    """
    Walk-forward expanding-window HMM regime labeling.

    At each refit point t, the HMM is trained on data up to t (indices 0..t-1) and used to
    Viterbi-decode the *following* window [t : next_t]. The model never sees the days it
    labels, so every regime label is strictly out-of-sample.

    Returns
    -------
    pd.Series[int]
        Regime labels for the OOS period (index = dates from min_train_days onward).
    """
    if hmm_kwargs is None:
        hmm_kwargs = {}

    X_all = feature_df.values.astype(float)
    dates = feature_df.index
    T = len(X_all)

    labels = pd.Series(index=dates, dtype=float)

    refit_points = list(range(min_train_days, T, refit_freq))
    model = None

    for i, t in enumerate(refit_points):
        new_model = HMMRegimeModel(**hmm_kwargs)
        try:
            new_model.fit(X_all[:t])       # trained on days 0..t-1 only
            model = new_model
        except Exception as e:
            print(f"  [HMM] Fit failed at t={t}: {e}. Using previous model.")
        if model is None or model.model_ is None:
            continue

        # Viterbi-decode the FOLLOWING window [t : next_t] — strictly out-of-sample
        next_t = refit_points[i + 1] if i + 1 < len(refit_points) else T
        if t < next_t:
            labels.iloc[t:next_t] = model.decode(X_all[t:next_t])

    labels = labels.dropna().astype(int)
    labels.name = "HMM_Regime"
    return labels, model   # return last trained model too
