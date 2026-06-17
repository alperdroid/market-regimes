"""
portfolio/ledoit_wolf.py
------------------------
Ledoit-Wolf linear shrinkage covariance estimator.

Uses sklearn's analytical LedoitWolf estimator which computes the
optimal shrinkage intensity α* in closed form (Oracle Approximating Shrinkage).
The shrinkage target is a scaled identity matrix.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf as _SklearnLW


def ledoit_wolf_shrinkage(
    returns: np.ndarray,
    assume_centered: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Compute the Ledoit-Wolf shrunk covariance matrix.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Return observations.
    assume_centered : bool
        If True, data is not mean-centred before estimation.

    Returns
    -------
    cov_lw : np.ndarray, shape (N, N)
        Shrunk covariance matrix (positive definite).
    alpha  : float
        Optimal shrinkage intensity ∈ [0, 1].
    """
    lw = _SklearnLW(assume_centered=assume_centered)
    lw.fit(returns)
    return lw.covariance_, lw.shrinkage_


def robust_covariance(
    returns: np.ndarray,
    use_shrinkage: bool = True,
    min_eigenvalue: float = 1e-8,
) -> np.ndarray:
    """
    Return a covariance matrix that is guaranteed to be positive definite.

    Steps:
    1. Estimate with Ledoit-Wolf (or sample covariance).
    2. Apply eigenvalue flooring to ensure positive definiteness.
    3. Return the matrix.
    """
    if len(returns) < 2:
        n = returns.shape[1] if returns.ndim == 2 else 1
        return np.eye(n) * 1e-4

    if use_shrinkage:
        cov, _ = ledoit_wolf_shrinkage(returns)
    else:
        cov = np.cov(returns.T)

    # Eigenvalue flooring for numerical stability
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    cov_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov_pd


def regime_conditional_moments(
    returns_df: pd.DataFrame,
    regime_labels: pd.Series,
    use_shrinkage: bool = True,
    min_obs: int = 30,
) -> dict:
    """
    Compute regime-conditional (mean, covariance) pairs.

    Returns
    -------
    dict mapping regime_code → {"mu": np.ndarray, "sigma": np.ndarray}
    """
    results = {}
    aligned = returns_df.loc[returns_df.index.isin(regime_labels.index)]
    reg_aligned = regime_labels.reindex(aligned.index).dropna()

    for code in [0, 1, 2]:
        mask = reg_aligned == code
        sub = aligned.loc[mask]
        if len(sub) < min_obs:
            # Fallback to overall moments
            mu    = aligned.mean().values
            sigma = robust_covariance(aligned.values, use_shrinkage)
        else:
            mu    = sub.mean().values
            sigma = robust_covariance(sub.values, use_shrinkage)
        results[code] = {"mu": mu, "sigma": sigma}
    return results


def rolling_unconditional_moments(
    returns_df: pd.DataFrame,
    window: int = 252,
    use_shrinkage: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Compute rolling unconditional mean and covariance for static strategies.
    Returns (rolling_means_df, list_of_cov_matrices).
    """
    n = len(returns_df)
    mu_list    = []
    sigma_list = []

    for t in range(window, n + 1):
        sub = returns_df.iloc[t - window:t].values
        mu    = sub.mean(axis=0)
        sigma = robust_covariance(sub, use_shrinkage)
        mu_list.append(mu)
        sigma_list.append(sigma)

    mu_df = pd.DataFrame(
        mu_list,
        index=returns_df.index[window:],
        columns=returns_df.columns,
    )
    return mu_df, sigma_list
