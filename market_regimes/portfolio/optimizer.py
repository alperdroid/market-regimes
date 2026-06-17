"""
portfolio/optimizer.py
----------------------
Long-only portfolio optimizers using scipy SLSQP:

  1. Minimum Variance Portfolio (MVP):
       min  w' Σ w
       s.t. Σ w_i = 1
            w_i ≥ 0

  2. Tangency Portfolio (TPF) — Maximum Sharpe Ratio:
       max  (μ' w - rf) / sqrt(w' Σ w)
       s.t. Σ w_i = 1
            w_i ≥ 0

Both optimizers are implemented with numerical safeguards and fallback
to the 1/N equal-weight portfolio if optimisation fails.
"""

import numpy as np
from scipy.optimize import minimize


def _equal_weight(n: int) -> np.ndarray:
    """Return equal-weight portfolio of size n."""
    return np.ones(n) / n


def _portfolio_variance(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(w @ sigma @ w)


def _neg_sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    port_ret = float(mu @ w)
    port_std = float(np.sqrt(max(_portfolio_variance(w, sigma), 1e-12)))
    return -(port_ret - rf) / port_std


def minimum_variance_portfolio(
    sigma: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Long-only Minimum Variance Portfolio.

    Parameters
    ----------
    sigma     : (N, N) covariance matrix
    min_weight: lower bound per asset (0 = long-only)
    max_weight: upper bound per asset (1 = no leverage per asset)

    Returns
    -------
    w : (N,) optimal weights
    """
    n = sigma.shape[0]
    w0 = _equal_weight(n)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n

    result = minimize(
        fun=_portfolio_variance,
        x0=w0,
        args=(sigma,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if result.success:
        w = np.clip(result.x, min_weight, max_weight)
        w = w / w.sum()
        return w
    else:
        return w0   # fallback: equal weight


def tangency_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Long-only Tangency (Maximum Sharpe Ratio) Portfolio.

    Parameters
    ----------
    mu        : (N,) expected returns vector (daily, excess)
    sigma     : (N, N) covariance matrix
    rf        : daily risk-free rate (scalar)
    min_weight: lower bound per asset
    max_weight: upper bound per asset

    Returns
    -------
    w : (N,) optimal weights
    """
    n = len(mu)
    w0 = _equal_weight(n)

    # Ensure some positive expected excess return to avoid degenerate case
    mu_excess = mu - rf
    if mu_excess.max() <= 0:
        # All assets have negative excess returns — fall back to MVP
        return minimum_variance_portfolio(sigma, min_weight, max_weight)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n

    result = minimize(
        fun=_neg_sharpe,
        x0=w0,
        args=(mu, sigma, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if result.success:
        w = np.clip(result.x, min_weight, max_weight)
        w = w / w.sum()
        return w
    else:
        return minimum_variance_portfolio(sigma, min_weight, max_weight)


def apply_crisis_cash(
    weights: np.ndarray,
    cash_fraction: float = 0.15,
) -> np.ndarray:
    """
    Scale risky asset weights by (1 - cash_fraction) during crisis regime.
    The remaining cash_fraction earns the risk-free rate.
    The returned vector has length N (risky assets only); cash tracked separately.
    """
    return weights * (1.0 - cash_fraction)


def compute_portfolio_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    strategy: str,
    rf_daily: float = 0.0,
    regime: int = None,
    crisis_cash_fraction: float = 0.15,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Unified weight computation for a given strategy string.

    strategy : "mvp" | "tpf" | "ew"
    """
    n = len(mu)

    if strategy == "ew":
        w = _equal_weight(n)
    elif strategy == "mvp":
        w = minimum_variance_portfolio(sigma, min_weight, max_weight)
    elif strategy == "tpf":
        w = tangency_portfolio(mu, sigma, rf_daily, min_weight, max_weight)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    # Apply cash buffer in crisis
    if regime == 2 and crisis_cash_fraction > 0:
        w = apply_crisis_cash(w, crisis_cash_fraction)

    return w
