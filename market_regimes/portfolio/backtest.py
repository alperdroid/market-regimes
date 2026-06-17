"""
portfolio/backtest.py
---------------------
Walk-forward backtest engine for all 10 strategies.

Strategies:
  0. SPY B&H         — passive buy-and-hold SPY
  1. EW 1/N          — equal-weight, monthly rebalance
  2. Static MVP      — unconditional rolling MVP (no regime)
  3. Static TPF      — unconditional rolling TPF (no regime)
  4. VIX-MVP         — VIX-gated regime-conditional MVP
  5. VIX-TPF         — VIX-gated regime-conditional TPF
  6. HMM-MVP         — HMM-gated regime-conditional MVP
  7. HMM-TPF         — HMM-gated regime-conditional TPF
  8. ML-MVP          — GMM+RF-gated regime-conditional MVP
  9. ML-TPF          — GMM+RF-gated regime-conditional TPF

Rebalancing: monthly (~21 trading days).
Transaction costs: applied at each rebalance proportional to weight turnover.
"""

import numpy as np
import pandas as pd

from portfolio.ledoit_wolf import (
    robust_covariance,
    regime_conditional_moments,
    rolling_unconditional_moments,
)
from portfolio.optimizer import compute_portfolio_weights


STRATEGY_NAMES = [
    "SPY B&H",
    "EW 1/N",
    "Static MVP",
    "Static TPF",
    "VIX-MVP",
    "VIX-TPF",
    "HMM-MVP",
    "HMM-TPF",
    "ML-MVP",
    "ML-TPF",
]


def _compute_turnover(w_old: np.ndarray, w_new: np.ndarray) -> float:
    """One-way portfolio turnover (sum of absolute weight changes / 2)."""
    return float(np.abs(w_new - w_old).sum() / 2.0)


def _apply_transaction_costs(
    ret: float,
    turnover: float,
    cost_bps: float,
) -> float:
    """Subtract one-way transaction costs from portfolio return."""
    cost = turnover * cost_bps / 10_000.0
    return ret - cost


def _rebalance_schedule(index: pd.Index, rebalance_freq: int) -> list[int]:
    """
    Generate list of integer positions where rebalancing occurs
    (every `rebalance_freq` trading days from start).
    """
    T = len(index)
    return list(range(0, T, rebalance_freq))


def run_backtest(
    excess_returns:  pd.DataFrame,     # (T, N) daily excess returns for ETFs
    spy_excess:      pd.Series,        # (T,) daily excess returns for SPY
    rf_daily:        pd.Series,        # (T,) daily risk-free rate
    vix_regimes:     pd.Series,        # (T,) VIX regime labels {0,1,2}
    hmm_regimes:     pd.Series,        # (T,) HMM regime labels  {0,1,2}
    ml_regimes:      pd.Series,        # (T,) ML  regime labels  {0,1,2}
    ml_return_preds: pd.DataFrame,     # (T, N) ML-forecasted returns
    rolling_window:  int   = 252,
    rebalance_freq:  int   = 21,
    cost_bps:        float = 10.0,
    crisis_cash:     float = 0.15,
    use_shrinkage:   bool  = True,
) -> pd.DataFrame:
    """
    Run all 10 strategies on aligned data.

    Returns
    -------
    pd.DataFrame, shape (T, 10)
        Daily gross portfolio returns for each strategy.
        Index = dates. Columns = STRATEGY_NAMES.
    """
    # ── Align all inputs on common trading dates ─────────────────────────────
    common_idx = (
        excess_returns.index
        .intersection(spy_excess.index)
        .intersection(vix_regimes.index)
        .intersection(hmm_regimes.index)
        .intersection(ml_regimes.index)
        .sort_values()
    )

    er   = excess_returns.loc[common_idx]
    spy  = spy_excess.loc[common_idx]
    rf   = rf_daily.reindex(common_idx).ffill().bfill()
    vr   = vix_regimes.reindex(common_idx).ffill().bfill().astype(int)
    hr   = hmm_regimes.reindex(common_idx).ffill().bfill().astype(int)
    mr   = ml_regimes.reindex(common_idx).ffill().bfill().astype(int)
    mlp  = ml_return_preds.reindex(common_idx).ffill().bfill()

    T, N = er.shape
    dates = common_idx

    # ── Pre-compute regime-conditional moments on full in-sample history ──────
    # (These are recomputed inside the walk-forward loop below)

    # ── Initialise weight arrays ─────────────────────────────────────────────
    w_static_mvp  = np.ones(N) / N
    w_static_tpf  = np.ones(N) / N
    w_vix_mvp     = np.ones(N) / N
    w_vix_tpf     = np.ones(N) / N
    w_hmm_mvp     = np.ones(N) / N
    w_hmm_tpf     = np.ones(N) / N
    w_ml_mvp      = np.ones(N) / N
    w_ml_tpf      = np.ones(N) / N
    w_ew          = np.ones(N) / N

    # Daily portfolio returns (excess), net of transaction costs
    port_rets = np.zeros((T, len(STRATEGY_NAMES)))
    # Parallel gross (pre-cost) returns, for break-even cost analysis
    gross_rets = np.zeros((T, len(STRATEGY_NAMES)))

    # Cumulative one-way turnover per strategy (for cost accounting + reporting)
    turnover_accum = np.zeros(len(STRATEGY_NAMES))

    rebal_days = set(_rebalance_schedule(dates, rebalance_freq))

    # Pre-convert to numpy — avoids any pandas boolean index issues
    er_np_ = er.to_numpy()
    vr_np_ = vr.to_numpy()
    hr_np_ = hr.to_numpy()
    mr_np_ = mr.to_numpy()

    def _regime_moments(regime_arr: np.ndarray, t: int) -> dict:
        moments = {}
        for code in [0, 1, 2]:
            pos = np.where(regime_arr[:t] == code)[0]
            sub = er_np_[pos] if len(pos) >= 30 else er_np_[:t]
            moments[code] = {
                "mu": sub.mean(axis=0),
                "sigma": robust_covariance(sub, use_shrinkage),
            }
        return moments

    for t in range(rolling_window, T):
        # Historical window for moment estimation
        hist = er_np_[max(0, t - rolling_window):t]
        rf_t       = float(rf.iloc[t])
        regime_vix = int(vr_np_[t])
        regime_hmm = int(hr_np_[t])
        regime_ml  = int(mr_np_[t])
        today_ret  = er_np_[t]
        spy_ret    = float(spy.iloc[t])

        # One-way turnover incurred *today* per strategy (0 except on rebalances)
        day_turn = np.zeros(len(STRATEGY_NAMES))

        # ── Rebalance on schedule ─────────────────────────────────────────
        if t in rebal_days:

            # --- Unconditional moments (rolling) ---
            sigma_unc = robust_covariance(hist, use_shrinkage)
            mu_unc    = hist.mean(axis=0)

            # --- Static MVP ---
            w_new = compute_portfolio_weights(mu_unc, sigma_unc, "mvp",
                                             rf_t, None, 0.0)
            day_turn[2] = _compute_turnover(w_static_mvp, w_new)
            w_static_mvp = w_new

            # --- Static TPF ---
            w_new = compute_portfolio_weights(mu_unc, sigma_unc, "tpf",
                                             rf_t, None, 0.0)
            day_turn[3] = _compute_turnover(w_static_tpf, w_new)
            w_static_tpf = w_new

            # --- Regime-conditional moments (VIX / HMM / ML) ---
            vix_mom = _regime_moments(vr_np_, t)
            hmm_mom = _regime_moments(hr_np_, t)
            ml_mom  = _regime_moments(mr_np_, t)

            # --- VIX-gated MVP ---
            mu_v, sg_v = vix_mom[regime_vix]["mu"], vix_mom[regime_vix]["sigma"]
            w_new = compute_portfolio_weights(mu_v, sg_v, "mvp", rf_t,
                                              regime_vix, crisis_cash)
            day_turn[4] = _compute_turnover(w_vix_mvp, w_new)
            w_vix_mvp = w_new
            # --- VIX-gated TPF ---
            w_new = compute_portfolio_weights(mu_v, sg_v, "tpf", rf_t,
                                              regime_vix, crisis_cash)
            day_turn[5] = _compute_turnover(w_vix_tpf, w_new)
            w_vix_tpf = w_new
            # --- HMM-gated MVP ---
            mu_h, sg_h = hmm_mom[regime_hmm]["mu"], hmm_mom[regime_hmm]["sigma"]
            w_new = compute_portfolio_weights(mu_h, sg_h, "mvp", rf_t,
                                              regime_hmm, crisis_cash)
            day_turn[6] = _compute_turnover(w_hmm_mvp, w_new)
            w_hmm_mvp = w_new
            # --- HMM-gated TPF ---
            w_new = compute_portfolio_weights(mu_h, sg_h, "tpf", rf_t,
                                              regime_hmm, crisis_cash)
            day_turn[7] = _compute_turnover(w_hmm_tpf, w_new)
            w_hmm_tpf = w_new

            # --- ML-gated MVP ---
            mu_ml = mlp.iloc[t].values if not mlp.iloc[t].isna().all() else mu_unc
            mu_ml_e = mu_ml - rf_t
            sg_ml  = ml_mom[regime_ml]["sigma"]
            w_new = compute_portfolio_weights(mu_ml_e, sg_ml, "mvp", rf_t,
                                              regime_ml, crisis_cash)
            day_turn[8] = _compute_turnover(w_ml_mvp, w_new)
            w_ml_mvp = w_new
            # --- ML-gated TPF ---
            w_new = compute_portfolio_weights(mu_ml_e, sg_ml, "tpf", rf_t,
                                              regime_ml, crisis_cash)
            day_turn[9] = _compute_turnover(w_ml_tpf, w_new)
            w_ml_tpf = w_new

            turnover_accum += day_turn

        # ── Daily gross portfolio returns (excess) ────────────────────────────
        gross = np.array([
            spy_ret,                          # SPY B&H (excess)
            float(w_ew @ today_ret),          # EW 1/N
            float(w_static_mvp @ today_ret),  # Static MVP
            float(w_static_tpf @ today_ret),  # Static TPF
            float(w_vix_mvp @ today_ret),     # VIX-MVP
            float(w_vix_tpf @ today_ret),     # VIX-TPF
            float(w_hmm_mvp @ today_ret),     # HMM-MVP
            float(w_hmm_tpf @ today_ret),     # HMM-TPF
            float(w_ml_mvp @ today_ret),      # ML-MVP
            float(w_ml_tpf @ today_ret),      # ML-TPF
        ])

        # ── Subtract one-way transaction costs charged on today's turnover ────
        gross_rets[t] = gross
        port_rets[t]  = gross - day_turn * cost_bps / 10_000.0

    result = pd.DataFrame(
        port_rets[rolling_window:],
        index=dates[rolling_window:],
        columns=STRATEGY_NAMES,
    )

    # Measured annualised one-way turnover per strategy (for break-even / reporting)
    n_years = max((T - rolling_window) / 252.0, 1e-9)
    result.attrs["annual_turnover"] = {
        name: float(turnover_accum[i] / n_years)
        for i, name in enumerate(STRATEGY_NAMES)
    }
    # Gross (pre-cost) returns for break-even analysis
    result.attrs["gross_returns"] = pd.DataFrame(
        gross_rets[rolling_window:],
        index=dates[rolling_window:],
        columns=STRATEGY_NAMES,
    )

    return result


def cumulative_wealth(
    daily_excess_returns: pd.DataFrame,
    rf_daily: pd.Series,
    initial_wealth: float = 1.0,
) -> pd.DataFrame:
    """
    Convert daily excess returns to cumulative wealth.
    Total return = excess return + risk-free rate.
    """
    rf_aligned = rf_daily.reindex(daily_excess_returns.index).ffill().bfill()
    total_returns = daily_excess_returns.add(rf_aligned, axis=0)
    wealth = (1.0 + total_returns).cumprod() * initial_wealth
    return wealth


def weight_history(
    excess_returns:  pd.DataFrame,
    spy_excess:      pd.Series,
    rf_daily:        pd.Series,
    vix_regimes:     pd.Series,
    hmm_regimes:     pd.Series,
    ml_regimes:      pd.Series,
    ml_return_preds: pd.DataFrame,
    rolling_window:  int = 252,
    rebalance_freq:  int = 21,
    crisis_cash:     float = 0.15,
    use_shrinkage:   bool  = True,
) -> dict:
    """
    Record portfolio weight history for all regime-gated strategies.
    Returns dict: strategy_name → pd.DataFrame (T, N_assets).
    """
    # Align
    common_idx = (
        excess_returns.index
        .intersection(vix_regimes.index)
        .intersection(hmm_regimes.index)
        .intersection(ml_regimes.index)
        .sort_values()
    )
    er  = excess_returns.loc[common_idx]
    rf  = rf_daily.reindex(common_idx).ffill().bfill()
    vr  = vix_regimes.reindex(common_idx).ffill().bfill().astype(int)
    hr  = hmm_regimes.reindex(common_idx).ffill().bfill().astype(int)
    mr  = ml_regimes.reindex(common_idx).ffill().bfill().astype(int)
    mlp = ml_return_preds.reindex(common_idx).ffill().bfill()
    T, N = er.shape
    cols = er.columns.tolist()
    dates = common_idx

    strategies = ["VIX-MVP", "VIX-TPF", "HMM-MVP", "HMM-TPF", "ML-MVP", "ML-TPF"]
    wh = {s: pd.DataFrame(np.nan, index=dates, columns=cols) for s in strategies}

    w_vix_mvp = np.ones(N)/N; w_vix_tpf = np.ones(N)/N
    w_hmm_mvp = np.ones(N)/N; w_hmm_tpf = np.ones(N)/N
    w_ml_mvp  = np.ones(N)/N; w_ml_tpf  = np.ones(N)/N

    rebal_days = set(_rebalance_schedule(dates, rebalance_freq))

    # Convert to numpy arrays upfront — avoids any pandas boolean index issues
    vr_np = vr.to_numpy()
    hr_np = hr.to_numpy()
    mr_np = mr.to_numpy()
    er_np = er.to_numpy()

    for t in range(rolling_window, T):
        rf_t = float(rf.iloc[t])
        rv, rh, rm = int(vr_np[t]), int(hr_np[t]), int(mr_np[t])

        if t in rebal_days:
            def _mom(regime_arr):
                moments = {}
                for code in [0, 1, 2]:
                    pos = np.where(regime_arr[:t] == code)[0]
                    sub = er_np[pos] if len(pos) >= 30 else er_np[:t]
                    moments[code] = {
                        "mu": sub.mean(axis=0),
                        "sigma": robust_covariance(sub, use_shrinkage)
                    }
                return moments

            vm = _mom(vr_np); hm = _mom(hr_np); mm = _mom(mr_np)

            mu_ml = (mlp.iloc[t].to_numpy() if not mlp.iloc[t].isna().all()
                     else er_np[:t].mean(axis=0))

            w_vix_mvp = compute_portfolio_weights(vm[rv]["mu"], vm[rv]["sigma"], "mvp", rf_t, rv, crisis_cash)
            w_vix_tpf = compute_portfolio_weights(vm[rv]["mu"], vm[rv]["sigma"], "tpf", rf_t, rv, crisis_cash)
            w_hmm_mvp = compute_portfolio_weights(hm[rh]["mu"], hm[rh]["sigma"], "mvp", rf_t, rh, crisis_cash)
            w_hmm_tpf = compute_portfolio_weights(hm[rh]["mu"], hm[rh]["sigma"], "tpf", rf_t, rh, crisis_cash)
            w_ml_mvp  = compute_portfolio_weights(mu_ml - rf_t, mm[rm]["sigma"], "mvp", rf_t, rm, crisis_cash)
            w_ml_tpf  = compute_portfolio_weights(mu_ml - rf_t, mm[rm]["sigma"], "tpf", rf_t, rm, crisis_cash)

        d = dates[t]
        wh["VIX-MVP"].loc[d] = w_vix_mvp
        wh["VIX-TPF"].loc[d] = w_vix_tpf
        wh["HMM-MVP"].loc[d] = w_hmm_mvp
        wh["HMM-TPF"].loc[d] = w_hmm_tpf
        wh["ML-MVP"].loc[d]  = w_ml_mvp
        wh["ML-TPF"].loc[d]  = w_ml_tpf

    return {k: v.dropna(how="all") for k, v in wh.items()}
