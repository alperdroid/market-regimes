"""
performance/metrics.py
-----------------------
Standard performance and risk-adjusted metrics for strategy evaluation.

All annualisation uses 252 trading days.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, skew, kurtosis


TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def annualised_return(daily_excess: pd.Series) -> float:
    """Annualised geometric mean excess return."""
    return float((1 + daily_excess).prod() ** (TRADING_DAYS / len(daily_excess)) - 1)


def annualised_vol(daily_excess: pd.Series) -> float:
    """Annualised volatility."""
    return float(daily_excess.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_excess: pd.Series) -> float:
    """
    Annualised Sharpe ratio (excess return already net of risk-free).
    SR = E[e] / σ[e] × √252
    """
    mu  = daily_excess.mean()
    std = daily_excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(mu / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(daily_excess: pd.Series) -> float:
    """
    Sortino ratio using downside semi-deviation.
    Sortino = E[e] × √252 / σ_downside
    """
    mu = daily_excess.mean()
    downside = daily_excess[daily_excess < 0]
    if len(downside) == 0:
        return np.nan
    # Annualised downside deviation (target return = 0)
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    if downside_dev == 0:
        return np.nan
    # Annualised mean (×252) over annualised downside deviation (×√252),
    # i.e. consistent annualisation with the Sharpe ratio above.
    ann_mean = mu * TRADING_DAYS
    return float(ann_mean / downside_dev)


def maximum_drawdown(daily_excess: pd.Series, rf_daily: pd.Series = None) -> float:
    """
    Maximum peak-to-trough drawdown of cumulative wealth.
    If rf_daily is provided, includes risk-free accrual (total return).
    """
    if rf_daily is not None:
        rf_aligned = rf_daily.reindex(daily_excess.index).ffill().bfill()
        total = daily_excess + rf_aligned
    else:
        total = daily_excess

    cumulative = (1 + total).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return float(drawdowns.min())


def jensens_alpha(
    strategy_excess: pd.Series,
    spy_excess:      pd.Series,
) -> tuple[float, float]:
    """
    Jensen's alpha: intercept of OLS regression of strategy on SPY.
    α = annualised intercept
    β = market beta

    Returns (alpha_annualised, beta, p_value_alpha)
    """
    common = strategy_excess.index.intersection(spy_excess.index)
    y = strategy_excess.loc[common].values
    X = sm.add_constant(spy_excess.loc[common].values)
    try:
        ols = sm.OLS(y, X).fit()
        alpha = ols.params[0] * TRADING_DAYS
        beta  = ols.params[1]
        pval  = ols.pvalues[0]
        return float(alpha), float(beta), float(pval)
    except Exception:
        return np.nan, np.nan, np.nan


def calmar_ratio(daily_excess: pd.Series, rf_daily: pd.Series = None) -> float:
    """Annualised return / |Maximum Drawdown|."""
    ann_ret = annualised_return(daily_excess)
    mdd     = maximum_drawdown(daily_excess, rf_daily)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(ann_ret / abs(mdd))


def breakeven_transaction_cost(
    strategy_excess: pd.Series,
    benchmark_sr:    float,
    annual_turnover: float,
) -> float:
    """
    Maximum one-way cost (in bps) the strategy can absorb before its
    Sharpe ratio falls to the benchmark Sharpe ratio.

    breakeven_bps = (SR_strategy - SR_benchmark) / (√252 * annual_turnover / 10000)

    Parameters
    ----------
    strategy_excess : daily excess return series
    benchmark_sr    : Sharpe ratio of benchmark (e.g. SPY)
    annual_turnover : average annual one-way portfolio turnover (fraction)
    """
    sr_strat = sharpe_ratio(strategy_excess)
    if np.isnan(sr_strat) or annual_turnover <= 0:
        return np.nan
    sr_diff = sr_strat - benchmark_sr
    # Cost degrades mean return: Δμ_daily = cost_bps/10000 × daily_turnover
    # ΔSR = Δμ_daily / σ × √252 → solve for cost_bps
    sigma_daily = float(strategy_excess.std())
    if sigma_daily == 0:
        return np.nan
    # SR reduction per unit cost (in bps):
    # ΔSR = (cost_bps/10000) × turnover_daily × √252 / σ_daily
    # cost_bps = ΔSR × 10000 × σ_daily / (turnover_daily × √252)
    daily_turnover = annual_turnover / TRADING_DAYS
    be_bps = sr_diff * 10_000 * sigma_daily / (daily_turnover * np.sqrt(TRADING_DAYS))
    return float(max(be_bps, 0.0))


def full_performance_table(
    port_returns:    pd.DataFrame,   # (T, N_strategies) daily excess returns (net of cost)
    spy_excess:      pd.Series,
    rf_daily:        pd.Series,
    turnover_dict:   dict = None,    # strategy → annual one-way turnover
    gross_returns:   pd.DataFrame = None,   # (T, N) gross returns for break-even
) -> pd.DataFrame:
    """
    Build the complete performance comparison table.

    All performance columns (Sharpe, alpha, …) are computed on the NET (after-cost)
    return series. The break-even column answers "what TOTAL one-way cost makes this
    strategy's Sharpe equal SPY's?" and is therefore computed on the GROSS series when
    one is supplied (else it falls back to the net series, in which case it reads as
    cost *beyond the baseline already deducted*).

    Returns
    -------
    pd.DataFrame
        Rows: strategies
        Columns: Ann.Return, Ann.Vol, Sharpe, Sortino, MDD, Alpha, Beta,
                 p_Alpha, Calmar, Breakeven_bps
    """
    spy_sr = sharpe_ratio(spy_excess.reindex(port_returns.index).ffill())

    rows = []
    for col in port_returns.columns:
        s  = port_returns[col].dropna()
        sp = spy_excess.reindex(s.index).ffill().bfill()
        rf = rf_daily.reindex(s.index).ffill().bfill()

        alpha, beta, p_alpha = jensens_alpha(s, sp)
        to = (turnover_dict or {}).get(col, 0.5)   # default 50% annual turnover

        # Break-even uses gross returns so the bps figure is a TOTAL cost, not
        # an increment on top of an already-deducted baseline.
        s_be = (gross_returns[col].dropna()
                if gross_returns is not None and col in gross_returns.columns
                else s)

        row = {
            "Strategy":      col,
            "Ann.Return (%)": annualised_return(s) * 100,
            "Ann.Vol (%)":    annualised_vol(s) * 100,
            "Sharpe":         sharpe_ratio(s),
            "Sortino":        sortino_ratio(s),
            "Max DD (%)":     maximum_drawdown(s, rf) * 100,
            "Jensen α (%)":   alpha * 100,
            "Beta":           beta,
            "p(α)":           p_alpha,
            "Calmar":         calmar_ratio(s, rf),
            "Breakeven (bps)": breakeven_transaction_cost(s_be, spy_sr, to),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")
    return df.round(4)


# ─────────────────────────────────────────────────────────────────────────────
#  MULTIPLE-TESTING / DATA-SNOOPING CORRECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def probabilistic_sharpe_ratio(
    daily_excess: pd.Series,
    sr_benchmark_daily: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    P(true SR > benchmark SR) given the observed sample, correcting for sample
    length and the non-normality (skew, kurtosis) of returns. All Sharpe ratios
    are in per-observation (daily) units.
    """
    r = np.asarray(daily_excess.dropna(), dtype=float)
    T = len(r)
    sd = r.std(ddof=1)
    if T < 10 or sd == 0 or np.isnan(sd):
        return np.nan
    sr = r.mean() / sd                       # daily (non-annualised) Sharpe
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))   # non-excess kurtosis
    denom = np.sqrt(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2)
    if denom == 0 or np.isnan(denom):
        return np.nan
    z = (sr - sr_benchmark_daily) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    candidate_returns: pd.DataFrame,
    selected: str = None,
) -> dict:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Accounts for the fact that the best strategy was selected out of N trials.
    The null benchmark SR0 is the *expected maximum* Sharpe achievable by chance
    given N trials whose Sharpes have cross-sectional variance V:

        SR0 = sqrt(V) · [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]

    DSR = PSR evaluated at SR* = SR0 for the selected (best) strategy.

    Parameters
    ----------
    candidate_returns : DataFrame of daily excess returns — the *trials* searched
                        over (exclude the passive benchmark).
    selected          : strategy to deflate; defaults to the highest-Sharpe trial.
    """
    srs = {}
    for c in candidate_returns.columns:
        r = candidate_returns[c].dropna().values
        sd = r.std(ddof=1) if len(r) > 2 else 0.0
        if len(r) > 2 and sd > 0:
            srs[c] = r.mean() / sd           # daily Sharpe
    if len(srs) < 2:
        return {}

    sr_vals = np.array(list(srs.values()))
    N = len(sr_vals)
    V = sr_vals.var(ddof=1)                   # variance of trial Sharpes (daily²)

    z1 = norm.ppf(1.0 - 1.0 / N)
    z2 = norm.ppf(1.0 - 1.0 / (N * np.e))
    sr0 = np.sqrt(V) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    if selected is None:
        selected = max(srs, key=srs.get)
    dsr = probabilistic_sharpe_ratio(candidate_returns[selected],
                                     sr_benchmark_daily=sr0)
    return {
        "selected":        selected,
        "n_trials":        N,
        "sr0_annual":      float(sr0 * np.sqrt(TRADING_DAYS)),
        "best_sr_annual":  float(srs[selected] * np.sqrt(TRADING_DAYS)),
        "dsr":             dsr,
    }


def _stationary_bootstrap_idx(T: int, avg_block: float, rng) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index sequence."""
    idx = np.empty(T, dtype=np.int64)
    q = 1.0 / avg_block
    i = int(rng.integers(0, T))
    for t in range(T):
        if t == 0 or rng.random() < q:
            i = int(rng.integers(0, T))
        else:
            i = (i + 1) % T
        idx[t] = i
    return idx


def whites_reality_check(
    port_returns: pd.DataFrame,
    spy_excess:   pd.Series,
    active:       list = None,
    n_boot:       int = 2000,
    avg_block:    float = 10.0,
    seed:         int = 42,
) -> dict:
    """
    White's Reality Check (2000) via the stationary bootstrap.

    Tests H0: the BEST strategy does not outperform the SPY benchmark, accounting
    for data-snooping across all `active` strategies. The relative-performance
    statistic is the daily return differential d_{k,t} = r_{k,t} − r_{spy,t}.

        V      = max_k √T · mean(d_k)
        V*_b   = max_k √T · (mean(d_k*) − mean(d_k))      (bootstrap, centred)
        p      = (1/B) Σ_b 1{ V*_b ≥ V }
    """
    if active is None:
        active = [c for c in port_returns.columns if c != "SPY B&H"]

    common = port_returns.index
    spy = spy_excess.reindex(common).ffill().bfill().to_numpy()
    D = np.column_stack([
        port_returns[c].reindex(common).to_numpy() - spy for c in active
    ])
    mask = ~np.isnan(D).any(axis=1)
    D = D[mask]
    T, K = D.shape
    if T < 30:
        return {}

    fbar = D.mean(axis=0)
    V = np.sqrt(T) * fbar.max()

    rng = np.random.default_rng(seed)
    Vstar = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_idx(T, avg_block, rng)
        fbar_b = D[idx].mean(axis=0)
        Vstar[b] = np.sqrt(T) * (fbar_b - fbar).max()

    p_value = float((Vstar >= V).mean())
    return {
        "best":      active[int(np.argmax(fbar))],
        "V":         float(V),
        "p_value":   p_value,
        "n_boot":    n_boot,
        "avg_block": avg_block,
        "n_strategies": K,
    }
