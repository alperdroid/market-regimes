"""
capm/beta_analysis.py
---------------------
Regime-conditional CAPM beta estimation.

Model (per regime s):
    e_{i,t} = α_i^(s) + β_i^(s) * e_{SPY,t} + ε_{i,t}

Estimated via OLS for each ETF × regime combination.
Also computes aggregate statistics: R², tracking error, Information Ratio.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


REGIME_NAMES = {0: "Calm", 1: "Transitional", 2: "Crisis"}


def estimate_capm_betas(
    excess_returns: pd.DataFrame,   # (T, N) ETF excess returns
    spy_excess:     pd.Series,      # (T,) SPY excess returns
    regime_labels:  pd.Series,      # (T,) integer regime labels {0, 1, 2}
) -> pd.DataFrame:
    """
    Estimate regime-conditional OLS CAPM betas.

    Returns
    -------
    pd.DataFrame
        Index: ETF tickers
        Columns: MultiIndex (Regime × {alpha, beta, r2, t_beta, p_beta})
    """
    # Align on common dates
    common = (
        excess_returns.index
        .intersection(spy_excess.index)
        .intersection(regime_labels.index)
        .sort_values()
    )
    er   = excess_returns.loc[common]
    spy  = spy_excess.loc[common]
    regs = regime_labels.reindex(common).ffill().bfill().astype(int)

    results = {}

    for code in [0, 1, 2]:
        mask = regs == code
        n_obs = mask.sum()
        regime_results = {}

        spy_sub = spy.loc[mask].values
        X = sm.add_constant(spy_sub)

        for ticker in er.columns:
            y = er.loc[mask, ticker].values
            if n_obs < 10 or len(y) < 10:
                regime_results[ticker] = {
                    "alpha": np.nan, "beta": np.nan,
                    "r2": np.nan, "t_beta": np.nan, "p_beta": np.nan,
                    "n_obs": n_obs,
                }
                continue
            try:
                ols = sm.OLS(y, X).fit()
                regime_results[ticker] = {
                    "alpha":  ols.params[0] * 252,        # annualised
                    "beta":   ols.params[1],
                    "r2":     ols.rsquared,
                    "t_beta": ols.tvalues[1],
                    "p_beta": ols.pvalues[1],
                    "n_obs":  n_obs,
                }
            except Exception:
                regime_results[ticker] = {
                    "alpha": np.nan, "beta": np.nan,
                    "r2": np.nan, "t_beta": np.nan, "p_beta": np.nan,
                    "n_obs": n_obs,
                }

        results[REGIME_NAMES[code]] = pd.DataFrame(regime_results).T

    # Combine into MultiIndex column DataFrame
    combined = pd.concat(results, axis=1)
    return combined


def beta_summary_table(beta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract just alpha and beta for each regime into a clean wide table.

    Returns
    -------
    pd.DataFrame
        Rows: ETF tickers
        Columns: (Regime, 'alpha'), (Regime, 'beta') etc.
    """
    rows = []
    for ticker in beta_df.index:
        row = {"Ticker": ticker}
        for regime in REGIME_NAMES.values():
            try:
                row[f"{regime}_alpha"] = beta_df.loc[ticker, (regime, "alpha")]
                row[f"{regime}_beta"]  = beta_df.loc[ticker, (regime, "beta")]
                row[f"{regime}_r2"]    = beta_df.loc[ticker, (regime, "r2")]
            except KeyError:
                row[f"{regime}_alpha"] = np.nan
                row[f"{regime}_beta"]  = np.nan
                row[f"{regime}_r2"]    = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")
