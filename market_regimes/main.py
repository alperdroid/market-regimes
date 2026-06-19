"""
main.py
-------
Master orchestration script for:
  "Comparative Taxonomy of Market Regimes:
   Machine Learning versus Hidden Markov Models in US Tactical Sector Allocation"

Run with:
    python main.py

Outputs (in results/):
  - 11 publication-quality PNG charts
  - performance_summary.csv
  - regime_stats.csv
  - beta_summary.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import config as CFG

from data.loader  import load_all_data
from data.features import (
    compute_log_returns, compute_excess_returns,
    build_hmm_features, build_gmm_features, build_rf_features,
)
from regimes.vix_classifier import classify_vix_regimes, regime_statistics
from regimes.hmm_model      import HMMRegimeModel, fit_hmm_walkforward
from regimes.ml_pipeline    import fit_ml_walkforward
from regimes.ensemble       import majority_vote_regimes

from portfolio.ledoit_wolf import robust_covariance
from portfolio.backtest    import (
    run_backtest, cumulative_wealth, weight_history, STRATEGY_NAMES
)
from capm.beta_analysis    import estimate_capm_betas, beta_summary_table
from performance.metrics   import (
    sharpe_ratio, full_performance_table,
    deflated_sharpe_ratio, whites_reality_check,
)
from visualization.plots   import generate_all_figures


def banner(text: str):
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def main():
    t0 = time.time()
    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # 1.  DATA LOADING
    # ═══════════════════════════════════════════════════════════════
    banner("Step 1/7 — Loading Market Data")
    data_raw = load_all_data(
        start=CFG.START_DATE,
        end=CFG.END_DATE,
        sector_etfs=CFG.SECTOR_ETFS,
        benchmark=CFG.BENCHMARK,
        cache_path=CFG.DATA_CACHE,
        force_reload=False,
    )

    prices = data_raw["prices"]
    vix    = data_raw["vix"]
    rf     = data_raw["rf"]
    ted    = data_raw["ted"]
    term   = data_raw["term"]

    # ═══════════════════════════════════════════════════════════════
    # 2.  FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════
    banner("Step 2/7 — Feature Engineering")
    log_ret_all = compute_log_returns(prices)          # all tickers + SPY

    sector_cols = [c for c in CFG.SECTOR_ETFS if c in log_ret_all.columns]
    spy_col     = CFG.BENCHMARK

    log_ret_sectors = log_ret_all[sector_cols]
    log_ret_spy     = log_ret_all[spy_col]

    excess_sectors  = compute_excess_returns(log_ret_sectors, rf)
    excess_spy      = compute_excess_returns(log_ret_spy.to_frame(), rf).squeeze()

    print(f"  Sector ETF log-returns: {log_ret_sectors.shape}")
    print(f"  Excess returns range: {excess_sectors.index[0].date()} → {excess_sectors.index[-1].date()}")

    # Macro feature matrix for ML
    rf_features = build_rf_features(vix, ted, term, log_ret_sectors,
                                    lags=CFG.RF_LAG_DAYS)
    hmm_feats, hmm_idx = build_hmm_features(log_ret_sectors, vix)
    gmm_feats, gmm_idx = build_gmm_features(log_ret_sectors, vix)

    rf_feature_names = rf_features.columns.tolist()

    # ═══════════════════════════════════════════════════════════════
    # 3.  REGIME LABELING
    # ═══════════════════════════════════════════════════════════════
    banner("Step 3/7 — Regime Identification")

    # 3a. VIX rule-based
    vix_regimes = classify_vix_regimes(
        vix,
        calm_threshold=CFG.VIX_CALM_THRESHOLD,
        transitional_threshold=CFG.VIX_TRANSITIONAL_THRESHOLD,
    )
    reg_stats = regime_statistics(vix, vix_regimes, log_ret_sectors)
    print("\nVIX Regime Statistics:")
    print(reg_stats.to_string())
    reg_stats.to_csv(os.path.join(CFG.RESULTS_DIR, "regime_stats.csv"))

    # 3b. HMM walk-forward
    print("\nFitting HMM (walk-forward, expanding window) …")
    hmm_kwargs = dict(
        n_states=CFG.HMM_N_STATES,
        n_iter=CFG.HMM_N_ITER,
        n_init=CFG.HMM_N_INIT,
        covariance_type=CFG.HMM_COVARIANCE_TYPE,
        random_state=CFG.HMM_RANDOM_STATE,
    )
    hmm_regimes, hmm_last_model = fit_hmm_walkforward(
        feature_df=hmm_feats,
        min_train_days=CFG.MIN_TRAIN_DAYS,
        refit_freq=CFG.REFIT_FREQ,
        hmm_kwargs=hmm_kwargs,
    )
    print(f"  HMM regime labels: {len(hmm_regimes)} days  "
          f"[Calm={( hmm_regimes==0).sum()}, "
          f"Trans={(hmm_regimes==1).sum()}, "
          f"Crisis={(hmm_regimes==2).sum()}]")

    if hasattr(hmm_last_model, 'transition_matrix'):
        print("\n  HMM Transition Matrix:")
        print(pd.DataFrame(
            hmm_last_model.transition_matrix,
            index=["Calm","Trans","Crisis"],
            columns=["Calm","Trans","Crisis"],
        ).round(3).to_string())

    # 3c. ML (GMM + supervised forecaster) walk-forward
    print(f"\nFitting ML pipeline (GMM + {CFG.ML_FORECAST_MODEL}, walk-forward) …")

    # Build aligned next-day return targets
    # Targets: log_ret_sectors shifted by -1 (next-day return)
    next_day_ret = log_ret_sectors.shift(-1).dropna()
    ml_kwargs = dict(
        gmm_n_components=CFG.GMM_N_COMPONENTS,
        gmm_n_init=CFG.GMM_N_INIT,
        gmm_covariance_type=CFG.GMM_COVARIANCE_TYPE,
        gmm_random_state=CFG.GMM_RANDOM_STATE,
        forecast_model=CFG.ML_FORECAST_MODEL,
        rf_n_estimators=CFG.RF_N_ESTIMATORS,
        rf_max_depth=CFG.RF_MAX_DEPTH,
        rf_min_samples=CFG.RF_MIN_SAMPLES,
        rf_max_features=CFG.RF_MAX_FEATURES,
        gb_learning_rate=CFG.GB_LEARNING_RATE,
        gb_l2_regularization=CFG.GB_L2_REGULARIZATION,
        rf_random_state=CFG.RF_RANDOM_STATE,
        rf_n_jobs=CFG.RF_N_JOBS,
    )

    ml_regimes, ml_return_preds, ml_last_pipeline = fit_ml_walkforward(
        gmm_features=gmm_feats,
        rf_features=rf_features,
        log_returns=next_day_ret,
        min_train_days=CFG.MIN_TRAIN_DAYS,
        refit_freq=CFG.REFIT_FREQ,
        ml_kwargs=ml_kwargs,
    )
    print(f"  ML regime labels:  {len(ml_regimes)} days  "
          f"[Calm={( ml_regimes==0).sum()}, "
          f"Trans={(ml_regimes==1).sum()}, "
          f"Crisis={(ml_regimes==2).sum()}]")

    ensemble_regimes = majority_vote_regimes(vix_regimes, hmm_regimes, ml_regimes)
    print(f"  Ensemble labels:   {len(ensemble_regimes)} days  "
          f"[Calm={( ensemble_regimes==0).sum()}, "
          f"Trans={(ensemble_regimes==1).sum()}, "
          f"Crisis={(ensemble_regimes==2).sum()}]")

    # ═══════════════════════════════════════════════════════════════
    # 4.  CAPM BETA ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    banner("Step 4/7 — Regime-Conditional CAPM Betas")

    beta_full = estimate_capm_betas(excess_sectors, excess_spy, vix_regimes)
    beta_summ = beta_summary_table(beta_full)
    print("\nCAPM Beta Summary (VIX regimes):")
    print(beta_summ.round(3).to_string())
    beta_summ.to_csv(os.path.join(CFG.RESULTS_DIR, "beta_summary.csv"))

    # ═══════════════════════════════════════════════════════════════
    # 5.  BACKTEST
    # ═══════════════════════════════════════════════════════════════
    banner("Step 5/7 — Walk-Forward Backtest")

    # Align all regime labels to common index
    common_idx = (
        excess_sectors.index
        .intersection(excess_spy.index)
        .intersection(vix_regimes.index)
        .intersection(hmm_regimes.index)
        .intersection(ml_regimes.index)
        .intersection(ensemble_regimes.index)
        .sort_values()
    )
    print(f"  Common backtest window: {common_idx[0].date()} → {common_idx[-1].date()} "
          f"({len(common_idx)} days)")

    # Align ml_return_preds to common index
    ml_preds_aligned = ml_return_preds.reindex(common_idx).ffill().bfill()
    if ml_preds_aligned.empty or ml_preds_aligned.isna().all().all():
        # Fallback: use rolling mean
        ml_preds_aligned = excess_sectors.reindex(common_idx).rolling(21).mean().fillna(0)

    port_returns = run_backtest(
        excess_returns=excess_sectors,
        spy_excess=excess_spy,
        rf_daily=rf,
        vix_regimes=vix_regimes,
        hmm_regimes=hmm_regimes,
        ml_regimes=ml_regimes,
        ensemble_regimes=ensemble_regimes,
        ml_return_preds=ml_preds_aligned,
        rolling_window=CFG.ROLLING_WINDOW,
        rebalance_freq=CFG.REBALANCE_FREQ,
        cost_bps=CFG.TRANSACTION_COST_BPS,
        crisis_cash=CFG.CRISIS_CASH_FRACTION,
        use_shrinkage=CFG.USE_LEDOIT_WOLF,
    )

    print(f"  Strategy return matrix: {port_returns.shape}")

    # Cumulative wealth
    rf_bt = rf.reindex(port_returns.index).ffill().bfill()
    wealth = cumulative_wealth(port_returns, rf_bt, initial_wealth=1.0)

    # Weight history for plotting
    print("  Recording portfolio weight history …")
    wt_hist = weight_history(
        excess_returns=excess_sectors,
        spy_excess=excess_spy,
        rf_daily=rf,
        vix_regimes=vix_regimes,
        hmm_regimes=hmm_regimes,
        ml_regimes=ml_regimes,
        ensemble_regimes=ensemble_regimes,
        ml_return_preds=ml_preds_aligned,
        rolling_window=CFG.ROLLING_WINDOW,
        rebalance_freq=CFG.REBALANCE_FREQ,
        crisis_cash=CFG.CRISIS_CASH_FRACTION,
        use_shrinkage=CFG.USE_LEDOIT_WOLF,
    )

    # ═══════════════════════════════════════════════════════════════
    # 6.  PERFORMANCE METRICS
    # ═══════════════════════════════════════════════════════════════
    banner("Step 6/7 — Performance Metrics")

    spy_sr = sharpe_ratio(excess_spy.reindex(port_returns.index).ffill())

    # Measured annual one-way turnover from the backtest (replaces hand-set heuristics)
    turnover_dict = dict(port_returns.attrs.get("annual_turnover", {}))
    if turnover_dict:
        print("\nMeasured annual one-way turnover:")
        for name in STRATEGY_NAMES:
            print(f"  {name:<11} {turnover_dict.get(name, float('nan')):6.1%}")

    # Gross (pre-cost) returns → break-even reads as a TOTAL one-way cost
    gross_returns = port_returns.attrs.get("gross_returns")

    perf_table = full_performance_table(
        port_returns=port_returns,
        spy_excess=excess_spy,
        rf_daily=rf,
        turnover_dict=turnover_dict,
        gross_returns=gross_returns,
    )
    print("\nPerformance Summary:")
    print(perf_table.to_string())
    perf_table.to_csv(os.path.join(CFG.RESULTS_DIR, "performance_summary.csv"))

    # Persist raw return matrices so significance tests can be re-run without
    # refitting the (slow) regime models.
    port_returns.to_csv(os.path.join(CFG.RESULTS_DIR, "port_returns_net.csv"))
    if gross_returns is not None:
        gross_returns.to_csv(os.path.join(CFG.RESULTS_DIR, "port_returns_gross.csv"))

    # ── Multiple-testing / data-snooping corrections ──────────────────────────
    print("\nMultiple-Testing Corrections (active strategies searched):")
    candidates = port_returns.drop(columns=["SPY B&H"], errors="ignore")
    dsr = deflated_sharpe_ratio(candidates)
    rc  = whites_reality_check(port_returns, excess_spy, n_boot=2000, avg_block=10.0)

    sig_rows = []
    if dsr:
        print(f"  Deflated Sharpe Ratio (Bailey & López de Prado 2014):")
        print(f"    Best strategy           : {dsr['selected']}")
        print(f"    Trials (N)              : {dsr['n_trials']}")
        print(f"    Best Sharpe (ann.)      : {dsr['best_sr_annual']:.3f}")
        print(f"    Null expected-max Sharpe: {dsr['sr0_annual']:.3f}")
        print(f"    DSR  P(SR>SR0)          : {dsr['dsr']:.4f}"
              f"   → {'significant' if (dsr['dsr'] or 0) > 0.95 else 'NOT significant'} at 95%")
        sig_rows.append({"Test": "Deflated Sharpe Ratio", "Strategy": dsr["selected"],
                         "Statistic": round(dsr["best_sr_annual"], 4),
                         "Null/Threshold": round(dsr["sr0_annual"], 4),
                         "p_or_prob": round(dsr["dsr"], 4) if dsr["dsr"] is not None else None})
    if rc:
        print(f"  White's Reality Check (2000), stationary bootstrap:")
        print(f"    Best strategy vs SPY    : {rc['best']}")
        print(f"    Bootstraps              : {rc['n_boot']} (avg block {rc['avg_block']:.0f})")
        print(f"    p-value                 : {rc['p_value']:.4f}"
              f"   → {'reject H0' if rc['p_value'] < 0.05 else 'CANNOT reject H0'} at 5%")
        sig_rows.append({"Test": "White Reality Check", "Strategy": rc["best"],
                         "Statistic": round(rc["V"], 6), "Null/Threshold": "max over %d strats" % rc["n_strategies"],
                         "p_or_prob": round(rc["p_value"], 4)})
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(
            os.path.join(CFG.RESULTS_DIR, "significance_tests.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════
    # 7.  VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    banner("Step 7/7 — Generating Figures")

    # Assemble data dict for plotting
    plot_data = {
        "vix":             vix,
        "vix_regimes":     vix_regimes,
        "hmm_regimes":     hmm_regimes,
        "ml_regimes":      ml_regimes,
        "ensemble_regimes": ensemble_regimes,
        "hmm_model":       hmm_last_model,
        "log_returns":     log_ret_sectors,
        "excess_returns":  excess_sectors,
        "spy_excess":      excess_spy,
        "wealth":          wealth,
        "port_returns":    port_returns,
        "rf_daily":        rf,
        "beta_summary":    beta_summ,
        "weight_history":  wt_hist,
        "ml_pipeline":     ml_last_pipeline,
        "rf_feature_names": rf_feature_names,
        "perf_table":      perf_table,
        "spy_sr":          spy_sr,
        "turnover_dict":   turnover_dict,
        "gross_returns":   gross_returns,
    }

    generate_all_figures(
        data=plot_data,
        out_dir=CFG.RESULTS_DIR,
        cost_range_bps=CFG.BREAKEVEN_COST_RANGE,
    )

    # ═══════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    banner(f"✓ Pipeline complete in {elapsed:.1f}s")
    print(f"\nOutputs saved to: {os.path.abspath(CFG.RESULTS_DIR)}/")
    print("  ├── performance_summary.csv")
    print("  ├── regime_stats.csv")
    print("  ├── beta_summary.csv")
    print("  └── 01_regime_timeline.png … 11_feature_importances.png")


if __name__ == "__main__":
    main()
