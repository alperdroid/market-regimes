"""
test_tpf_fix.py
---------------
Standalone test to validate the TPF crisis cash fix.

This script runs a simplified version of the backtest with and without the fix,
comparing TPF vs MVP returns to verify the efficient frontier property is restored.

Usage:
    python test_tpf_fix.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import config as CFG
from data.loader import load_all_data
from data.features import compute_log_returns, compute_excess_returns, build_hmm_features
from regimes.vix_classifier import classify_vix_regimes
from regimes.hmm_model import fit_hmm_walkforward
from portfolio.ledoit_wolf import robust_covariance
from portfolio.optimizer import (
    compute_portfolio_weights,
    minimum_variance_portfolio,
    tangency_portfolio,
)
from performance.metrics import annualised_return, annualised_vol, sharpe_ratio


def compute_portfolio_weights_original(
    mu: np.ndarray,
    sigma: np.ndarray,
    strategy: str,
    rf_daily: float = 0.0,
    regime: int = None,
    crisis_cash_fraction: float = 0.15,
) -> np.ndarray:
    """Original version: applies crisis cash to both MVP and TPF."""
    n = len(mu)
    
    if strategy == "ew":
        w = np.ones(n) / n
    elif strategy == "mvp":
        w = minimum_variance_portfolio(sigma, 0.0, 1.0)
    elif strategy == "tpf":
        w = tangency_portfolio(mu, sigma, rf_daily, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    
    # ORIGINAL BEHAVIOR: Apply cash buffer to BOTH strategies in crisis
    if regime == 2 and crisis_cash_fraction > 0:
        w = w * (1.0 - crisis_cash_fraction)
    
    return w


def compute_portfolio_weights_fixed(
    mu: np.ndarray,
    sigma: np.ndarray,
    strategy: str,
    rf_daily: float = 0.0,
    regime: int = None,
    crisis_cash_fraction: float = 0.15,
) -> np.ndarray:
    """Fixed version: applies crisis cash to MVP only, keeps TPF on efficient frontier."""
    n = len(mu)
    
    if strategy == "ew":
        w = np.ones(n) / n
    elif strategy == "mvp":
        w = minimum_variance_portfolio(sigma, 0.0, 1.0)
    elif strategy == "tpf":
        w = tangency_portfolio(mu, sigma, rf_daily, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    
    # FIXED BEHAVIOR: Apply cash buffer to MVP only
    if regime == 2 and crisis_cash_fraction > 0 and strategy == "mvp":
        w = w * (1.0 - crisis_cash_fraction)
    
    return w


def mini_backtest(
    excess_returns: pd.DataFrame,
    vix_regimes: pd.Series,
    rf_daily: pd.Series,
    weight_func,
    window: int = 252,
    rebal_freq: int = 21,
    crisis_cash: float = 0.15,
) -> pd.DataFrame:
    """
    Simplified walk-forward backtest for MVP vs TPF.
    
    Returns DataFrame with columns: ['MVP', 'TPF']
    """
    er_np = excess_returns.to_numpy()
    vr_np = vix_regimes.to_numpy()
    T, N = er_np.shape
    dates = excess_returns.index
    
    port_rets = np.zeros((T, 2))  # [MVP, TPF]
    w_mvp = np.ones(N) / N
    w_tpf = np.ones(N) / N
    
    rebal_days = set(range(0, T, rebal_freq))
    
    for t in range(window, T):
        hist = er_np[max(0, t - window):t]
        rf_t = float(rf_daily.iloc[t])
        regime_t = int(vr_np[t])
        today_ret = er_np[t]
        
        if t in rebal_days:
            mu = hist.mean(axis=0)
            sigma = robust_covariance(hist, use_shrinkage=True)
            
            # Compute weights using the test function (original or fixed)
            w_mvp = weight_func(mu, sigma, "mvp", rf_t, regime_t, crisis_cash)
            w_tpf = weight_func(mu, sigma, "tpf", rf_t, regime_t, crisis_cash)
        
        # Daily returns
        port_rets[t] = [
            float(w_mvp @ today_ret),
            float(w_tpf @ today_ret),
        ]
    
    return pd.DataFrame(
        port_rets[window:],
        index=dates[window:],
        columns=["MVP", "TPF"],
    )


def main():
    print("\n" + "="*70)
    print("  TPF Crisis Cash Fix Validation Test")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading market data...")
    data_raw = load_all_data(
        start=CFG.START_DATE,
        end=CFG.END_DATE,
        sector_etfs=CFG.SECTOR_ETFS,
        benchmark=CFG.BENCHMARK,
        cache_path=CFG.DATA_CACHE,
        force_reload=False,
    )
    
    prices = data_raw["prices"]
    vix = data_raw["vix"]
    rf = data_raw["rf"]
    
    # Feature engineering
    print("[2/4] Computing returns and regimes...")
    log_ret_all = compute_log_returns(prices)
    sector_cols = [c for c in CFG.SECTOR_ETFS if c in log_ret_all.columns]
    log_ret_sectors = log_ret_all[sector_cols]
    excess_sectors = compute_excess_returns(log_ret_sectors, rf)
    
    # Regime classification
    vix_regimes = classify_vix_regimes(
        vix,
        calm_threshold=CFG.VIX_CALM_THRESHOLD,
        transitional_threshold=CFG.VIX_TRANSITIONAL_THRESHOLD,
    )
    
    # Align
    common_idx = (
        excess_sectors.index
        .intersection(vix_regimes.index)
        .sort_values()
    )
    excess_sectors = excess_sectors.loc[common_idx]
    vix_regimes = vix_regimes.loc[common_idx]
    rf = rf.reindex(common_idx).ffill().bfill()
    
    print(f"   Backtest window: {common_idx[0].date()} → {common_idx[-1].date()}")
    print(f"   Total days: {len(common_idx)}")
    
    # Run backtests
    print("\n[3/4] Running backtests...")
    print("   • Testing ORIGINAL code (crisis cash applied to both MVP and TPF)...")
    results_original = mini_backtest(
        excess_sectors, vix_regimes, rf, compute_portfolio_weights_original
    )
    
    print("   • Testing FIXED code (crisis cash applied to MVP only)...")
    results_fixed = mini_backtest(
        excess_sectors, vix_regimes, rf, compute_portfolio_weights_fixed
    )
    
    # Compute metrics
    print("\n[4/4] Computing performance metrics...\n")
    
    def report_strategy(results, label):
        mvp_ret = annualised_return(results["MVP"])
        mvp_vol = annualised_vol(results["MVP"])
        mvp_sr = sharpe_ratio(results["MVP"])
        
        tpf_ret = annualised_return(results["TPF"])
        tpf_vol = annualised_vol(results["TPF"])
        tpf_sr = sharpe_ratio(results["TPF"])
        
        print(f"\n{label}")
        print("-" * 70)
        print(f"{'Metric':<20} {'MVP':<20} {'TPF':<20} {'TPF - MVP':<20}")
        print("-" * 70)
        print(f"{'Ann. Return (%)':<20} {mvp_ret*100:>18.2f} {tpf_ret*100:>18.2f} {(tpf_ret-mvp_ret)*100:>18.2f}")
        print(f"{'Ann. Vol (%)':<20} {mvp_vol*100:>18.2f} {tpf_vol*100:>18.2f} {(tpf_vol-mvp_vol)*100:>18.2f}")
        print(f"{'Sharpe Ratio':<20} {mvp_sr:>18.4f} {tpf_sr:>18.4f} {tpf_sr-mvp_sr:>18.4f}")
        
        # Check efficient frontier property
        print("\n" + "─" * 70)
        if tpf_ret >= mvp_ret - 0.001:  # Allow small numerical tolerance
            print("✓ EFFICIENT FRONTIER PROPERTY SATISFIED: TPF return ≥ MVP return")
        else:
            print(f"✗ EFFICIENT FRONTIER VIOLATION: TPF return < MVP return (diff: {(mvp_ret-tpf_ret)*100:.3f}%)")
        
        return {
            "mvp_ret": mvp_ret,
            "mvp_vol": mvp_vol,
            "mvp_sr": mvp_sr,
            "tpf_ret": tpf_ret,
            "tpf_vol": tpf_vol,
            "tpf_sr": tpf_sr,
        }
    
    metrics_orig = report_strategy(results_original, "ORIGINAL CODE (Crisis cash on both)")
    metrics_fixed = report_strategy(results_fixed, "FIXED CODE (Crisis cash on MVP only)")
    
    # Summary comparison
    print("\n" + "="*70)
    print("  COMPARISON: Original vs Fixed")
    print("="*70)
    print(f"{'Metric':<25} {'Original':<20} {'Fixed':<20} {'Change':<20}")
    print("-" * 70)
    print(f"{'TPF Return Gain':<25} {(metrics_orig['tpf_ret']-metrics_orig['mvp_ret'])*100:>18.3f}% {(metrics_fixed['tpf_ret']-metrics_fixed['mvp_ret'])*100:>18.3f}% {((metrics_fixed['tpf_ret']-metrics_fixed['mvp_ret'])-(metrics_orig['tpf_ret']-metrics_orig['mvp_ret']))*100:>18.3f}%")
    print(f"{'TPF Sharpe Advantage':<25} {metrics_orig['tpf_sr']-metrics_orig['mvp_sr']:>18.4f} {metrics_fixed['tpf_sr']-metrics_fixed['mvp_sr']:>18.4f} {(metrics_fixed['tpf_sr']-metrics_fixed['mvp_sr'])-(metrics_orig['tpf_sr']-metrics_orig['mvp_sr']):>18.4f}")
    print("-" * 70)
    
    print("\n" + "="*70)
    print("  CONCLUSION")
    print("="*70)
    
    ef_violated_orig = metrics_orig['tpf_ret'] < metrics_orig['mvp_ret'] - 0.001
    ef_violated_fixed = metrics_fixed['tpf_ret'] < metrics_fixed['mvp_ret'] - 0.001
    
    if ef_violated_orig and not ef_violated_fixed:
        print("✓ FIX SUCCESSFUL!")
        print("  • Original code violated efficient frontier property")
        print("  • Fixed code restores efficient frontier property")
        print("  • TPF now correctly outperforms MVP by ~%.2f%% annually" % ((metrics_fixed['tpf_ret']-metrics_fixed['mvp_ret'])*100))
    elif not ef_violated_fixed:
        print("✓ FIX VALIDATES THEORETICAL EXPECTATION")
        print("  • Both versions satisfy efficient frontier")
        print("  • Fixed version provides cleaner risk/return decomposition")
    else:
        print("✗ FIX DID NOT RESOLVE ISSUE")
        print("  • Investigate other sources of bias in return estimation")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
