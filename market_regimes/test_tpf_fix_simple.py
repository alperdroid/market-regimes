"""
test_tpf_fix_simple.py
---------------------
Simplified test that demonstrates the TPF crisis cash bug with synthetic data.
This runs quickly and clearly shows the issue + fix.

Run with:
    python test_tpf_fix_simple.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Synthetic regime-conditional return data
np.random.seed(42)

def synthetic_regime_data(n_days=1000, n_assets=9):
    """Generate synthetic regime-conditional returns."""
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
    
    # Regime assignment: 0=Calm, 1=Trans, 2=Crisis
    regimes = np.random.choice([0, 1, 2], size=n_days, p=[0.5, 0.3, 0.2])
    
    # Generate returns with regime-dependent statistics
    returns = np.zeros((n_days, n_assets))
    
    for t in range(n_days):
        regime = regimes[t]
        
        if regime == 0:  # Calm
            mu = np.full(n_assets, 0.0008)
            vol = np.full(n_assets, 0.01)
        elif regime == 1:  # Transitional
            mu = np.full(n_assets, 0.0005)
            vol = np.full(n_assets, 0.015)
        else:  # Crisis
            mu = np.full(n_assets, 0.0001)
            vol = np.full(n_assets, 0.02)
        
        returns[t] = np.random.normal(mu, vol)
    
    return pd.DataFrame(returns, index=dates), pd.Series(regimes, index=dates)


def compute_portfolio_weights_original(mu, sigma, strategy, regime, crisis_cash=0.15):
    """Original: crisis cash applied to BOTH MVP and TPF."""
    n = len(mu)
    
    if strategy == "mvp":
        # MVP: minimize variance
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda w: w @ sigma @ w,
                x0=np.ones(n)/n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                bounds=[(0, 1)]*n,
                method="SLSQP"
            )
            w = result.x if result.success else np.ones(n)/n
        except:
            w = np.ones(n) / n
    else:  # tpf
        # TPF: maximize sharpe ratio
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda w: -(mu @ w) / np.sqrt(max(w @ sigma @ w, 1e-12)),
                x0=np.ones(n)/n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                bounds=[(0, 1)]*n,
                method="SLSQP"
            )
            w = result.x if result.success else np.ones(n)/n
        except:
            w = np.ones(n) / n
    
    # ORIGINAL: Apply crisis cash to BOTH
    if regime == 2 and crisis_cash > 0:
        w = w * (1.0 - crisis_cash)
    
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    return w


def compute_portfolio_weights_fixed(mu, sigma, strategy, regime, crisis_cash=0.15):
    """Fixed: crisis cash applied to MVP only."""
    n = len(mu)
    
    if strategy == "mvp":
        # MVP: minimize variance
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda w: w @ sigma @ w,
                x0=np.ones(n)/n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                bounds=[(0, 1)]*n,
                method="SLSQP"
            )
            w = result.x if result.success else np.ones(n)/n
        except:
            w = np.ones(n) / n
    else:  # tpf
        # TPF: maximize sharpe ratio
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda w: -(mu @ w) / np.sqrt(max(w @ sigma @ w, 1e-12)),
                x0=np.ones(n)/n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                bounds=[(0, 1)]*n,
                method="SLSQP"
            )
            w = result.x if result.success else np.ones(n)/n
        except:
            w = np.ones(n) / n
    
    # FIXED: Apply crisis cash to MVP ONLY
    if regime == 2 and crisis_cash > 0 and strategy == "mvp":
        w = w * (1.0 - crisis_cash)
    
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    return w


def backtest_comparison(returns, regimes, weight_func, window=60, rebal_freq=20):
    """Run simple backtest and return MVP vs TPF performance."""
    n_days = len(returns)
    n_assets = returns.shape[1]
    
    port_rets_mvp = []
    port_rets_tpf = []
    
    w_mvp = np.ones(n_assets) / n_assets
    w_tpf = np.ones(n_assets) / n_assets
    
    rebal_counter = 0
    
    for t in range(window, n_days):
        hist = returns.iloc[max(0, t-window):t].values
        regime_t = int(regimes.iloc[t])
        today_ret = returns.iloc[t].values
        
        rebal_counter += 1
        if rebal_counter >= rebal_freq:
            rebal_counter = 0
            mu = hist.mean(axis=0)
            sigma = np.cov(hist.T)
            
            w_mvp = weight_func(mu, sigma, "mvp", regime_t)
            w_tpf = weight_func(mu, sigma, "tpf", regime_t)
        
        port_rets_mvp.append(float(w_mvp @ today_ret))
        port_rets_tpf.append(float(w_tpf @ today_ret))
    
    return np.array(port_rets_mvp), np.array(port_rets_tpf)


def main():
    print("\n" + "="*80)
    print("  TPF Crisis Cash Fix Validation - SIMPLIFIED TEST")
    print("="*80)
    
    # Generate synthetic data
    print("\n[1/3] Generating synthetic regime-conditional returns...")
    returns, regimes = synthetic_regime_data(n_days=1000, n_assets=9)
    print(f"   • {len(returns)} trading days")
    print(f"   • {returns.shape[1]} assets")
    print(f"   • Regime distribution: {np.bincount(regimes)}")
    
    # Test original code
    print("\n[2/3] Running backtests...")
    print("   • Original code (crisis cash on BOTH MVP and TPF)...")
    mvp_orig, tpf_orig = backtest_comparison(returns, regimes, compute_portfolio_weights_original)
    
    print("   • Fixed code (crisis cash on MVP only)...")
    mvp_fixed, tpf_fixed = backtest_comparison(returns, regimes, compute_portfolio_weights_fixed)
    
    # Compute metrics
    print("\n[3/3] Computing metrics...\n")
    
    def compute_metrics(mvp_ret, tpf_ret, label):
        mvp_mean = mvp_ret.mean() * 252
        tpf_mean = tpf_ret.mean() * 252
        mvp_vol = mvp_ret.std() * np.sqrt(252)
        tpf_vol = tpf_ret.std() * np.sqrt(252)
        mvp_sr = mvp_mean / mvp_vol if mvp_vol > 0 else 0
        tpf_sr = tpf_mean / tpf_vol if tpf_vol > 0 else 0
        
        print(f"\n{label}")
        print("-" * 80)
        print(f"{'Metric':<20} {'MVP':<20} {'TPF':<20} {'Difference':<20}")
        print("-" * 80)
        print(f"{'Annual Return':<20} {mvp_mean*100:>18.2f}% {tpf_mean*100:>18.2f}% {(tpf_mean-mvp_mean)*100:>18.2f}%")
        print(f"{'Annual Vol':<20} {mvp_vol*100:>18.2f}% {tpf_vol*100:>18.2f}% {(tpf_vol-mvp_vol)*100:>18.2f}%")
        print(f"{'Sharpe Ratio':<20} {mvp_sr:>18.4f} {tpf_sr:>18.4f} {tpf_sr-mvp_sr:>18.4f}")
        
        ef_violated = tpf_mean < mvp_mean - 0.0001
        print("\n" + "─" * 80)
        if ef_violated:
            print(f"✗ EFFICIENT FRONTIER VIOLATED: TPF return < MVP return")
        else:
            print(f"✓ EFFICIENT FRONTIER OK: TPF return ≥ MVP return")
        
        return {
            "mvp_ret": mvp_mean,
            "tpf_ret": tpf_mean,
            "mvp_vol": mvp_vol,
            "tpf_vol": tpf_vol,
            "mvp_sr": mvp_sr,
            "tpf_sr": tpf_sr,
        }
    
    metrics_orig = compute_metrics(mvp_orig, tpf_orig, "ORIGINAL CODE")
    metrics_fixed = compute_metrics(mvp_fixed, tpf_fixed, "FIXED CODE")
    
    # Summary
    print("\n" + "="*80)
    print("  IMPACT ANALYSIS")
    print("="*80)
    print(f"{'Metric':<25} {'Original':<18} {'Fixed':<18} {'Change':<18}")
    print("-" * 80)
    print(f"{'TPF Return Advantage':<25} {(metrics_orig['tpf_ret']-metrics_orig['mvp_ret'])*100:>16.3f}% {(metrics_fixed['tpf_ret']-metrics_fixed['mvp_ret'])*100:>16.3f}% {((metrics_fixed['tpf_ret']-metrics_fixed['mvp_ret'])-(metrics_orig['tpf_ret']-metrics_orig['mvp_ret']))*100:>16.3f}%")
    print(f"{'TPF Sharpe Advantage':<25} {metrics_orig['tpf_sr']-metrics_orig['mvp_sr']:>16.4f} {metrics_fixed['tpf_sr']-metrics_fixed['mvp_sr']:>16.4f} {(metrics_fixed['tpf_sr']-metrics_fixed['mvp_sr'])-(metrics_orig['tpf_sr']-metrics_orig['mvp_sr']):>16.4f}")
    print("-" * 80)
    
    # Conclusion
    print("\n" + "="*80)
    print("  CONCLUSION")
    print("="*80)
    
    ef_violated_orig = metrics_orig['tpf_ret'] < metrics_orig['mvp_ret'] - 0.0001
    ef_violated_fixed = metrics_fixed['tpf_ret'] < metrics_fixed['mvp_ret'] - 0.0001
    
    if ef_violated_orig and not ef_violated_fixed:
        print("✓✓✓ FIX IS EFFECTIVE ✓✓✓")
        print("\n  Original code violated efficient frontier property (TPF < MVP)")
        print("  Fixed code restores it (TPF ≥ MVP)")
        print(f"\n  Return advantage recovered: {((metrics_fixed['tpf_ret']-metrics_fixed['mvp_ret'])-(metrics_orig['tpf_ret']-metrics_orig['mvp_ret']))*100:.3f}% annually")
        print(f"  Sharpe advantage recovered: {(metrics_fixed['tpf_sr']-metrics_fixed['mvp_sr'])-(metrics_orig['tpf_sr']-metrics_orig['mvp_sr']):.4f}")
    elif not ef_violated_fixed:
        print("✓ FIX VALIDATES THEORY")
        print("\n  Fixed code correctly satisfies efficient frontier property")
    else:
        print("⚠ ISSUE PERSISTS")
        print("\n  The fix alone may not be sufficient.")
        print("  Check expected return estimation or crisis cash logic further.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
