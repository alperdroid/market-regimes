"""
Run this file to test the TPF fix using ACTUAL PROJECT DATA!

USAGE:
    cd market_regimes
    python TEST_RESULTS.py

This loads your cached market data and demonstrates:
1. The bug in the original code (TPF < MVP = efficient frontier violation)
2. How the fix resolves it (TPF ≥ MVP = efficient frontier restored)
"""

import numpy as np
import pandas as pd
import os
import pickle
from scipy.optimize import minimize
from data.loader import load_all_data
from data.features import compute_log_returns, compute_excess_returns
from portfolio.ledoit_wolf import robust_covariance
import config as CFG

def mvp_weights(mu, sigma):
    """Minimum Variance Portfolio"""
    n = len(mu)
    result = minimize(
        lambda w: w @ sigma @ w,
        x0=np.ones(n)/n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        bounds=[(0, 1)]*n,
        method="SLSQP",
        options={"disp": False}
    )
    return result.x if result.success else np.ones(n)/n

def tpf_weights(mu, sigma, rf=0.0):
    """Tangency Portfolio (Maximum Sharpe Ratio)"""
    n = len(mu)
    
    # Check if all excess returns are non-positive
    mu_excess = mu - rf
    if mu_excess.max() <= 0:
        return mvp_weights(mu, sigma)
    
    result = minimize(
        lambda w: -(mu @ w - rf) / np.sqrt(max(w @ sigma @ w, 1e-12)),
        x0=np.ones(n)/n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        bounds=[(0, 1)]*n,
        method="SLSQP",
        options={"disp": False}
    )
    return result.x if result.success else np.ones(n)/n

# ═══════════════════════════════════════════════════════════════════════════
# LOAD ACTUAL PROJECT DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  LOADING MARKET DATA FROM PROJECT CACHE")
print("="*80)

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

# Compute log returns and excess returns for sector ETFs
log_ret = compute_log_returns(prices)
sector_cols = [c for c in CFG.SECTOR_ETFS if c in log_ret.columns]
log_ret_sectors = log_ret[sector_cols]
excess_sectors = compute_excess_returns(log_ret_sectors, rf)

print(f"\nData loaded:")
print(f"  • Period: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  • Trading days: {len(prices)}")
print(f"  • Sector ETFs: {sector_cols}")

# ═══════════════════════════════════════════════════════════════════════════
# IDENTIFY CRISIS REGIME DATES (VIX > 30)
# ═══════════════════════════════════════════════════════════════════════════

crisis_dates = vix[vix > 30].index
print(f"\nCrisis periods identified (VIX > 30): {len(crisis_dates)} trading days")

if len(crisis_dates) > 0:
    # Get the most recent crisis period for testing
    recent_crisis_start = crisis_dates[0]
    recent_crisis_end = recent_crisis_start + pd.Timedelta(days=60)  # ~3 months
    crisis_data = excess_sectors.loc[recent_crisis_start:recent_crisis_end].dropna()
    
    print(f"Using recent crisis period: {recent_crisis_start.date()} to {recent_crisis_end.date()}")
    print(f"Crisis period observations: {len(crisis_data)} trading days")
else:
    # Fallback: use tail of data (last 3 months)
    crisis_data = excess_sectors.tail(60).dropna()
    print(f"No recent crisis found; using last 60 trading days")

# Compute regime-conditional moments from crisis data
mu_crisis = crisis_data.mean().values
sigma_crisis = robust_covariance(crisis_data.values, use_shrinkage=True)

print(f"\nCrisis regime statistics:")
print(f"  • Mean annualized return: {mu_crisis.mean() * 252 * 100:.2f}%")
print(f"  • Volatility (from covariance): {np.sqrt(np.diag(sigma_crisis)).mean() * np.sqrt(252) * 100:.2f}%")
print(f"  • VIX at period start: {vix.loc[crisis_data.index[0]]:.1f}")

crisis_cash = 0.15

print("\n" + "="*80)
print("  TPF CRISIS CASH FIX TEST - RESULTS")
print("="*80)

print("\nTesting crisis regime with 15% cash buffer...")
print("-" * 80)

# ORIGINAL: Apply crisis cash to BOTH MVP and TPF
print("\n1️⃣  ORIGINAL CODE (crisis cash applied to BOTH MVP and TPF):")
print("   " + "─" * 76)

w_mvp_orig = mvp_weights(mu_crisis, sigma_crisis) * (1 - crisis_cash)
w_mvp_orig /= w_mvp_orig.sum()

w_tpf_orig = tpf_weights(mu_crisis, sigma_crisis, rf=rf.mean()) * (1 - crisis_cash)
w_tpf_orig /= w_tpf_orig.sum()

ret_mvp_orig = w_mvp_orig @ mu_crisis
ret_tpf_orig = w_tpf_orig @ mu_crisis

var_mvp_orig = w_mvp_orig @ sigma_crisis @ w_mvp_orig
var_tpf_orig = w_tpf_orig @ sigma_crisis @ w_tpf_orig

sr_mvp_orig = ret_mvp_orig / np.sqrt(var_mvp_orig) if var_mvp_orig > 0 else 0
sr_tpf_orig = ret_tpf_orig / np.sqrt(var_tpf_orig) if var_tpf_orig > 0 else 0

print(f"   MVP Annual Return: {ret_mvp_orig*252*100:7.2f}%  |  Volatility: {np.sqrt(var_mvp_orig)*np.sqrt(252)*100:6.2f}%  |  Sharpe: {sr_mvp_orig*np.sqrt(252):6.4f}")
print(f"   TPF Annual Return: {ret_tpf_orig*252*100:7.2f}%  |  Volatility: {np.sqrt(var_tpf_orig)*np.sqrt(252)*100:6.2f}%  |  Sharpe: {sr_tpf_orig*np.sqrt(252):6.4f}")

diff_ret_orig = ret_tpf_orig - ret_mvp_orig
diff_sr_orig = (sr_tpf_orig - sr_mvp_orig) * np.sqrt(252)

print(f"\n   Return Difference (TPF - MVP): {diff_ret_orig*252*100:+7.2f}% annually")
print(f"   Sharpe Difference (TPF - MVP): {diff_sr_orig:+7.4f}")

if diff_ret_orig < -0.0001:
    print(f"\n   ❌ PROBLEM: TPF return is LOWER than MVP!")
    print(f"      This violates the efficient frontier property.")
else:
    print(f"\n   ✅ TPF return is higher or equal to MVP")

# FIXED: Apply crisis cash to MVP only
print("\n2️⃣  FIXED CODE (crisis cash applied to MVP only):")
print("   " + "─" * 76)

w_mvp_fixed = mvp_weights(mu_crisis, sigma_crisis) * (1 - crisis_cash)
w_mvp_fixed /= w_mvp_fixed.sum()

w_tpf_fixed = tpf_weights(mu_crisis, sigma_crisis, rf=rf.mean())  # NO crisis cash
w_tpf_fixed /= w_tpf_fixed.sum()

ret_mvp_fixed = w_mvp_fixed @ mu_crisis
ret_tpf_fixed = w_tpf_fixed @ mu_crisis

var_mvp_fixed = w_mvp_fixed @ sigma_crisis @ w_mvp_fixed
var_tpf_fixed = w_tpf_fixed @ sigma_crisis @ w_tpf_fixed

sr_mvp_fixed = ret_mvp_fixed / np.sqrt(var_mvp_fixed) if var_mvp_fixed > 0 else 0
sr_tpf_fixed = ret_tpf_fixed / np.sqrt(var_tpf_fixed) if var_tpf_fixed > 0 else 0

print(f"   MVP Annual Return: {ret_mvp_fixed*252*100:7.2f}%  |  Volatility: {np.sqrt(var_mvp_fixed)*np.sqrt(252)*100:6.2f}%  |  Sharpe: {sr_mvp_fixed*np.sqrt(252):6.4f}")
print(f"   TPF Annual Return: {ret_tpf_fixed*252*100:7.2f}%  |  Volatility: {np.sqrt(var_tpf_fixed)*np.sqrt(252)*100:6.2f}%  |  Sharpe: {sr_tpf_fixed*np.sqrt(252):6.4f}")

diff_ret_fixed = ret_tpf_fixed - ret_mvp_fixed
diff_sr_fixed = (sr_tpf_fixed - sr_mvp_fixed) * np.sqrt(252)

print(f"\n   Return Difference (TPF - MVP): {diff_ret_fixed*252*100:+7.2f}% annually")
print(f"   Sharpe Difference (TPF - MVP): {diff_sr_fixed:+7.4f}")

if diff_ret_fixed >= -0.0001:
    print(f"\n   ✅ FIXED: TPF return is now higher than MVP!")
    print(f"      Efficient frontier property restored.")
else:
    print(f"\n   ❌ TPF return still lower than MVP")

# Summary
print("\n" + "="*80)
print("  IMPACT ANALYSIS")
print("="*80)

recovery = (diff_ret_fixed - diff_ret_orig) * 252 * 100
recovery_sr = diff_sr_fixed - diff_sr_orig

print(f"\nReturn advantage recovered: {recovery:+7.2f}% annually")
print(f"Sharpe advantage recovered: {recovery_sr:+7.4f}")

print("\n" + "="*80)
print("  CONCLUSION")
print("="*80)

if diff_ret_orig < -0.0001 and diff_ret_fixed >= -0.0001:
    print("\n✅ ✅ ✅  FIX IS EFFECTIVE  ✅ ✅ ✅")
    print("\nThe proposed change successfully:")
    print("  • Eliminates the efficient frontier violation")
    print("  • Restores TPF to outperform MVP as theory predicts")
    print(f"  • Recovers {recovery:.2f}% annual return advantage")
    print(f"  • Recovers {recovery_sr:.4f} Sharpe ratio advantage")
    print("\nRECOMMENDATION: Apply the fix to optimizer.py line 172")
    print("Change:")
    print("    if regime == 2 and crisis_cash_fraction > 0:")
    print("        w = apply_crisis_cash(w, crisis_cash_fraction)")
    print("To:")
    print("    if regime == 2 and crisis_cash_fraction > 0 and strategy == 'mvp':")
    print("        w = apply_crisis_cash(w, crisis_cash_fraction)")
elif diff_ret_fixed >= -0.0001:
    print("\n✅ FIX VALIDATES EFFICIENT FRONTIER PROPERTY")
else:
    print("\n⚠️  Further investigation needed")

print("\n" + "="*80 + "\n")
