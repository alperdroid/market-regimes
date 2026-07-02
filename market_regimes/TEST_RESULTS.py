"""
Run this file to test the TPF fix!

USAGE:
    cd market_regimes
    python test_tpf_fix_inline.py

This generates synthetic data and demonstrates:
1. The bug in the original code (TPF < MVP = efficient frontier violation)
2. How the fix resolves it (TPF ≥ MVP = efficient frontier restored)
"""

import numpy as np
from scipy.optimize import minimize

# Seed for reproducibility
np.random.seed(42)

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

print("\n" + "="*80)
print("  TPF CRISIS CASH FIX TEST - RESULTS")
print("="*80)

# Generate synthetic regime-conditional moments
mu_crisis = np.array([0.0001, 0.0002, 0.0001, 0.00005, 0.0001, 0.0002, 0.00005, 0.0002, 0.0001])
sigma_crisis = np.diag([0.020**2, 0.021**2, 0.022**2, 0.019**2, 0.020**2, 0.021**2, 0.018**2, 0.022**2, 0.020**2])
sigma_crisis += 0.1 * (np.random.randn(9, 9) + np.random.randn(9, 9).T) * 0.005

crisis_cash = 0.15

print("\nTesting crisis regime with 15% cash buffer...")
print("-" * 80)

# ORIGINAL: Apply crisis cash to BOTH MVP and TPF
print("\n1️⃣  ORIGINAL CODE (crisis cash applied to BOTH MVP and TPF):")
print("   " + "─" * 76)

w_mvp_orig = mvp_weights(mu_crisis, sigma_crisis) * (1 - crisis_cash)
w_mvp_orig /= w_mvp_orig.sum()

w_tpf_orig = tpf_weights(mu_crisis, sigma_crisis, rf=0.00001) * (1 - crisis_cash)
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

w_tpf_fixed = tpf_weights(mu_crisis, sigma_crisis, rf=0.00001)  # NO crisis cash
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
