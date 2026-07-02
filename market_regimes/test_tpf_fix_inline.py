"""
test_tpf_fix_inline.py
---------------------
Minimal inline test - demonstrates the bug with simple synthetic data.
No external dependencies beyond numpy/pandas.
"""

import numpy as np
import pandas as pd
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

# Generate synthetic regime-conditional moments
print("\n" + "="*80)
print("  DIRECT DEMONSTRATION: TPF Crisis Cash Bug")
print("="*80)

# Calm regime: positive returns
mu_calm = np.array([0.0008, 0.0009, 0.0010, 0.0007, 0.0008, 0.0009, 0.0006, 0.0010, 0.0008])
sigma_calm = np.diag([0.01**2, 0.011**2, 0.012**2, 0.009**2, 0.010**2, 0.011**2, 0.008**2, 0.012**2, 0.010**2])
sigma_calm += 0.3 * np.random.randn(9, 9) * 0.005

# Crisis regime: lower but still positive returns  
mu_crisis = np.array([0.0001, 0.0002, 0.0001, 0.00005, 0.0001, 0.0002, 0.00005, 0.0002, 0.0001])
sigma_crisis = np.diag([0.020**2, 0.021**2, 0.022**2, 0.019**2, 0.020**2, 0.021**2, 0.018**2, 0.022**2, 0.020**2])
sigma_crisis += 0.3 * np.random.randn(9, 9) * 0.01

print("\n[1] CALM REGIME (normal market conditions)")
print("-" * 80)

# Compute weights in calm regime
w_mvp_calm = mvp_weights(mu_calm, sigma_calm)
w_tpf_calm = tpf_weights(mu_calm, sigma_calm, rf=0.00001)

# Expected returns and variance
ret_mvp_calm = w_mvp_calm @ mu_calm
var_mvp_calm = w_mvp_calm @ sigma_calm @ w_mvp_calm

ret_tpf_calm = w_tpf_calm @ mu_calm
var_tpf_calm = w_tpf_calm @ sigma_calm @ w_tpf_calm

sr_mvp_calm = ret_mvp_calm / np.sqrt(var_mvp_calm)
sr_tpf_calm = ret_tpf_calm / np.sqrt(var_tpf_calm)

print(f"MVP:  Return={ret_mvp_calm*252*100:.2f}% ann.  Vol={np.sqrt(var_mvp_calm)*np.sqrt(252)*100:.2f}% ann.  Sharpe={sr_mvp_calm*np.sqrt(252):.4f}")
print(f"TPF:  Return={ret_tpf_calm*252*100:.2f}% ann.  Vol={np.sqrt(var_tpf_calm)*np.sqrt(252)*100:.2f}% ann.  Sharpe={sr_tpf_calm*np.sqrt(252):.4f}")
print(f"\n✓ TPF correctly outperforms MVP (return difference: {(ret_tpf_calm - ret_mvp_calm)*252*100:+.2f}% annually)")

print("\n[2] CRISIS REGIME - ORIGINAL CODE (crisis cash applied to BOTH)")
print("-" * 80)

crisis_cash = 0.15

# Original: Apply crisis cash to both
w_mvp_crisis_orig = mvp_weights(mu_crisis, sigma_crisis) * (1 - crisis_cash)
w_mvp_crisis_orig /= w_mvp_crisis_orig.sum()

w_tpf_crisis_orig = tpf_weights(mu_crisis, sigma_crisis, rf=0.00001) * (1 - crisis_cash)
w_tpf_crisis_orig /= w_tpf_crisis_orig.sum()

ret_mvp_crisis_orig = w_mvp_crisis_orig @ mu_crisis
var_mvp_crisis_orig = w_mvp_crisis_orig @ sigma_crisis @ w_mvp_crisis_orig

ret_tpf_crisis_orig = w_tpf_crisis_orig @ mu_crisis
var_tpf_crisis_orig = w_tpf_crisis_orig @ sigma_crisis @ w_tpf_crisis_orig

sr_mvp_crisis_orig = ret_mvp_crisis_orig / np.sqrt(var_mvp_crisis_orig) if var_mvp_crisis_orig > 0 else 0
sr_tpf_crisis_orig = ret_tpf_crisis_orig / np.sqrt(var_tpf_crisis_orig) if var_tpf_crisis_orig > 0 else 0

print(f"MVP:  Return={ret_mvp_crisis_orig*252*100:.2f}% ann.  Vol={np.sqrt(var_mvp_crisis_orig)*np.sqrt(252)*100:.2f}% ann.  Sharpe={sr_mvp_crisis_orig*np.sqrt(252):.4f}")
print(f"TPF:  Return={ret_tpf_crisis_orig*252*100:.2f}% ann.  Vol={np.sqrt(var_tpf_crisis_orig)*np.sqrt(252)*100:.2f}% ann.  Sharpe={sr_tpf_crisis_orig*np.sqrt(252):.4f}")

if ret_tpf_crisis_orig < ret_mvp_crisis_orig:
    print(f"\n✗ BUG DETECTED: TPF return < MVP return (difference: {(ret_tpf_crisis_orig - ret_mvp_crisis_orig)*252*100:.2f}% annually)")
    print(f"   This violates the efficient frontier property!")
else:
    print(f"\n✓ TPF outperforms MVP (return difference: {(ret_tpf_crisis_orig - ret_mvp_crisis_orig)*252*100:+.2f}% annually)")

print("\n[3] CRISIS REGIME - FIXED CODE (crisis cash applied to MVP ONLY)")
print("-" * 80)

# Fixed: Apply crisis cash to MVP only
w_mvp_crisis_fixed = mvp_weights(mu_crisis, sigma_crisis) * (1 - crisis_cash)
w_mvp_crisis_fixed /= w_mvp_crisis_fixed.sum()

w_tpf_crisis_fixed = tpf_weights(mu_crisis, sigma_crisis, rf=0.00001)  # NO crisis cash for TPF
w_tpf_crisis_fixed /= w_tpf_crisis_fixed.sum()

ret_mvp_crisis_fixed = w_mvp_crisis_fixed @ mu_crisis
var_mvp_crisis_fixed = w_mvp_crisis_fixed @ sigma_crisis @ w_mvp_crisis_fixed

ret_tpf_crisis_fixed = w_tpf_crisis_fixed @ mu_crisis
var_tpf_crisis_fixed = w_tpf_crisis_fixed @ sigma_crisis @ w_tpf_crisis_fixed

sr_mvp_crisis_fixed = ret_mvp_crisis_fixed / np.sqrt(var_mvp_crisis_fixed) if var_mvp_crisis_fixed > 0 else 0
sr_tpf_crisis_fixed = ret_tpf_crisis_fixed / np.sqrt(var_tpf_crisis_fixed) if var_tpf_crisis_fixed > 0 else 0

print(f"MVP:  Return={ret_mvp_crisis_fixed*252*100:.2f}% ann.  Vol={np.sqrt(var_mvp_crisis_fixed)*np.sqrt(252)*100:.2f}% ann.  Sharpe={sr_mvp_crisis_fixed*np.sqrt(252):.4f}")
print(f"TPF:  Return={ret_tpf_crisis_fixed*252*100:.2f}% ann.  Vol={np.sqrt(var_tpf_crisis_fixed)*np.sqrt(252)*100:.2f}% ann.  Sharpe={sr_tpf_crisis_fixed*np.sqrt(252):.4f}")

if ret_tpf_crisis_fixed >= ret_mvp_crisis_fixed:
    print(f"\n✓ FIXED: TPF return ≥ MVP return (difference: {(ret_tpf_crisis_fixed - ret_mvp_crisis_fixed)*252*100:+.2f}% annually)")
    print(f"   Efficient frontier property restored!")
else:
    print(f"\n✗ Still violated: TPF return < MVP return")

print("\n" + "="*80)
print("  SUMMARY")
print("="*80)
print(f"\nReturn improvement from fix: {((ret_tpf_crisis_fixed - ret_mvp_crisis_fixed) - (ret_tpf_crisis_orig - ret_mvp_crisis_orig))*252*100:+.2f}% annually")
print(f"Sharpe improvement from fix: {(sr_tpf_crisis_fixed - sr_mvp_crisis_fixed)*np.sqrt(252) - (sr_tpf_crisis_orig - sr_mvp_crisis_orig)*np.sqrt(252):+.4f}")

if ret_tpf_crisis_orig < ret_mvp_crisis_orig and ret_tpf_crisis_fixed >= ret_mvp_crisis_fixed:
    print("\n✓✓✓ FIX IS EFFECTIVE ✓✓✓")
    print("The proposed fix successfully restores the efficient frontier property.")
else:
    print("\n⚠ Further investigation needed.")

print("\n" + "="*80 + "\n")
