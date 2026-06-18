"""
config.py
---------
Central configuration for the Market Regimes Comparative Study.
All parameters, tickers, thresholds, and strategy settings live here.
"""

import numpy as np

# ─────────────────────────────────────────────
# DATE RANGE
# ─────────────────────────────────────────────
START_DATE = "2004-01-01"
END_DATE   = "2025-12-31"

# Walk-forward refit frequency (trading days)
REFIT_FREQ = 252          # refit every ~1 year
MIN_TRAIN_DAYS = 504      # minimum 2 years of history before first prediction

# ─────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLY", "XLF", "XLI", "XLB", "XLE", "XLP", "XLV", "XLU"]
SECTOR_NAMES = {
    "XLK": "Technology",
    "XLY": "Consumer Discretionary",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLE": "Energy",
    "XLP": "Consumer Staples",
    "XLV": "Health Care",
    "XLU": "Utilities",
}
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
RF_TICKER  = "^IRX"        # 13-week T-Bill annualised yield (%)

N_ASSETS = len(SECTOR_ETFS)
TRADING_DAYS_PER_YEAR = 252

# ─────────────────────────────────────────────
# VIX REGIME THRESHOLDS
# ─────────────────────────────────────────────
# Literature-standard fixed thresholds (preferred over percentile-based cuts):
#   VIX < 20  — Calm:         below long-run VIX average; "complacency" zone
#                              (Whaley 2009 JPM; CBOE; S&P Global)
#   20 ≤ VIX < 30 — Transitional: elevated uncertainty, active hedging begins
#                              (Bloom 2009 NBER; Ang & Bekaert 2004 RFS;
#                               Amundi Institute; DeMiguel et al. 2009 RFS)
#   VIX ≥ 30  — Crisis:       significant fear / market dislocation; well above
#                              historical crisis entry points (2002, 2008, 2020)
#                              (Whaley 2009; St. Louis Fed; ECB working papers)
#
# Previous percentile-based values (17.8 / 23.1) lacked theoretical grounding
# and set the crisis threshold too low — VIX of 23 is normal elevated vol,
# not a crisis.  VIX=30 is the standard practitioner and academic cut.
VIX_CALM_THRESHOLD        = 20.0   # Calm → Transitional
VIX_TRANSITIONAL_THRESHOLD = 30.0  # Transitional → Crisis
# Regime integer codes
REGIME_CALM        = 0
REGIME_TRANSITIONAL = 1
REGIME_CRISIS      = 2
REGIME_LABELS = {0: "Calm", 1: "Transitional", 2: "Crisis"}
REGIME_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

# ─────────────────────────────────────────────
# HMM SETTINGS
# ─────────────────────────────────────────────
HMM_N_STATES        = 3
HMM_N_ITER          = 200
HMM_N_INIT          = 20       # random restarts; keep best BIC
HMM_COVARIANCE_TYPE = "full"   # multivariate Gaussian per state
HMM_RANDOM_STATE    = 42

# ─────────────────────────────────────────────
# GMM SETTINGS (Step 1 of ML Pipeline)
# ─────────────────────────────────────────────
GMM_N_COMPONENTS   = 3
GMM_N_INIT         = 20
GMM_COVARIANCE_TYPE = "full"
GMM_RANDOM_STATE   = 42

# ─────────────────────────────────────────────
# RANDOM FOREST SETTINGS (Step 2 of ML Pipeline)
# ─────────────────────────────────────────────
ML_FORECAST_MODEL = "extra_trees"  # random_forest, extra_trees, hist_gradient_boosting
RF_N_ESTIMATORS  = 300
RF_MAX_DEPTH     = 6
RF_MIN_SAMPLES   = 15
RF_MAX_FEATURES  = 0.7
GB_LEARNING_RATE = 0.05
GB_L2_REGULARIZATION = 0.01
RF_RANDOM_STATE  = 42
RF_N_JOBS        = 1
RF_LAG_DAYS      = [1, 2, 3, 5]   # lagged return features

# ─────────────────────────────────────────────
# PORTFOLIO OPTIMISATION
# ─────────────────────────────────────────────
ROLLING_WINDOW     = 252          # days for unconditional rolling estimates
MIN_WEIGHT         = 0.0          # long-only
MAX_WEIGHT         = 1.0          # no single-asset cap (optimizer free)
WEIGHT_TOL         = 1e-6
REBALANCE_FREQ     = 21           # monthly rebalance (~21 trading days)

# Shrinkage
USE_LEDOIT_WOLF = True

# ─────────────────────────────────────────────
# STATIC REGIME TARGET WEIGHTS  (Section 5)
# Used for economic context / interpretation only
# ─────────────────────────────────────────────
STATIC_WEIGHTS = {
    REGIME_CALM: {
        "XLK": 0.30, "XLY": 0.25, "XLF": 0.20,
        "XLI": 0.15, "XLP": 0.10,
        "XLB": 0.00, "XLE": 0.00, "XLV": 0.00, "XLU": 0.00,
    },
    REGIME_TRANSITIONAL: {
        "XLK": 0.15, "XLF": 0.15, "XLI": 0.15,
        "XLP": 0.20, "XLV": 0.20, "XLU": 0.15,
        "XLY": 0.00, "XLB": 0.00, "XLE": 0.00,
    },
    REGIME_CRISIS: {
        "XLP": 0.30, "XLV": 0.30, "XLU": 0.25,
        "XLK": 0.00, "XLY": 0.00, "XLF": 0.00,
        "XLI": 0.00, "XLB": 0.00, "XLE": 0.00,
        # 15% to cash — handled by scaling weights to 0.85 and adding rf component
    },
}
CRISIS_CASH_FRACTION = 0.15   # 15% held in risk-free asset during crisis

# ─────────────────────────────────────────────
# PERFORMANCE & BENCHMARKING
# ─────────────────────────────────────────────
TRANSACTION_COST_BPS = 10      # baseline one-way transaction cost (10 bps)
BREAKEVEN_COST_RANGE = np.arange(0, 51, 1)   # 0–50 bps for break-even sweep
RISK_FREE_ANNUAL     = 0.02    # fallback if RF data unavailable

# ─────────────────────────────────────────────
# STRATEGY NAMES (used in tables and charts)
# ─────────────────────────────────────────────
STRATEGIES = [
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

# ─────────────────────────────────────────────
# FRED SERIES
# ─────────────────────────────────────────────
FRED_TED_SPREAD  = "TEDRATE"    # TED Spread (discontinued 2023; we extend manually)
FRED_TERM_SPREAD = "T10Y2Y"     # 10Y-2Y Treasury term spread

# ─────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────
RESULTS_DIR = "results"
DATA_CACHE  = "results/data_cache.pkl"
