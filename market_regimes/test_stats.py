import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
import config as CFG
from data.loader import load_all_data
from regimes.vix_classifier import classify_vix_regimes, regime_statistics

data_raw = load_all_data(
    start=CFG.START_DATE,
    end=CFG.END_DATE,
    sector_etfs=CFG.SECTOR_ETFS,
    benchmark=CFG.BENCHMARK,
    cache_path=os.path.join(CFG.RESULTS_DIR, "data_cache.pkl"),
    force_reload=False,
)
vix = data_raw["vix"]
prices = data_raw["prices"]
log_ret_all = np.log(prices / prices.shift(1)).dropna()
log_ret_sectors = log_ret_all[[c for c in CFG.SECTOR_ETFS if c in log_ret_all.columns]]

vix_regimes = classify_vix_regimes(
    vix,
    calm_threshold=20,
    transitional_threshold=30,
)
reg_stats = regime_statistics(vix, vix_regimes, log_ret_sectors)

print("VIX Regime Statistics:")
print(reg_stats.round(4))
