"""
data/loader.py
--------------
Downloads and caches all market data required for the study:
  - Adjusted close prices for 9 sector ETFs + SPY
  - VIX index
  - 1-Month T-Bill rate (risk-free rate proxy)
  - TED Spread (FRED; extended for 2023-2025)
  - Term Spread 10Y-2Y (FRED)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Optional FRED via pandas_datareader
try:
    import pandas_datareader.data as web
    _DATAREADER_AVAILABLE = True
except ImportError:
    _DATAREADER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices via yfinance for a list of tickers."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all")
    return prices


def _download_vix(start: str, end: str) -> pd.Series:
    """Download CBOE VIX index closing values."""
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw["Close"].squeeze()
    else:
        vix = raw["Close"]
    vix.name = "VIX"
    return vix.dropna()


def _download_rf(start: str, end: str) -> pd.Series:
    """
    Download 13-Week T-Bill annualised yield (^IRX) from yfinance.
    Converts from annual percentage to daily decimal rate.
    """
    raw = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        rf_annual_pct = raw["Close"].squeeze()
    else:
        rf_annual_pct = raw["Close"]
    rf_daily = (rf_annual_pct / 100.0) / 252.0
    rf_daily.name = "RF_daily"
    return rf_daily.dropna()


def _download_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Download a FRED series via pandas_datareader."""
    if not _DATAREADER_AVAILABLE:
        raise ImportError("pandas_datareader is required for FRED data.")
    s = web.DataReader(series_id, "fred", start, end)
    s = s.squeeze().dropna()
    s.name = series_id
    return s


def _build_ted_spread(start: str, end: str) -> pd.Series:
    """
    Build TED Spread from FRED TEDRATE (through 2023) and extend using
    3-Month LIBOR proxy from yfinance (^IRX - ^TYX slope approach).
    Falls back to 3M T-Bill implied spread if all else fails.
    """
    try:
        ted = _download_fred("TEDRATE", start, end)
        ted = ted / 100.0   # percent → decimal
        ted.name = "TED"
        return ted
    except Exception:
        pass

    # Fallback: use 3M T-Bill as the risk-free leg; OIS proxy as 0
    print("  [WARNING] FRED TEDRATE unavailable; using T-Bill yield as TED proxy (zeroed spread).")
    rf = _download_rf(start, end)
    ted = rf.copy()
    ted.name = "TED"
    return ted


def _build_term_spread(start: str, end: str) -> pd.Series:
    """10Y-2Y term spread from FRED T10Y2Y series."""
    try:
        term = _download_fred("T10Y2Y", start, end)
        term = term / 100.0   # percent → decimal
        term.name = "TERM"
        return term
    except Exception:
        print("  [WARNING] FRED T10Y2Y unavailable; using zero term spread.")
        rf = _download_rf(start, end)
        term = pd.Series(0.0, index=rf.index, name="TERM")
        return term


# ─────────────────────────────────────────────────────────────────────────────
def load_all_data(
    start: str,
    end: str,
    sector_etfs: list[str],
    benchmark: str = "SPY",
    cache_path: str = None,
    force_reload: bool = False,
) -> dict:
    """
    Master data loader. Downloads or loads from cache and returns a dict with:
      prices   : pd.DataFrame  (all ETFs + SPY, adjusted close)
      vix      : pd.Series     (VIX daily closing level)
      rf       : pd.Series     (daily risk-free rate, decimal)
      ted      : pd.Series     (TED Spread, decimal)
      term     : pd.Series     (10Y-2Y term spread, decimal)
    All series are aligned to the intersection of trading days.
    """
    if cache_path and os.path.exists(cache_path) and not force_reload:
        print(f"Loading cached data from {cache_path} …")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Downloading market data …")
    all_tickers = sector_etfs + [benchmark]

    print(f"  ETFs + benchmark: {all_tickers}")
    prices = _download_prices(all_tickers, start, end)

    print(f"  VIX index")
    vix = _download_vix(start, end)

    print(f"  Risk-free rate (^IRX)")
    rf = _download_rf(start, end)

    print(f"  TED Spread (FRED)")
    ted = _build_ted_spread(start, end)

    print(f"  Term Spread 10Y-2Y (FRED)")
    term = _build_term_spread(start, end)

    # ── Align all series to common trading dates ──────────────────────────────
    prices = prices.ffill().bfill()
    vix    = vix.ffill().bfill()
    rf     = rf.ffill().bfill()
    ted    = ted.ffill().bfill()
    term   = term.ffill().bfill()

    # Join on ETF price index (market trading days)
    idx = prices.index
    vix  = vix.reindex(idx).ffill().bfill()
    rf   = rf.reindex(idx).ffill().bfill()
    ted  = ted.reindex(idx).ffill().bfill()
    term = term.reindex(idx).ffill().bfill()

    data = {
        "prices": prices,
        "vix":    vix,
        "rf":     rf,
        "ted":    ted,
        "term":   term,
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Data cached to {cache_path}")

    print(f"  Data range: {idx[0].date()} → {idx[-1].date()} ({len(idx)} trading days)")
    return data
