"""
visualization/plots.py
----------------------
Publication-quality charts for the seminar paper.
All figures use a consistent dark academic theme.

Figures generated:
  01_regime_timeline.png            VIX time series with coloured regime bands
  02_regime_comparison_heatmap.png  VIX / HMM / ML regime assignment comparison
  03_transition_matrix.png          HMM transition probability heatmap
  04_return_distributions.png       Regime-conditional return density plots
  05_capm_betas.png                 State-conditional CAPM betas (grouped bar)
  06_portfolio_weights.png          Stacked area charts of dynamic weights
  07_cumulative_wealth.png          Cumulative wealth curves — all strategies
  08_performance_table.png          Formatted KPI table
  09_drawdown_profiles.png          Underwater (drawdown) plot
  10_breakeven_costs.png            SR vs. transaction cost sweep
  11_feature_importances.png        RF feature importances per regime
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "border":   "#30363D",
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
    "calm":     "#2ECC71",
    "trans":    "#F39C12",
    "crisis":   "#E74C3C",
    "accent":   "#58A6FF",
}

STRATEGY_COLORS = {
    "SPY B&H":    "#8B949E",
    "EW 1/N":     "#6E7681",
    "Static MVP":  "#58A6FF",
    "Static TPF":  "#1F6FEB",
    "VIX-MVP":    "#2ECC71",
    "VIX-TPF":    "#27AE60",
    "HMM-MVP":    "#F39C12",
    "HMM-TPF":    "#E67E22",
    "ML-MVP":     "#E74C3C",
    "ML-TPF":     "#C0392B",
}

SECTOR_COLORS = [
    "#58A6FF", "#2ECC71", "#F39C12", "#E74C3C", "#A855F7",
    "#EC4899", "#06B6D4", "#84CC16", "#FB923C",
]

SECTOR_NAMES = {
    "XLK": "Technology",
    "XLY": "Consumer Disc.",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLE": "Energy",
    "XLP": "Consumer Staples",
    "XLV": "Health Care",
    "XLU": "Utilities",
}

def get_sector_name(ticker):
    if ticker in SECTOR_NAMES:
        return SECTOR_NAMES[ticker]
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        name = t.info.get("shortName") or t.info.get("longName") or ticker
        if "Select Sector" in name:
            name = name.split("Select Sector")[0].replace("The", "").replace("State Street", "").strip()
        SECTOR_NAMES[ticker] = name
        return name
    except Exception:
        return ticker

def _apply_theme(fig, axes_list=None):
    fig.patch.set_facecolor(PALETTE["bg"])
    if axes_list is not None:
        for ax in axes_list:
            ax.set_facecolor(PALETTE["panel"])
            ax.tick_params(colors=PALETTE["text"], labelsize=8)
            ax.xaxis.label.set_color(PALETTE["text"])
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.title.set_color(PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
            ax.grid(True, color=PALETTE["border"], alpha=0.5, linewidth=0.5)


def _save(fig, path: str, dpi: int = 180):
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def _add_regime_bands(ax, dates, regime_labels, alpha=0.15):
    """Shade background with regime colours."""
    colors = {0: PALETTE["calm"], 1: PALETTE["trans"], 2: PALETTE["crisis"]}
    if len(dates) != len(regime_labels):
        return
    prev_regime = None
    start_date  = None
    for d, r in zip(dates, regime_labels):
        if r != prev_regime:
            if prev_regime is not None:
                ax.axvspan(start_date, d, color=colors[prev_regime], alpha=alpha)
            start_date  = d
            prev_regime = r
    if prev_regime is not None:
        ax.axvspan(start_date, dates[-1], color=colors[prev_regime], alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — Regime Timeline
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_timeline(
    vix: pd.Series,
    regime_labels: pd.Series,
    out_dir: str,
    calm_thr: float = 20.0,
    trans_thr: float = 30.0,
):
    fig, ax = plt.subplots(figsize=(14, 5))
    _apply_theme(fig, [ax])

    common = vix.index.intersection(regime_labels.index)
    vix_   = vix.loc[common]
    regs_  = regime_labels.loc[common]

    _add_regime_bands(ax, common.to_list(), regs_.values, alpha=0.18)
    ax.plot(common, vix_.values, color=PALETTE["accent"], lw=1.2, zorder=5)
    ax.axhline(calm_thr,  color=PALETTE["calm"],   ls="--", lw=1.0, alpha=0.8,
               label=f"Calm threshold ({calm_thr})")
    ax.axhline(trans_thr, color=PALETTE["crisis"],  ls="--", lw=1.0, alpha=0.8,
               label=f"Crisis threshold ({trans_thr})")

    patches = [
        mpatches.Patch(color=PALETTE["calm"],   alpha=0.5, label="Calm"),
        mpatches.Patch(color=PALETTE["trans"],  alpha=0.5, label="Transitional"),
        mpatches.Patch(color=PALETTE["crisis"], alpha=0.5, label="Crisis"),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0],[0], color=PALETTE["accent"], lw=1.2, label="VIX"),
    ], loc="upper left", facecolor=PALETTE["panel"],
       labelcolor=PALETTE["text"], fontsize=8, framealpha=0.8)

    ax.set_title("Figure 1 — US Market Regime Timeline: VIX-Defined Regimes (2004–2025)",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], pad=12)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("VIX Level", fontsize=9)
    ax.set_xlim(common[0], common[-1])

    _save(fig, os.path.join(out_dir, "01_regime_timeline.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — Regime Comparison Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_comparison(
    vix_regimes: pd.Series,
    hmm_regimes: pd.Series,
    ml_regimes:  pd.Series,
    out_dir: str,
):
    common = (
        vix_regimes.index
        .intersection(hmm_regimes.index)
        .intersection(ml_regimes.index)
        .sort_values()
    )
    df = pd.DataFrame({
        "VIX Rules": vix_regimes.loc[common].values,
        "HMM":       hmm_regimes.loc[common].values,
        "GMM (ML)":  ml_regimes.loc[common].values,
    }, index=common).T

    # Subsample monthly for readability
    monthly_idx = pd.date_range(common[0], common[-1], freq="MS")
    monthly_idx = monthly_idx[monthly_idx.isin(df.columns)]
    df_plot = df[monthly_idx] if len(monthly_idx) > 0 else df.iloc[:, ::21]

    fig, ax = plt.subplots(figsize=(16, 3))
    _apply_theme(fig, [ax])

    cmap = matplotlib.colors.ListedColormap([PALETTE["calm"], PALETTE["trans"], PALETTE["crisis"]])
    im = ax.imshow(df_plot.values, aspect="auto", cmap=cmap,
                   vmin=0, vmax=2, interpolation="nearest")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["VIX Rules", "HMM", "GMM (ML)"],
                       color=PALETTE["text"], fontsize=9)
    n_ticks = min(10, len(df_plot.columns))
    tick_pos = np.linspace(0, len(df_plot.columns)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        [df_plot.columns[i].strftime("%Y-%m") for i in tick_pos],
        rotation=35, ha="right", color=PALETTE["text"], fontsize=7
    )

    patches = [
        mpatches.Patch(color=PALETTE["calm"],   label="Calm"),
        mpatches.Patch(color=PALETTE["trans"],  label="Transitional"),
        mpatches.Patch(color=PALETTE["crisis"], label="Crisis"),
    ]
    ax.legend(handles=patches, loc="lower right", facecolor=PALETTE["panel"],
              labelcolor=PALETTE["text"], fontsize=8, framealpha=0.8)
    ax.set_title("Figure 2 — Regime Agreement: VIX Rules vs. HMM vs. GMM (ML)",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], pad=12)
    _save(fig, os.path.join(out_dir, "02_regime_comparison_heatmap.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — HMM Transition Matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_transition_matrix(
    trans_matrix: np.ndarray,
    out_dir: str,
):
    labels = ["Calm", "Transitional", "Crisis"]
    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_theme(fig, [ax])

    cmap = sns.color_palette("Blues", as_cmap=True)
    im = ax.imshow(trans_matrix, cmap=cmap, vmin=0, vmax=1)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=PALETTE["text"])
    cb.ax.yaxis.set_tick_params(color=PALETTE["text"])
    cb.outline.set_edgecolor(PALETTE["border"])

    for i in range(3):
        for j in range(3):
            val = trans_matrix[i, j]
            color = "white" if val > 0.6 else PALETTE["text"]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=9)
    ax.set_yticklabels(labels, color=PALETTE["text"], fontsize=9)
    ax.set_xlabel("To State", fontsize=9)
    ax.set_ylabel("From State", fontsize=9)
    ax.set_title("Figure 3 — HMM Transition Probability Matrix",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], pad=12)
    _save(fig, os.path.join(out_dir, "03_transition_matrix.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Regime-Conditional Return Distributions
# ─────────────────────────────────────────────────────────────────────────────
def plot_return_distributions(
    log_returns:   pd.DataFrame,
    regime_labels: pd.Series,
    out_dir: str,
    n_assets: int = 3,
):
    """Plot KDE of daily returns for top 3 sectors per regime."""
    common = log_returns.index.intersection(regime_labels.index)
    lr = log_returns.loc[common]
    rg = regime_labels.reindex(common).ffill()

    regime_info = {
        0: ("Calm",         PALETTE["calm"]),
        1: ("Transitional", PALETTE["trans"]),
        2: ("Crisis",       PALETTE["crisis"]),
    }
    show_tickers = ["XLK", "XLP", "XLF"]
    show_tickers = [t for t in show_tickers if t in lr.columns]
    if not show_tickers:
        show_tickers = lr.columns[:3].tolist()

    fig, axes = plt.subplots(1, len(show_tickers), figsize=(14, 4), sharey=False)
    _apply_theme(fig, list(axes))

    for ax_idx, ticker in enumerate(show_tickers):
        ax = axes[ax_idx]
        for code, (name, color) in regime_info.items():
            mask = rg == code
            sub  = lr.loc[mask, ticker].dropna()
            if len(sub) < 10:
                continue
            sub.plot.kde(ax=ax, color=color, lw=1.8, label=name, bw_method=0.3)
        ax.axvline(0, color=PALETTE["subtext"], lw=0.8, ls="--")
        sector_name = get_sector_name(ticker)
        ax.set_title(f"{ticker}\n{sector_name}", fontsize=10, fontweight="bold",
                     color=PALETTE["text"])
        ax.set_xlabel("Daily Log-Return", fontsize=8)
        ax.set_ylabel("Density" if ax_idx == 0 else "", fontsize=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=7, framealpha=0.8)
        ax.set_xlim(-0.12, 0.12)

    fig.suptitle(
        "Figure 4 — Regime-Conditional Return Distributions (KDE)",
        fontsize=11, fontweight="bold", color=PALETTE["text"], y=1.02
    )
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "04_return_distributions.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5 — Regime-Conditional CAPM Betas
# ─────────────────────────────────────────────────────────────────────────────
def plot_capm_betas(
    beta_summary: pd.DataFrame,
    out_dir: str,
):
    """
    beta_summary: rows=tickers, cols like Calm_beta, Transitional_beta, Crisis_beta
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_theme(fig, [ax])

    tickers = beta_summary.index.tolist()
    x = np.arange(len(tickers))
    width = 0.25

    regimes = [("Calm_beta", PALETTE["calm"]), ("Transitional_beta", PALETTE["trans"]),
               ("Crisis_beta", PALETTE["crisis"])]
    offsets = [-width, 0, width]

    for (col, color), off in zip(regimes, offsets):
        if col in beta_summary.columns:
            vals = beta_summary[col].values.astype(float)
            bars = ax.bar(x + off, vals, width, color=color, alpha=0.85,
                          label=col.replace("_beta", ""),
                          edgecolor=PALETTE["bg"], linewidth=0.5)

    ax.axhline(1.0, color=PALETTE["subtext"], ls="--", lw=1.0, label="β = 1.0")
    ax.axhline(0.0, color=PALETTE["border"],  ls="-",  lw=0.6)

    ax.set_xticks(x)
    x_labels = [f"{t}\n{get_sector_name(t)}" for t in tickers]
    ax.set_xticklabels(x_labels, rotation=0, ha="center",
                       color=PALETTE["text"], fontsize=7.5)
    ax.set_ylabel("CAPM Beta (β)", fontsize=9)
    ax.set_title("Figure 5 — Regime-Conditional CAPM Betas: 9 Sector ETFs × 3 States",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], pad=12)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=8, framealpha=0.8)
    ax.set_ylim(-0.2, 2.0)
    _save(fig, os.path.join(out_dir, "05_capm_betas.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 6 — Dynamic Portfolio Weights
# ─────────────────────────────────────────────────────────────────────────────
def plot_portfolio_weights(
    weight_history: dict,   # strategy_name → pd.DataFrame(T, N_assets)
    out_dir: str,
    strategies_to_plot: list = None,
):
    if strategies_to_plot is None:
        strategies_to_plot = ["HMM-MVP", "ML-MVP"]

    avail = [s for s in strategies_to_plot if s in weight_history]
    if not avail:
        print("  [WARN] No weight history to plot.")
        return

    n = len(avail)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    _apply_theme(fig, axes)

    for ax, strat in zip(axes, avail):
        wdf = weight_history[strat].dropna(how="all").fillna(0)
        if wdf.empty:
            continue
        tickers = wdf.columns.tolist()
        colors  = SECTOR_COLORS[:len(tickers)]
        legend_labels = [f"{t} ({get_sector_name(t)})" for t in tickers]
        wdf.plot.area(ax=ax, stacked=True, color=colors, alpha=0.85, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Portfolio Weight", fontsize=8)
        ax.set_title(f"{strat} — Dynamic Sector Allocation",
                     fontsize=9, fontweight="bold", color=PALETTE["text"])
        ax.legend(legend_labels, loc="upper left", facecolor=PALETTE["panel"],
                  labelcolor=PALETTE["text"], fontsize=6, ncol=3, framealpha=0.8)

    fig.suptitle("Figure 6 — Dynamic Portfolio Weights Over Time",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "06_portfolio_weights.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 7 — Cumulative Wealth Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_cumulative_wealth(
    wealth_df: pd.DataFrame,
    regime_labels: pd.Series,
    out_dir: str,
):
    fig, ax = plt.subplots(figsize=(14, 6))
    _apply_theme(fig, [ax])

    common = wealth_df.index.intersection(regime_labels.index)
    if len(common) > 0:
        _add_regime_bands(ax, common.to_list(),
                          regime_labels.loc[common].values, alpha=0.10)

    for col in wealth_df.columns:
        color = STRATEGY_COLORS.get(col, PALETTE["accent"])
        lw = 2.0 if col in ("ML-TPF", "HMM-TPF", "SPY B&H") else 1.2
        ls = "--" if "B&H" in col or "1/N" in col else "-"
        ax.plot(wealth_df.index, wealth_df[col], color=color, lw=lw,
                ls=ls, label=col, zorder=5)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"${x:.1f}"
    ))
    ax.set_ylabel("Cumulative Wealth ($)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_title("Figure 7 — Cumulative Wealth (Log Scale) — All Strategies vs. Benchmarks",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], pad=12)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=7, ncol=2, framealpha=0.8, loc="upper left")
    ax.set_xlim(wealth_df.index[0], wealth_df.index[-1])
    _save(fig, os.path.join(out_dir, "07_cumulative_wealth.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 8 — Performance Table
# ─────────────────────────────────────────────────────────────────────────────
def plot_performance_table(perf_df: pd.DataFrame, out_dir: str):
    cols_show = ["Ann.Return (%)", "Ann.Vol (%)", "Sharpe", "Sortino",
                 "Max DD (%)", "Jensen α (%)", "Beta", "Calmar", "Breakeven (bps)"]
    cols_show = [c for c in cols_show if c in perf_df.columns]
    df_plot = perf_df[cols_show].copy()
    df_plot = df_plot.round(3)

    fig, ax = plt.subplots(figsize=(16, 4))
    _apply_theme(fig, [ax])
    ax.axis("off")

    col_labels = ["Strategy"] + cols_show
    cell_data  = [[idx] + [f"{v:.3f}" if pd.notna(v) else "—"
                            for v in row] for idx, row in df_plot.iterrows()]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PALETTE["panel"] if row > 0 else PALETTE["border"])
        cell.set_text_props(color=PALETTE["text"])
        cell.set_edgecolor(PALETTE["bg"])
        if row == 0:
            cell.set_text_props(fontweight="bold")
        # Highlight best Sharpe
        if row > 0 and col == col_labels.index("Sharpe") if "Sharpe" in col_labels else -1:
            pass

    ax.set_title("Figure 8 — Strategy Performance Summary Table",
                 fontsize=11, fontweight="bold", color=PALETTE["text"],
                 pad=20, loc="center")
    _save(fig, os.path.join(out_dir, "08_performance_table.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 9 — Drawdown Profiles
# ─────────────────────────────────────────────────────────────────────────────
def plot_drawdown_profiles(
    wealth_df: pd.DataFrame,
    out_dir: str,
    top_n: int = 6,
):
    strategies = ["SPY B&H", "EW 1/N", "HMM-MVP", "HMM-TPF", "ML-MVP", "ML-TPF"]
    available  = [s for s in strategies if s in wealth_df.columns]

    fig, ax = plt.subplots(figsize=(14, 5))
    _apply_theme(fig, [ax])

    for strat in available:
        w   = wealth_df[strat].dropna()
        dd  = (w - w.cummax()) / w.cummax()
        color = STRATEGY_COLORS.get(strat, PALETTE["accent"])
        lw    = 2.0 if strat in ("ML-TPF", "SPY B&H") else 1.2
        ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25)
        ax.plot(dd.index, dd.values, color=color, lw=lw, label=strat)

    ax.set_ylabel("Drawdown (%)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("Figure 9 — Drawdown (Underwater) Profiles — Selected Strategies",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], pad=12)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=8, framealpha=0.8)
    ax.set_xlim(wealth_df.index[0], wealth_df.index[-1])
    _save(fig, os.path.join(out_dir, "09_drawdown_profiles.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 10 — Break-Even Transaction Costs
# ─────────────────────────────────────────────────────────────────────────────
def plot_breakeven_costs(
    port_returns: pd.DataFrame,
    spy_sr: float,
    cost_range_bps: np.ndarray,
    out_dir: str,
    annual_turnover_dict: dict = None,
):
    """
    Plot Sharpe ratio vs. one-way transaction cost for active strategies.
    """
    active = ["Static MVP", "VIX-MVP", "VIX-TPF", "HMM-MVP", "HMM-TPF", "ML-MVP", "ML-TPF"]
    active = [s for s in active if s in port_returns.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_theme(fig, [ax])

    for strat in active:
        s = port_returns[strat].dropna()
        mu_daily    = float(s.mean())
        sigma_daily = float(s.std())
        to = (annual_turnover_dict or {}).get(strat, 0.5)
        daily_to = to / 252.0

        sr_sweep = []
        for bps in cost_range_bps:
            cost_per_day = bps / 10_000.0 * daily_to
            adj_mu = mu_daily - cost_per_day
            sr = adj_mu / sigma_daily * np.sqrt(252) if sigma_daily > 0 else np.nan
            sr_sweep.append(sr)

        color = STRATEGY_COLORS.get(strat, PALETTE["accent"])
        ax.plot(cost_range_bps, sr_sweep, color=color, lw=1.8, label=strat)

    ax.axhline(spy_sr, color=PALETTE["subtext"], ls="--", lw=1.5,
               label=f"SPY Sharpe ({spy_sr:.2f})")
    ax.axhline(0, color=PALETTE["border"], lw=0.6)
    ax.set_xlabel("One-Way Transaction Cost (bps)", fontsize=9)
    ax.set_ylabel("Adjusted Sharpe Ratio", fontsize=9)
    ax.set_title("Figure 10 — Break-Even Transaction Cost Analysis",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], pad=12)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=8, framealpha=0.8)
    ax.set_xlim(cost_range_bps[0], cost_range_bps[-1])
    _save(fig, os.path.join(out_dir, "10_breakeven_costs.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 11 — RF Feature Importances
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importances(
    feature_importances: dict,   # regime_code → np.ndarray
    feature_names: list,
    out_dir: str,
):
    regime_names = {0: "Calm", 1: "Transitional", 2: "Crisis"}
    regime_colors = {0: PALETTE["calm"], 1: PALETTE["trans"], 2: PALETTE["crisis"]}

    available = {k: v for k, v in feature_importances.items() if v is not None}
    if not available:
        print("  [WARN] No feature importances to plot.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    _apply_theme(fig, axes)

    for ax, (code, importances) in zip(axes, available.items()):
        n_feat = len(importances)
        names  = feature_names[:n_feat] if len(feature_names) >= n_feat else feature_names
        # Average across output columns if 2D
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
        importances = importances[:len(names)]

        sorted_idx = np.argsort(importances)[::-1]
        colors_bar = [regime_colors[code]] * len(names)
        ax.barh(range(len(names)), importances[sorted_idx],
                color=colors_bar, alpha=0.85, edgecolor=PALETTE["bg"])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[i] for i in sorted_idx],
                           color=PALETTE["text"], fontsize=7)
        ax.set_xlabel("Feature Importance", fontsize=8)
        ax.set_title(f"{regime_names[code]} Regime",
                     fontsize=9, fontweight="bold", color=PALETTE["text"])

    fig.suptitle("Figure 11 — Random Forest Feature Importances per Regime",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "11_feature_importances.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER PLOT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_figures(
    data:            dict,       # output from main pipeline
    out_dir:         str = "results",
    cost_range_bps:  np.ndarray = None,
):
    """
    Call all 11 plot functions from a single data dictionary.
    Expected keys in `data`:
      vix, vix_regimes, hmm_regimes, ml_regimes,
      hmm_model, log_returns, excess_returns, spy_excess,
      wealth, port_returns, rf_daily, beta_summary,
      weight_history, ml_pipeline, rf_feature_names
    """
    if cost_range_bps is None:
        cost_range_bps = np.arange(0, 51, 1)

    print("\n[Visualization] Generating figures …")

    # 1
    plot_regime_timeline(
        data["vix"], data["vix_regimes"], out_dir
    )

    # 2
    plot_regime_comparison(
        data["vix_regimes"], data["hmm_regimes"], data["ml_regimes"], out_dir
    )

    # 3
    if data.get("hmm_model") and hasattr(data["hmm_model"], "transition_matrix"):
        plot_transition_matrix(data["hmm_model"].transition_matrix, out_dir)

    # 4
    plot_return_distributions(
        data["log_returns"], data["vix_regimes"], out_dir
    )

    # 5
    if data.get("beta_summary") is not None:
        plot_capm_betas(data["beta_summary"], out_dir)

    # 6
    if data.get("weight_history"):
        plot_portfolio_weights(data["weight_history"], out_dir)

    # 7
    plot_cumulative_wealth(
        data["wealth"], data["vix_regimes"], out_dir
    )

    # 8
    if data.get("perf_table") is not None:
        plot_performance_table(data["perf_table"], out_dir)

    # 9
    plot_drawdown_profiles(data["wealth"], out_dir)

    # 10 — break-even sweeps total cost, so use gross (pre-cost) returns
    spy_sr = data.get("spy_sr", 0.5)
    breakeven_returns = data.get("gross_returns")
    if breakeven_returns is None:
        breakeven_returns = data["port_returns"]
    plot_breakeven_costs(
        breakeven_returns, spy_sr, cost_range_bps, out_dir,
        annual_turnover_dict=data.get("turnover_dict"),
    )

    # 11
    if data.get("ml_pipeline") and data["ml_pipeline"].ensemble_:
        feat_imp = data["ml_pipeline"].ensemble_.feature_importances()
        plot_feature_importances(
            feat_imp,
            data.get("rf_feature_names", [f"f{i}" for i in range(50)]),
            out_dir,
        )

    print(f"[Visualization] All figures saved to: {out_dir}/")
