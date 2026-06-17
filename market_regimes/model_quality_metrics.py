"""
model_quality_metrics.py
------------------------
Model Quality & Statistical Validation Metrics for the Seminar Paper.

This module provides functions to compute and display all quality metrics
for both the HMM and GMM+RF pipelines. These are designed to be called
from the Jupyter notebook after all models have been fitted.

The module covers:
  1. HMM Quality Metrics
     - BIC/AIC model selection curves (k=2..5)
     - Transition matrix persistence & regime half-lives
     - Regime statistics (annualized mean return, annualized volatility)
     - Stationary (ergodic) distribution
     - Residual diagnostics (ACF of standardized residuals)

  2. GMM + RF Quality Metrics
     - GMM: BIC/AIC model selection curves (k=2..5)
     - GMM: Silhouette scores
     - RF: Out-of-sample classification report (Precision, Recall, F1)
     - RF: Confusion matrix
     - RF: Feature importance breakdown
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from statsmodels.graphics.tsaplots import plot_acf

# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE (matches the notebook's visualization module)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":   "#1a1a2e",
    "text": "#eaeaea",
    "calm": "#2ecc71",
    "trans": "#f39c12",
    "crisis": "#e74c3c",
    "accent1": "#3498db",
    "accent2": "#9b59b6",
    "grid": "#2d2d4e",
}

REGIME_LABELS = {0: "Calm", 1: "Transitional", 2: "Crisis"}
REGIME_COLORS = {0: PALETTE["calm"], 1: PALETTE["trans"], 2: PALETTE["crisis"]}
TRADING_DAYS_PER_YEAR = 252


def _style_figure(fig, ax_or_axes, title=""):
    """Apply dark styling consistent with the notebook."""
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = [ax_or_axes] if not hasattr(ax_or_axes, '__len__') else ax_or_axes
    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["grid"])
        ax.grid(True, alpha=0.3, color=PALETTE["grid"])
    if title:
        fig.suptitle(title, color=PALETTE["text"], fontsize=14, fontweight="bold")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1: HMM QUALITY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def hmm_bic_aic_comparison(
    hmm_feats: pd.DataFrame,
    k_range: range = range(2, 6),
    n_iter: int = 200,
    n_init: int = 10,
    random_state: int = 42,
    results_dir: str = None,
):
    """
    Fit HMMs with k=2..5 states and compute BIC/AIC for model selection.

    Returns a DataFrame with columns: k, BIC, AIC, LogLikelihood.
    Also generates a publication-quality plot.
    """
    X = hmm_feats.values.astype(float)
    T, d = X.shape
    records = []

    for k in k_range:
        best_model = None
        best_score = -np.inf

        rng = np.random.RandomState(random_state)
        seeds = rng.randint(0, 10_000, size=n_init)

        for seed in seeds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=int(seed),
                )
                try:
                    model.fit(X)
                    score = model.score(X)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception:
                    continue

        if best_model is None:
            continue

        log_lik = best_model.score(X) * T  # total log-likelihood
        # Number of free parameters for a Gaussian HMM with full covariance
        n_params = (
            k * k - k              # transition matrix (rows sum to 1)
            + k * d                # means
            + k * d * (d + 1) / 2  # full covariances (symmetric)
            + k - 1                # initial state distribution
        )
        bic = -2 * log_lik + n_params * np.log(T)
        aic = -2 * log_lik + 2 * n_params

        records.append({
            "k": k,
            "LogLikelihood": log_lik,
            "BIC": bic,
            "AIC": aic,
            "n_params": int(n_params),
        })

    df = pd.DataFrame(records)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _style_figure(fig, [ax1, ax2], "HMM — Model Selection: BIC & AIC Curves")

    ax1.plot(df["k"], df["BIC"], "o-", color=PALETTE["accent1"], linewidth=2,
             markersize=8, label="BIC")
    ax1.set_xlabel("Number of States (k)")
    ax1.set_ylabel("BIC Score")
    ax1.set_title("Bayesian Information Criterion")
    ax1.set_xticks(list(k_range))
    ax1.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"])

    ax2.plot(df["k"], df["AIC"], "s-", color=PALETTE["accent2"], linewidth=2,
             markersize=8, label="AIC")
    ax2.set_xlabel("Number of States (k)")
    ax2.set_ylabel("AIC Score")
    ax2.set_title("Akaike Information Criterion")
    ax2.set_xticks(list(k_range))
    ax2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"])

    plt.tight_layout()
    plt.show()
    if results_dir:
        import os
        fig.savefig(os.path.join(results_dir, "hmm_bic_aic.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"], edgecolor="none")
        print(f"  💾 Saved: hmm_bic_aic.png")
    plt.close(fig)

    print("\n📊 HMM Model Selection (BIC / AIC):")
    display(df.round(2))
    return df


def hmm_transition_analysis(hmm_last_model, results_dir: str = None):
    """
    Analyze the HMM transition matrix:
      - Diagonal persistence probabilities
      - Regime half-lives
      - Stationary (ergodic) distribution
    """
    A = hmm_last_model.transition_matrix  # already sorted

    # ── Persistence ──────────────────────────────────────────────────────
    diag = np.diag(A)
    persistence_df = pd.DataFrame({
        "Regime": [REGIME_LABELS[i] for i in range(len(diag))],
        "P(stay)": diag,
        "Half-Life (days)": np.log(0.5) / np.log(diag),
        "Expected Duration (days)": 1 / (1 - diag),
    })

    print("📊 HMM Regime Persistence & Duration:")
    display(persistence_df.round(3))

    # ── Stationary distribution ──────────────────────────────────────────
    # Solve π A = π, sum(π) = 1  →  π (A^T - I) = 0
    n = A.shape[0]
    AT = A.T - np.eye(n)
    AT[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(AT, b)

    ergodic_df = pd.DataFrame({
        "Regime": [REGIME_LABELS[i] for i in range(n)],
        "Stationary Probability": pi,
        "% of Time": pi * 100,
    })

    print("\n📊 HMM Stationary (Ergodic) Distribution:")
    display(ergodic_df.round(4))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _style_figure(fig, [ax1, ax2],
                  "HMM — Regime Persistence & Ergodic Distribution")

    colors = [REGIME_COLORS[i] for i in range(len(diag))]
    labels = [REGIME_LABELS[i] for i in range(len(diag))]

    ax1.bar(labels, diag, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("P(stay in regime)")
    ax1.set_title("Diagonal Persistence Coefficients")
    ax1.set_ylim(0, 1)
    for i, (v, hl) in enumerate(zip(diag, persistence_df["Half-Life (days)"])):
        ax1.text(i, v + 0.02, f"{v:.3f}\n(t½={hl:.1f}d)",
                 ha="center", va="bottom", fontsize=9, color=PALETTE["text"])

    ax2.bar(labels, pi * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("% of Time")
    ax2.set_title("Stationary Distribution")
    for i, v in enumerate(pi * 100):
        ax2.text(i, v + 1, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=10, color=PALETTE["text"])

    plt.tight_layout()
    plt.show()
    if results_dir:
        import os
        fig.savefig(os.path.join(results_dir, "hmm_persistence_ergodic.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"], edgecolor="none")
        print(f"  💾 Saved: hmm_persistence_ergodic.png")
    plt.close(fig)

    return persistence_df, ergodic_df


def hmm_regime_statistics(
    hmm_regimes: pd.Series,
    excess_spy: pd.Series,
    results_dir: str = None,
):
    """
    Regime separation quality: annualized mean return & volatility per regime.
    """
    spy = excess_spy.reindex(hmm_regimes.index).dropna()
    reg = hmm_regimes.reindex(spy.index)

    rows = []
    for r in sorted(reg.unique()):
        mask = reg == r
        ret_r = spy[mask]
        ann_ret = ret_r.mean() * TRADING_DAYS_PER_YEAR * 100
        ann_vol = ret_r.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        n_days = mask.sum()
        rows.append({
            "Regime": REGIME_LABELS.get(r, str(r)),
            "N Days": n_days,
            "Ann. Mean Return (%)": ann_ret,
            "Ann. Volatility (%)": ann_vol,
            "Sharpe Ratio": (ann_ret / ann_vol) if ann_vol > 0 else 0,
        })

    df = pd.DataFrame(rows)
    print("📊 HMM Regime Statistics (SPY Excess Returns):")
    display(df.round(3))

    if results_dir:
        import os
        df.to_csv(os.path.join(results_dir, "hmm_regime_stats.csv"), index=False)
        print(f"  💾 Saved: hmm_regime_stats.csv")

    return df


def hmm_residual_diagnostics(
    hmm_last_model,
    hmm_feats: pd.DataFrame,
    results_dir: str = None,
):
    """
    Residual diagnostics: ACF of standardized residuals.
    Checks that the HMM absorbs volatility clustering.
    """
    X = hmm_feats.values.astype(float)
    model = hmm_last_model.model_
    order = hmm_last_model._order

    # Decode states
    _, raw_states = model.decode(X, algorithm="viterbi")
    inv = np.argsort(order)
    states = inv[raw_states]

    # Compute standardized residuals
    means = model.means_[order]
    covars = model.covars_[order]

    residuals = np.zeros_like(X)
    for t in range(len(X)):
        s = states[t]
        mu = means[s]
        sigma = covars[s]
        # Mahalanobis-style: for simplicity, use diagonal std
        std = np.sqrt(np.diag(sigma))
        std[std == 0] = 1e-10
        residuals[t] = (X[t] - mu) / std

    # Use only the first feature (ΔVIX) for illustration
    resid_series = pd.Series(residuals[:, 0], index=hmm_feats.index,
                             name="Std. Residual (ΔVIX)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    _style_figure(fig, axes.flatten(),
                  "HMM — Residual Diagnostics (ΔVIX Feature)")

    # Time series of residuals
    axes[0, 0].plot(resid_series.index, resid_series.values,
                    color=PALETTE["accent1"], linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title("Standardized Residuals")
    axes[0, 0].set_ylabel("Residual")

    # Histogram
    axes[0, 1].hist(resid_series.values, bins=80, density=True,
                    color=PALETTE["accent2"], alpha=0.7, edgecolor="none")
    axes[0, 1].set_title("Distribution of Residuals")
    axes[0, 1].set_xlabel("Residual")

    # ACF of residuals
    plot_acf(resid_series.dropna(), ax=axes[1, 0], lags=40,
             title="ACF of Residuals", alpha=0.05,
             color=PALETTE["accent1"], vlines_kwargs={"color": PALETTE["accent1"]})
    axes[1, 0].set_title("ACF of Residuals")

    # ACF of squared residuals
    plot_acf(resid_series.dropna() ** 2, ax=axes[1, 1], lags=40,
             title="ACF of Squared Residuals", alpha=0.05,
             color=PALETTE["crisis"], vlines_kwargs={"color": PALETTE["crisis"]})
    axes[1, 1].set_title("ACF of Squared Residuals")

    plt.tight_layout()
    plt.show()
    if results_dir:
        import os
        fig.savefig(os.path.join(results_dir, "hmm_residual_diagnostics.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"], edgecolor="none")
        print(f"  💾 Saved: hmm_residual_diagnostics.png")
    plt.close(fig)

    return resid_series


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2: GMM + RF QUALITY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def gmm_bic_aic_comparison(
    gmm_feats: pd.DataFrame,
    k_range: range = range(2, 6),
    n_init: int = 20,
    random_state: int = 42,
    results_dir: str = None,
):
    """
    Fit GMMs with k=2..5 components and compute BIC/AIC + Silhouette scores.

    Returns a DataFrame with columns: k, BIC, AIC, Silhouette.
    """
    X = gmm_feats.values.astype(float)
    records = []

    for k in k_range:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=n_init,
                random_state=random_state,
            )
            gmm.fit(X)

        bic = gmm.bic(X)
        aic = gmm.aic(X)

        labels = gmm.predict(X)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan

        records.append({
            "k": k,
            "BIC": bic,
            "AIC": aic,
            "Silhouette": sil,
        })

    df = pd.DataFrame(records)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _style_figure(fig, axes,
                  "GMM — Model Selection: BIC, AIC & Silhouette Scores")

    axes[0].plot(df["k"], df["BIC"], "o-", color=PALETTE["accent1"],
                 linewidth=2, markersize=8, label="BIC")
    axes[0].set_xlabel("Number of Components (k)")
    axes[0].set_ylabel("BIC Score")
    axes[0].set_title("Bayesian Information Criterion")
    axes[0].set_xticks(list(k_range))
    axes[0].legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                   labelcolor=PALETTE["text"])

    axes[1].plot(df["k"], df["AIC"], "s-", color=PALETTE["accent2"],
                 linewidth=2, markersize=8, label="AIC")
    axes[1].set_xlabel("Number of Components (k)")
    axes[1].set_ylabel("AIC Score")
    axes[1].set_title("Akaike Information Criterion")
    axes[1].set_xticks(list(k_range))
    axes[1].legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                   labelcolor=PALETTE["text"])

    axes[2].plot(df["k"], df["Silhouette"], "D-", color=PALETTE["calm"],
                 linewidth=2, markersize=8, label="Silhouette")
    axes[2].set_xlabel("Number of Components (k)")
    axes[2].set_ylabel("Silhouette Score")
    axes[2].set_title("Average Silhouette Width")
    axes[2].set_xticks(list(k_range))
    axes[2].legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                   labelcolor=PALETTE["text"])

    plt.tight_layout()
    plt.show()
    if results_dir:
        import os
        fig.savefig(os.path.join(results_dir, "gmm_bic_aic_silhouette.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"], edgecolor="none")
        print(f"  💾 Saved: gmm_bic_aic_silhouette.png")
    plt.close(fig)

    print("\n📊 GMM Model Selection (BIC / AIC / Silhouette):")
    display(df.round(4))
    return df


def rf_classification_report(
    vix_regimes: pd.Series,
    ml_regimes: pd.Series,
    results_dir: str = None,
):
    """
    Out-of-sample classification evaluation of the Random Forest classifier.

    Uses VIX-based regime labels as ground-truth proxy to evaluate
    how well the GMM+RF pipeline reproduces market regime structure.

    Reports:
      - Confusion matrix
      - Precision, Recall, F1-Score (macro-averaged)
    """
    # Align indices
    common = vix_regimes.index.intersection(ml_regimes.index).sort_values()
    y_true = vix_regimes.reindex(common).values.astype(int)
    y_pred = ml_regimes.reindex(common).values.astype(int)

    target_names = [REGIME_LABELS[i] for i in sorted(set(y_true) | set(y_pred))]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=target_names,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T

    print("📊 GMM+RF — Out-of-Sample Classification Report (vs VIX Regimes):")
    display(report_df.round(4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.index.name = "True"
    cm_df.columns.name = "Predicted"

    print("\n📊 Confusion Matrix:")
    display(cm_df)

    # ── Plot confusion matrix ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    _style_figure(fig, ax,
                  "GMM+RF — Confusion Matrix (vs VIX Regimes)")

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, color=PALETTE["text"])
    ax.set_yticklabels(target_names, color=PALETTE["text"])
    ax.set_xlabel("Predicted Regime")
    ax.set_ylabel("True Regime (VIX)")

    for i in range(len(target_names)):
        for j in range(len(target_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else PALETTE["text"]
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    plt.tight_layout()
    plt.show()
    if results_dir:
        import os
        fig.savefig(os.path.join(results_dir, "rf_confusion_matrix.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"], edgecolor="none")
        print(f"  💾 Saved: rf_confusion_matrix.png")
    plt.close(fig)

    return report_df, cm_df


def rf_feature_importance_analysis(
    ml_last_pipeline,
    rf_feature_names: list,
    results_dir: str = None,
):
    """
    Feature importance breakdown for the RF ensemble.
    Reports Gini Importance / Mean Decrease in Impurity (MDI) per regime.
    """
    if ml_last_pipeline is None or not hasattr(ml_last_pipeline, "ensemble_"):
        print("⚠️  ML pipeline not available for feature importance analysis.")
        return None

    feat_imp = ml_last_pipeline.ensemble_.feature_importances()

    rows = []
    for regime_id, importances in feat_imp.items():
        for fname, imp in zip(rf_feature_names, importances):
            rows.append({
                "Regime": REGIME_LABELS.get(regime_id, str(regime_id)),
                "Feature": fname,
                "Importance (%)": imp * 100,
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Feature", columns="Regime", values="Importance (%)")
    pivot = pivot.sort_values(pivot.columns[0], ascending=False)

    print("📊 RF Feature Importance Breakdown (Gini / MDI %):")
    display(pivot.round(2))

    # ── Plot ──────────────────────────────────────────────────────────────
    n_regimes = len(feat_imp)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 8),
                             sharey=True)
    if n_regimes == 1:
        axes = [axes]
    _style_figure(fig, axes,
                  "RF — Feature Importance by Regime (Gini / MDI)")

    for idx, (regime_id, importances) in enumerate(feat_imp.items()):
        order = np.argsort(importances)[::-1]
        top_n = min(15, len(order))
        order = order[:top_n]

        names = [rf_feature_names[i] for i in order]
        vals = importances[order] * 100

        color = REGIME_COLORS.get(regime_id, PALETTE["accent1"])
        axes[idx].barh(range(len(names)), vals[::-1],
                       color=color, edgecolor="white", linewidth=0.5)
        axes[idx].set_yticks(range(len(names)))
        axes[idx].set_yticklabels(names[::-1], fontsize=9,
                                  color=PALETTE["text"])
        axes[idx].set_xlabel("Importance (%)")
        axes[idx].set_title(f"Regime {regime_id}: "
                            f"{REGIME_LABELS.get(regime_id, '?')}")

    plt.tight_layout()
    plt.show()
    if results_dir:
        import os
        fig.savefig(os.path.join(results_dir, "rf_feature_importance_detail.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"], edgecolor="none")
        print(f"  💾 Saved: rf_feature_importance_detail.png")
    plt.close(fig)

    return pivot


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION: Run all quality metrics
# ═════════════════════════════════════════════════════════════════════════════

def run_all_quality_metrics(
    hmm_feats: pd.DataFrame,
    hmm_last_model,
    hmm_regimes: pd.Series,
    gmm_feats: pd.DataFrame,
    ml_regimes: pd.Series,
    ml_last_pipeline,
    rf_feature_names: list,
    vix_regimes: pd.Series,
    excess_spy: pd.Series,
    results_dir: str = None,
):
    """
    Run ALL model quality metrics for both HMM and GMM+RF pipelines.
    Call this from the notebook after all models have been fitted.

    Parameters
    ----------
    hmm_feats : pd.DataFrame
        HMM feature matrix (ΔVIX + log-returns).
    hmm_last_model : HMMRegimeModel
        The last fitted HMM model from walk-forward.
    hmm_regimes : pd.Series
        HMM regime labels (walk-forward OOS).
    gmm_feats : pd.DataFrame
        GMM feature matrix (ΔVIX + log-returns).
    ml_regimes : pd.Series
        ML (GMM+RF) regime labels (walk-forward OOS).
    ml_last_pipeline : MLRegimePipeline
        The last fitted ML pipeline from walk-forward.
    rf_feature_names : list
        List of feature names for the Random Forest.
    vix_regimes : pd.Series
        VIX-based regime labels (used as ground-truth proxy).
    excess_spy : pd.Series
        SPY excess returns.
    results_dir : str, optional
        Directory to save figures and CSVs.
    """
    print("=" * 80)
    print("  MODEL QUALITY METRICS — HMM")
    print("=" * 80)

    print("\n" + "─" * 60)
    print("  1. BIC / AIC Model Selection")
    print("─" * 60)
    hmm_bic_df = hmm_bic_aic_comparison(hmm_feats, results_dir=results_dir)

    print("\n" + "─" * 60)
    print("  2. Transition Matrix Analysis: Persistence & Half-Lives")
    print("─" * 60)
    persist_df, ergodic_df = hmm_transition_analysis(
        hmm_last_model, results_dir=results_dir
    )

    print("\n" + "─" * 60)
    print("  3. Regime Statistics (Annualized Returns & Volatility)")
    print("─" * 60)
    regime_stats_df = hmm_regime_statistics(
        hmm_regimes, excess_spy, results_dir=results_dir
    )

    print("\n" + "─" * 60)
    print("  4. Residual Diagnostics (ACF)")
    print("─" * 60)
    residuals = hmm_residual_diagnostics(
        hmm_last_model, hmm_feats, results_dir=results_dir
    )

    print("\n\n")
    print("=" * 80)
    print("  MODEL QUALITY METRICS — GMM + RF PIPELINE")
    print("=" * 80)

    print("\n" + "─" * 60)
    print("  1. GMM: BIC / AIC / Silhouette Model Selection")
    print("─" * 60)
    gmm_bic_df = gmm_bic_aic_comparison(gmm_feats, results_dir=results_dir)

    print("\n" + "─" * 60)
    print("  2. RF: Out-of-Sample Classification Report")
    print("─" * 60)
    report_df, cm_df = rf_classification_report(
        vix_regimes, ml_regimes, results_dir=results_dir
    )

    print("\n" + "─" * 60)
    print("  3. RF: Feature Importance Breakdown")
    print("─" * 60)
    feat_imp_df = rf_feature_importance_analysis(
        ml_last_pipeline, rf_feature_names, results_dir=results_dir
    )

    print("\n" + "=" * 80)
    print("  ✅ ALL QUALITY METRICS COMPLETE")
    print("=" * 80)

    return {
        "hmm_bic_aic": hmm_bic_df,
        "hmm_persistence": persist_df,
        "hmm_ergodic": ergodic_df,
        "hmm_regime_stats": regime_stats_df,
        "hmm_residuals": residuals,
        "gmm_bic_aic_silhouette": gmm_bic_df,
        "rf_classification_report": report_df,
        "rf_confusion_matrix": cm_df,
        "rf_feature_importances": feat_imp_df,
    }
