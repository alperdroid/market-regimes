# Market Regimes: ML vs. Hidden Markov Models in US Tactical Sector Allocation

A reproducible research project comparing three families of **market-regime
classifiers** and testing whether conditioning a US sector-rotation strategy on the
inferred regime beats a passive S&P 500 benchmark — out-of-sample and net of
transaction costs.

> **Headline result.** Regime conditioning delivers a large, robust reduction in risk
> (drawdowns fall from ~−60% to ~−37%, Sharpe rises modestly above the benchmark), but
> the best machine-learning strategy's apparent *return* advantage does **not** survive
> a data-snooping correction (White's Reality Check, *p* = 0.56) even though its
> risk-adjusted Sharpe does (Deflated Sharpe Ratio = 0.98). **The value is in risk
> control, not abnormal returns.**

The full write-up is in [`market_regimes/PAPER.md`](market_regimes/PAPER.md) (Markdown)
and [`market_regimes/paper.tex`](market_regimes/paper.tex) (LaTeX, with figures/tables).

---

## What it does

The pipeline ([`market_regimes/main.py`](market_regimes/main.py)) runs end-to-end in
seven steps:

1. **Data** — daily prices for 9 SPDR sector ETFs + SPY, VIX, the risk-free rate, and
   FRED macro spreads (TED, 10Y–2Y), 2004–2025.
2. **Features** — log/excess returns, ΔVIX, and a macro + lagged-return matrix.
3. **Regime identification** — three classifiers, each labelling every day Calm /
   Transitional / Crisis:
   - **VIX rule** — fixed thresholds (20 / 30).
   - **Gaussian HMM** — 3-state, full-covariance, Baum–Welch + Viterbi, with state
     sorting to fix label-switching.
   - **ML pipeline** — Gaussian Mixture clustering → three regime-specialist Random
     Forest return forecasters.
   All learned models are fit **walk-forward** (expanding window, yearly refit) so
   labels and forecasts are strictly out-of-sample.
4. **Regime-conditional CAPM betas** — how sector market-sensitivity shifts across the
   volatility cycle.
5. **Backtest** — 10 strategies (passive, 1/N, static/VIX/HMM/ML × MVP/TPF), monthly
   rebalance, Ledoit–Wolf covariance shrinkage, crisis cash buffer, and measured
   transaction costs.
6. **Performance & significance** — Sharpe, Sortino, drawdown, Jensen's α, break-even
   cost, plus **Deflated Sharpe Ratio** and **White's Reality Check** for data snooping.
7. **Figures** — 11 publication charts + diagnostic plots written to `results/`.

---

## Results at a glance

Out-of-sample, net of 10 bps one-way costs (2006–2025):

| Strategy   | Ann. Ret. % | Vol. % | Sharpe | Max DD % | Turnover | Break-even bps |
|------------|------------:|-------:|-------:|---------:|---------:|---------------:|
| SPY B&H    | 7.01 | 19.8 | 0.44 | −59.6 | 0%   | — |
| ML-MVP     | 6.21 | 14.2 | 0.50 | −38.9 | 77%  | **110.9** |
| HMM-MVP    | 5.80 | 14.1 | 0.47 | −41.3 | 108% | 49.1 |
| VIX-MVP    | 5.65 | 13.8 | 0.47 | −36.6 | 59%  | 72.8 |
| **ML-TPF** | **9.48** | 17.3 | **0.61** | −38.0 | 621% | 56.9 |
| HMM-TPF    | 2.95 | 19.7 | 0.25 | −54.9 | 496% | 0.0 |

The minimum-variance (MVP) strategies are the robust, low-turnover, cost-tolerant
winners. The high-turnover tangency (TPF) portfolios are flattered by frictionless
accounting; ML-TPF's edge is risk reduction, not return.

---

## Repository layout

```
market_regimes/
├── main.py                 # end-to-end pipeline
├── config.py               # all parameters, tickers, thresholds
├── data/                   # loaders + feature engineering
├── regimes/                # vix_classifier, hmm_model, ml_pipeline
├── portfolio/              # optimizer (MVP/TPF), ledoit_wolf, backtest
├── capm/                   # regime-conditional beta analysis
├── performance/            # metrics + significance tests
├── visualization/          # all figures
├── results/                # generated figures (.png), tables (.csv), data cache
├── PAPER.md                # full paper (Markdown)
└── paper.tex               # full paper (LaTeX)
market_regimes.ipynb        # exploratory notebook
```

---

## Reproducing

Requires Python 3.11+ with the packages in
[`market_regimes/requirements.txt`](market_regimes/requirements.txt)
(`numpy`, `pandas`, `scipy`, `scikit-learn`, `hmmlearn`, `statsmodels`, `yfinance`,
`pandas_datareader`, `matplotlib`).

```bash
cd market_regimes
pip install -r requirements.txt
python main.py          # ~15–20 min (the HMM walk-forward dominates)
```

Outputs are written to `market_regimes/results/`. A cached copy of the market data
(`results/data_cache.pkl`) is committed so the pipeline runs offline; delete it to
force a fresh download.

### Build the paper

```bash
cd market_regimes
pdflatex paper.tex && pdflatex paper.tex   # run twice for refs + citations
```

Only standard LaTeX packages are used; it also compiles as-is on Overleaf (upload
`paper.tex` together with the `results/` folder).

---

## Method notes & references

- **Walk-forward, out-of-sample** estimation throughout — no look-ahead in labels,
  forecasts, or moment estimation.
- **Data-snooping corrections**: Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
  and White's Reality Check (White, 2000) via the stationary bootstrap (Politis &
  Romano, 1994). See [`results/significance_methods.md`](market_regimes/results/significance_methods.md).
- Regime-switching: Hamilton (1989); Ang & Bekaert (2002); Guidolin & Timmermann
  (2007). VIX thresholds: Whaley (2009); Bloom (2009). Covariance shrinkage: Ledoit &
  Wolf (2004). Estimation risk / 1/N: DeMiguel, Garlappi & Uppal (2009); minimum
  variance: Clarke, de Silva & Thorley (2006). Random Forests: Breiman (2001).

Full reference list in the paper.
