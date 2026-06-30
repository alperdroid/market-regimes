# Market Regimes: ML vs. Hidden Markov Models in US Tactical Sector Allocation

A reproducible research project comparing three families of **market-regime
classifiers** and testing whether conditioning a US sector-rotation strategy on the
inferred regime beats a passive S&P 500 benchmark — out-of-sample and net of
transaction costs.

> **Headline result.** All three regime classifiers feed an *identical* regime-conditional
> portfolio construction, so performance gaps are attributable to the labelling method
> alone. Regime conditioning delivers a large, robust reduction in risk (drawdowns fall
> from ~−60% to ~−37%, Sharpe rises from 0.44 to 0.48–0.54 via minimum-variance). But the
> gain is purely risk targeting: the return-timing (tangency) variants fail uniformly, and
> **no strategy out-returns the S&P 500 once data snooping is corrected for** (White's
> Reality Check *p* = 0.98). **The value is in risk control, not abnormal returns — and it
> is largely insensitive to the sophistication of the regime model.**

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
   - **GMM** — unsupervised Gaussian Mixture clustering (memoryless; no transition
     matrix). Produces regime *labels only*.
   All learned models are fit **walk-forward** (expanding window, yearly refit) so the
   labels are strictly out-of-sample. Every classifier then feeds the *same* downstream
   portfolio construction (moments from the historical days in the prevailing regime), so
   differences are attributable to the regime map, not to an auxiliary return model.
4. **Regime-conditional CAPM betas** — how sector market-sensitivity shifts across the
   volatility cycle.
5. **Backtest** — 12 strategies (passive, 1/N, and static/VIX/HMM/GMM/Ensemble × MVP/TPF),
   monthly rebalance, Ledoit–Wolf covariance shrinkage, crisis cash buffer, and measured
   transaction costs.
6. **Performance & significance** — Sharpe, Sortino, drawdown, Jensen's α, break-even
   cost, plus **Deflated Sharpe Ratio** and **White's Reality Check** for data snooping.
7. **Figures** — 10 publication charts + diagnostic plots written to `results/`.

---

## Results at a glance

Out-of-sample, net of 10 bps one-way costs (2006–2025):

| Strategy   | Ann. Ret. % | Vol. % | Sharpe | Max DD % | Turnover | Break-even bps |
|------------|------------:|-------:|-------:|---------:|---------:|---------------:|
| SPY B&H    | 7.00 | 19.8 | 0.44 | −59.6 | 0%   | — |
| **GMM-MVP**| **6.69** | 13.9 | **0.54** | −39.8 | 121% | **118.0** |
| VIX-MVP    | 6.04 | 13.4 | 0.50 | −34.4 | 73%  | 125.3 |
| HMM-MVP    | 6.19 | 14.0 | 0.50 | −38.7 | 122% | 77.5 |
| HMM-TPF    | 5.16 | 15.7 | 0.40 | −55.1 | 464% | 0.0 |
| GMM-TPF    | 3.86 | 18.1 | 0.30 | −44.5 | 433% | 0.0 |

The minimum-variance (MVP) strategies are the robust, low-turnover, cost-tolerant winners —
and they are nearly identical across classifiers (GMM 0.54, HMM 0.50, VIX 0.50), so the gain
comes from risk targeting, not the regime model (the memoryless GMM even edges out the HMM).
Every tangency (TPF) variant fails: with all methods sharing one portfolio construction,
regime-conditional expected returns are too noisy to time, and no strategy beats the
benchmark on raw return.

---

## Repository layout

```
market_regimes/
├── main.py                 # end-to-end pipeline
├── config.py               # all parameters, tickers, thresholds
├── data/                   # loaders + feature engineering
├── regimes/                # vix_classifier, hmm_model, gmm_pipeline, ensemble
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

- **Walk-forward, out-of-sample** estimation throughout — no look-ahead in regime labels
  or moment estimation.
- **Data-snooping corrections**: Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
  and White's Reality Check (White, 2000) via the stationary bootstrap (Politis &
  Romano, 1994). See [`results/significance_methods.md`](market_regimes/results/significance_methods.md).
- Regime-switching: Hamilton (1989); Ang & Bekaert (2002); Guidolin & Timmermann
  (2007). VIX thresholds: Whaley (2009); Bloom (2009). Covariance shrinkage: Ledoit &
  Wolf (2004). Estimation risk / 1/N: DeMiguel, Garlappi & Uppal (2009); minimum
  variance: Clarke, de Silva & Thorley (2006). Gaussian mixtures: McLachlan & Peel (2000).

Full reference list in the paper.
