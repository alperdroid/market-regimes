# Robustness and Statistical Significance

## Methods paragraph (paper-ready)

**Robustness: correcting for data snooping.** Because we evaluate ten competing
strategies on a single historical sample, the in-sample outperformance of the
best performer is mechanically inflated by selection: the more configurations we
search, the higher the maximum Sharpe ratio we expect to observe even under the
null of no genuine skill (Harvey, Liu, and Zhu, 2016). We therefore subject the
leading strategy (ML-TPF) to two complementary multiple-testing corrections, both
computed on the net-of-cost daily excess-return series.

First, we apply **White's (2000) Reality Check**, which tests the null hypothesis
that the best strategy does not outperform the passive SPY benchmark once the full
search universe is taken into account. For each strategy *k* we form the daily
relative-performance series *d*<sub>k,t</sub> = *r*<sub>k,t</sub> − *r*<sub>SPY,t</sub>
and compute the studentized statistic *V* = max<sub>k</sub> √T · d̄<sub>k</sub>.
The sampling distribution of *V* under the null is obtained by the stationary
bootstrap of Politis and Romano (1994) with mean block length 10 days (2,000
resamples), which preserves the serial dependence and conditional
heteroskedasticity of daily returns; the bootstrap *p*-value is the share of
recentered resampled statistics that exceed *V*. Second, we report the **Deflated
Sharpe Ratio** (DSR) of Bailey and López de Prado (2014), which evaluates the
selected strategy's Sharpe ratio against the *expected maximum* Sharpe ratio
attainable by chance given the number of trials *N* and the cross-sectional
variance of the trials' Sharpe ratios, while correcting the probabilistic Sharpe
ratio for the non-normality (skewness and excess kurtosis) and finite length of
the return sample.

**Findings.** The two tests are deliberately distinct — White's Reality Check
evaluates *raw* outperformance over the benchmark, whereas the DSR evaluates the
*risk-adjusted* return — and they yield a sharp, economically informative
contrast. The Deflated Sharpe Ratio of ML-TPF is 0.977 (annualized Sharpe 0.61
against a snooping-adjusted null threshold of 0.15), indicating that its
risk-adjusted performance is unlikely to be an artifact of the search and is
significant at the 95% level. By contrast, White's Reality Check returns a
*p*-value of 0.56, so we cannot reject the hypothesis that ML-TPF fails to beat
SPY in *return* terms once data snooping is accounted for; the strategy's
single-equation CAPM alpha is likewise only borderline (Jensen α = 4.25% p.a.,
*p* = 0.059, net of 10 bps one-way costs). Taken together, the evidence indicates
that the gains from regime-conditional, machine-learning-driven allocation are
concentrated in **volatility and drawdown reduction rather than abnormal returns**:
the strategy's superior Sharpe ratio is robust to multiple testing, but its
apparent return advantage over a passive benchmark is not statistically
distinguishable from chance after correcting for the breadth of the strategy
search. This pattern is reinforced by the minimum-variance variants, whose
risk-reduction benefit is large, low-turnover, and robust across all three regime
classifiers.

## Reproducibility note

All statistics are computed in `performance/metrics.py`
(`whites_reality_check`, `deflated_sharpe_ratio`, `probabilistic_sharpe_ratio`)
and written to `results/significance_tests.csv`. The bootstrap uses a fixed seed
(42), 2,000 resamples, and a mean block length of 10 trading days. The underlying
net and gross daily return matrices are persisted to
`results/port_returns_net.csv` and `results/port_returns_gross.csv`, so the tests
can be reproduced without refitting the regime models.

## Suggested references

- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica* 68(5), 1097–1126.
- Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap." *JASA* 89(428), 1303–1313.
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management* 40(5), 94–107.
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk* 15(2), 3–44.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns." *Review of Financial Studies* 29(1), 5–68.
