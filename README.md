# diffuse — information diffusion in markets

Information doesn't arrive in all assets at the same time. When big news hits, index futures move first, then the ETFs, then the less liquid stuff catches up — sometimes minutes later, sometimes the next bar. This project tries to measure that structure properly instead of hand-waving about it.

The idea: estimate a **directed information-flow network** between liquid ETFs (transfer entropy + Granger causality), simulate how shocks propagate through it (SIR-style diffusion), and then test the only question that matters — **do the "leaders" actually predict the followers out-of-sample?** With transaction costs on, walk-forward validation, and a null model to keep me honest.

## who this is for

Quant researchers / traders who are tired of backtests that only work in-sample. And anyone interviewing for quant roles who wants a project they can defend line by line — every number in here comes with its sample size and its caveats.

## the pipeline

```
yfinance (1h bars, ~20 liquid ETFs)
        │
        ▼
┌───────────────────┐
│  network.py       │  transfer entropy (histogram estimator, k=l=1)
│                   │  + permutation test per pair (200 shuffles)
│                   │  + granger causality as a second opinion
└────────┬──────────┘
         │  directed graph, edge weight = TE in nats
         ▼
┌───────────────────┐
│  diffusion.py     │  SIR shock simulation on the graph
│                   │  sim travel times vs empirical lead-lag (spearman)
└────────┬──────────┘
         ▼
┌───────────────────┐
│  backtest.py      │  walk-forward: estimate network on 30d,
│                   │  trade next 5d. leaders → followers, 1-bar hold.
│                   │  costs ON (2 bps/trade). signals shift(1)'d.
└────────┬──────────┘
         ▼
┌───────────────────┐
│  null_model.py    │  same backtest on permuted networks.
│                   │  if we don't beat random wiring, it's noise.
└───────────────────┘
```

Run it: `python run.py` (full) or `python run.py --quick` (8 tickers, smoke test).
Notebooks `01_network.ipynb` / `02_diffusion.ipynb` walk through the pieces interactively.

## results

The estimated network from the full run:

![information-flow network](results/network.png)

Full run: 20 liquid ETFs, 455 hourly bars (3 months via yfinance), network estimated
with 200-shuffle permutation tests per pair, 8 walk-forward windows (30d estimate /
5d test), 2 bps per trade, 20 permuted-network nulls.

| metric | strategy | null (permuted nets) |
|---|---|---|
| OOS bars | 256 (~40 trading days) | 256 × 20 |
| network edges | 28 / 380 pairs | — |
| top leaders (full-sample) | XLF, GLD, XLC, SPY, XLY | — |
| total return | −0.86% | — |
| Sharpe (ann., rf=0, 1638 bars/yr) | −0.62 | −2.78 ± 2.22 |
| max drawdown | −3.07% | — |
| hit rate | 45.3% | — |
| cost | 2 bps/trade | 2 bps/trade |
| P(null Sharpe ≥ real Sharpe) | — | **0.24** |

Diffusion check: Spearman ρ between simulated travel times and empirical lead-lag
= **−0.25** (p = 0.020, n = 86 pairs). Significant — and negative. The SIR cascade
runs *backwards* relative to measured lead-lag, so the diffusion metaphor as
specified doesn't describe the data. Honest conclusion: the network finds
statistically significant directed edges, but neither the shock-propagation story
nor the trading rule built on it survives contact with out-of-sample data.

On the backtest: the strategy loses a little money (−0.86%) and doesn't
significantly beat randomly rewired networks (p = 0.24). It does beat the *average*
null, which is faint praise — the nulls lose badly, mostly on turnover. With only
256 OOS bars the confidence bands are wide; a longer history could change the
verdict either way, and I'm not going to pretend otherwise.

## what didn't work (the honest part)

1. **Hourly is probably the wrong frequency for this.** Most ETF co-movement resolves within minutes, not hours. By the next hourly bar the information is already in the price — and what's left looks like noise or slight mean reversion. The TE network finds *something* (28 edges survive the permutation test), but it doesn't translate into next-bar predictability.
2. **The leaders don't persist.** Across the 8 walk-forward windows the top-5 leader set churns heavily (XLC, USO, TLT, XLRE, QQQ, XLF all take turns). If the leadership structure were a stable feature of the market, it wouldn't reshuffle every 5 days. Either the true structure moves fast, or most edges are estimation noise — probably some of both.
3. **The diffusion metaphor failed its own test.** Simulated travel times correlate *negatively* (−0.25, significant) with empirical lead-lag. So whatever the TE edges capture, SIR-style cascade dynamics aren't it. Killed by its own validation check — as designed.
4. **Costs dominate.** Even if there were a faint signal, flipping positions every hour at 2 bps with ~15 names turns over the whole book constantly. The cost model doing its job is the point — most public backtests just leave it off.
5. **The market factor swamps everything.** SPY/QQQ/XLF have high out-strength mostly because they're the market, not because they carry *incremental* timing info. A better version would orthogonalize against the market factor first and look for *residual* lead-lag. That's the obvious next experiment.

What I'd try next: 5-minute bars (finer diffusion timescale), market-neutralized returns, and sector-rotation signals with multi-day holds instead of 1-bar holds.

## limitations

- yfinance hourly bars only go back ~2 years and have survivorship bias (delisted ETFs aren't in the universe).
- Transfer entropy with 5 quantile bins and k=l=1 is a coarse estimator — it can miss longer-memory effects and fine nonlinear structure.
- The permutation test is per-pair with no multiple-testing correction across 380 pairs; at α=0.05 expect ~19 false edges under the global null. The network is probably noisier than it looks.
- Walk-forward windows are short (5 test days); Sharpe estimates on ~250 hourly bars have wide confidence bands.
- No borrow costs, no market impact model — 2 bps flat is a simplification.

## setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --quick   # smoke test, a few minutes
python run.py           # full run, ~1h (network estimation is the bottleneck)
pytest tests/           # 14 tests, incl. a no-look-ahead test
```

## what i learned

- Transfer entropy is easy to implement and surprisingly easy to fool yourself with. The permutation test is non-negotiable — raw TE values are meaningless without a null.
- Writing the null model *before* looking at results is the best decision I made here. It would have been very tempting to tune the universe/thresholds until the Sharpe turned positive.
- Backtest bugs don't look like crashes, they look like profits. The `test_no_lookahead` test (reverse the future, early signals must not change) is the most valuable test in the repo.
- Honestly, the most useful output of this project isn't the strategy — it's the machinery: a tested TE estimator, a point-in-time walk-forward harness, and a null-model framework I can reuse on better ideas.
