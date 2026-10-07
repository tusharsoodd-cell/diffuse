# interview notes — diffuse

Questions I expect, and my answers. Written so future-me (or an interviewer) can
see the reasoning, not just the code.

## why transfer entropy over correlation?

Correlation is symmetric and linear. If I just want "these move together", correlation
is fine — but I want *direction*: does X's move tell me something about Y's *next*
move that Y's own history doesn't already say? That's exactly what TE measures:
TE(x→y) = H(y+|y) − H(y+|y,x), the reduction in uncertainty about y's next return
once you know x's current return, beyond what y's own past gives you.

Two practical reasons on top of the theoretical one:
1. Returns are nonlinear and fat-tailed. A linear VAR (which is what Granger is)
   can miss structure that a model-free estimator catches.
2. TE is still just a statistic, not magic — that's why every pair goes through
   a permutation test, and why Granger p-values ride along as a second opinion.
   When they disagree, I trust neither blindly.

## did the null model kill the result?

Mostly. On hourly bars the strategy Sharpe was −0.62 over 256 out-of-sample bars,
and 24% of permuted networks did better (p = 0.24, 20 nulls) — not statistically
significant either way. The real network beat the *average* null (−2.78), which is
faint praise: the nulls mostly lost on turnover, and with 256 OOS bars the
confidence bands are wide enough that a longer history could move the verdict
either way.

My read: at hourly frequency the information is already in the price by the next
bar. The TE edges are probably picking up contemporaneous correlation leaking
across bar boundaries plus some stale-pricing effects, neither of which is
tradeable at this horizon with these costs. And the diffusion check backed that
up — simulated travel times correlated *negatively* with empirical lead-lag
(ρ = −0.25, p = 0.02), so the SIR cascade story doesn't hold either. The fix isn't
a better estimator, it's a better timescale (5-min bars) or a different signal
construction (market-neutral residuals, multi-day holds).

## what breaks under regime change?

A few things, and they're all in the limitations section:

- **The network is estimated on 30-day windows.** In a vol regime shift (e.g. March
  2020-style), the leaders change — flight-to-quality makes TLT/GLD behave
  completely differently. The walk-forward re-estimates, but there's a lag: you're
  always trading last month's network.
- **TE assumes stationarity within the window.** If the dependence structure
  itself is changing inside the estimation window, the estimate is mush.
- **Liquidity regimes matter more than vol regimes for this.** The whole premise
  is "some assets impound news faster". In a liquidity crunch, *everything*
  becomes the slow asset and the leader/follower distinction collapses.

If I were running this for real, I'd estimate the network on exponentially-weighted
data and monitor edge stability (Jaccard similarity of edge sets across windows)
as a regime indicator — when the network itself is unstable, stand down.

## where could look-ahead leak in, and how did you prevent it?

The four classic leak points, in order of how likely I was to mess each one up:

1. **Signal timing.** The signal for bar t+1 must use only data through bar t.
   I compute signals from leader returns at t, then `shift(1)` the positions.
   There's a dedicated test: reverse the second half of the return series and
   assert the first-half signals don't change. If anything in the pipeline
   peeked forward, that test fails.

2. **Network estimation window.** The network for test window [T, T+5d] is
   estimated on [T−30d, T] only. The walk-forward loop enforces this by
   construction — `est` and `tst` are disjoint slices.

3. **Universe selection.** The 20 ETFs were picked *a priori* (largest liquid
   ETFs), not by screening for "ones where this works". Picking the universe
   on full-sample performance would be survivorship/selection bias.

4. **Normalization / z-scoring.** If you z-score signals using full-sample
   mean/std, that's a leak. I deliberately use raw sign() of the weighted sum —
   no scaling parameters estimated anywhere, so there's nothing to leak.

What I did *not* fully solve: the permutation-test threshold (α=0.05) and
hyperparameters (n_leaders=5, window lengths) were chosen by judgment, not tuned —
but they're also not validated. A purist would nest another validation loop.
For a research repo, stating the choices beats pretending they were optimized.
