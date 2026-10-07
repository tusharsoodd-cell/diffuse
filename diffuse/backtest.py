"""walk-forward backtest: do information leaders predict followers?

protocol, per estimation window:
  1. estimate the TE network on the window (in-sample only)
  2. pick top-k leaders by out-strength
  3. for each follower, signal = sign(weighted sum of leader returns)
  4. hold the position for one bar, then re-evaluate

everything is point-in-time: the position for bar t+1 is decided with
information available at the close of bar t. signals are shift(1)'d and
there's an assertion in the tests that would catch a leak.
"""

import numpy as np
import pandas as pd

from .network import build_network, top_leaders, out_strength


def leader_signals(returns_window, leaders, G, followers=None):
    """signal matrix for one window. +1/-1 per follower per bar.

    signal_j,t = sign(sum_i w_ij * r_i,t), computed at close t,
    traded over bar t -> t+1.
    """
    followers = followers or [n for n in G.nodes() if n not in leaders]
    sig = pd.DataFrame(0.0, index=returns_window.index, columns=followers)
    for f in followers:
        wsum = pd.Series(0.0, index=returns_window.index)
        for l in leaders:
            if G.has_edge(l, f):
                w = G[l][f]["te"]
                wsum += w * returns_window[l]
        sig[f] = np.sign(wsum)
    # no signal where leaders had no edge into the follower -> flat, not 0-cost churn
    return sig


def apply_costs(positions, cost_bps):
    """turnover * cost. positions are in [-1, 1], gross exposure normalized."""
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    return turnover * (cost_bps / 1e4)


def walk_forward(prices, estimate_bars=30 * 7, test_bars=5 * 7, n_leaders=5,
                 cost_bps=2.0, n_shuffles=100, verbose=True):
    """rolling estimation -> trade the next window -> stitch OOS returns.

    estimate_bars / test_bars are in bars (default ~30d / ~5d of hourly bars).
    network_kwargs lets you cheapen the TE estimation for the walk-forward.
    """
    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    rets = rets.dropna(axis=1, how="any")  # need full history per ticker

    oos = []
    t = 0
    n = len(rets)
    while t + estimate_bars + test_bars <= n:
        est = rets.iloc[t:t + estimate_bars]
        tst = rets.iloc[t + estimate_bars:t + estimate_bars + test_bars]

        G = build_network(est, n_shuffles=n_shuffles, verbose=False)
        leaders = top_leaders(G, k=n_leaders)
        if not leaders or G.number_of_edges() == 0:
            t += test_bars
            continue

        sig_test = leader_signals(tst, leaders, G)
        # NOTE: signals are computed on the TEST window's returns but using
        # only the leaders and edge weights estimated on the TRAIN window.
        # the position for bar t+1 uses leader returns at bar t (shift(1)
        # below), so nothing from the future leaks in.

        # gross exposure = 1, split equally across followers with a signal
        pos = sig_test.div(sig_test.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        pos = pos.shift(1).fillna(0.0)  # decided at close t, held over t->t+1

        pnl = (pos * tst).sum(axis=1)
        costs = apply_costs(pos, cost_bps)
        oos.append(pnl - costs)

        if verbose:
            print(f"window {t}: leaders={leaders}, test pnl={float((pnl - costs).sum()):.4f}")
        t += test_bars

    if not oos:
        return pd.Series(dtype=float)
    return pd.concat(oos).sort_index()


def sharpe(returns, periods_per_year):
    """annualized sharpe. convention: mean/std * sqrt(ppy), risk-free = 0."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(returns):
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    return float(((cum - peak) / peak).min())


def summarize(returns, periods_per_year, cost_bps, label="strategy"):
    """one-row results table, honest version."""
    r = returns.dropna()
    n = len(r)
    return {
        "label": label,
        "n_bars": n,
        "total_return": float((1 + r).prod() - 1) if n else np.nan,
        "sharpe_ann": sharpe(r, periods_per_year),
        "sharpe_convention": f"annualized, rf=0, {periods_per_year} bars/yr",
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()) if n else np.nan,
        "cost_bps_per_trade": cost_bps,
        "turnover_per_bar": np.nan,  # filled by caller if needed
    }
