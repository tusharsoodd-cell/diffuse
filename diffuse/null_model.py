"""null models: is the network actually saying anything?

the backtest means nothing without a baseline. two nulls:
  1. permuted network - same nodes, same number of edges, targets shuffled.
     keeps the degree sequence-ish structure, destroys the information
     content. if the real network doesn't beat this, the "effect" is noise.
  2. random signals - same turnover profile, random signs. sanity check
     that we're not just harvesting some trivial bias.
"""

import numpy as np
import pandas as pd
import networkx as nx

from .backtest import walk_forward


def permuted_network(G, seed=0):
    """shuffle edge targets. preserves node set, edge count, and the
    multiset of weights - only the *wiring* is randomized."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    edges = [(u, v, d) for u, v, d in G.edges(data=True)]
    targets = [v for _, v, _ in edges]
    rng.shuffle(targets)
    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    for (u, _, d), v_new in zip(edges, targets):
        if u != v_new:
            H.add_edge(u, v_new, **d)
    return H


def null_walk_forward(prices, n_perms=20, seed=123, **bt_kwargs):
    """run the identical walk-forward backtest on permuted networks.

    returns one OOS pnl series per permutation. the real (unpermuted) run
    is computed separately by the caller. keeps the comparison apples-to-
    apples: same windows, same costs, same leader rule, only the wiring
    of the network changes.
    """
    from .network import build_network, top_leaders
    from .backtest import leader_signals, apply_costs

    rets = np.log(prices / prices.shift(1)).dropna(how="all").dropna(axis=1, how="any")
    estimate_bars = bt_kwargs.get("estimate_bars", 30 * 7)
    test_bars = bt_kwargs.get("test_bars", 5 * 7)
    n_leaders = bt_kwargs.get("n_leaders", 5)
    cost_bps = bt_kwargs.get("cost_bps", 2.0)
    n_shuffles = bt_kwargs.get("n_shuffles", 100)

    # estimate the real network ONCE per window, then permute each one
    # n_perms times. re-estimating inside the perm loop would take hours.
    windows = []
    t = 0
    n = len(rets)
    while t + estimate_bars + test_bars <= n:
        a, b, c = t, t + estimate_bars, t + estimate_bars + test_bars
        G = build_network(rets.iloc[a:b], n_shuffles=n_shuffles, verbose=False)
        windows.append((rets.iloc[b:c], G))
        t += test_bars

    null_pnls = []
    for p in range(n_perms):
        oos = []
        for tst, G in windows:
            Gp = permuted_network(G, seed=seed + p)
            leaders = top_leaders(Gp, k=n_leaders)
            if not leaders or Gp.number_of_edges() == 0:
                continue
            sig_test = leader_signals(tst, leaders, Gp)
            pos = sig_test.div(sig_test.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            pos = pos.shift(1).fillna(0.0)
            pnl = (pos * tst).sum(axis=1)
            oos.append(pnl - apply_costs(pos, cost_bps))
        null_pnls.append(pd.concat(oos).sort_index() if oos else pd.Series(dtype=float))

    return null_pnls


def compare_to_null(real_pnl, null_pnls, periods_per_year):
    """sharpe of the real run vs the distribution of null sharpes.

    reports the fraction of nulls that beat the real thing - that's the
    empirical p-value for "the network adds nothing".
    """
    from .backtest import sharpe
    real = sharpe(real_pnl, periods_per_year)
    nulls = [sharpe(p, periods_per_year) for p in null_pnls]
    nulls = [s for s in nulls if not np.isnan(s)]
    if not nulls or np.isnan(real):
        return {"real_sharpe": real, "null_mean": np.nan, "p_value": np.nan,
                "n_nulls": len(nulls)}
    p = (sum(s >= real for s in nulls) + 1) / (len(nulls) + 1)
    return {"real_sharpe": real,
            "null_mean": float(np.mean(nulls)),
            "null_std": float(np.std(nulls)),
            "p_value": float(p),
            "n_nulls": len(nulls)}
