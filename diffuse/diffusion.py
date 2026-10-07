"""shock propagation on the estimated network.

SIR-style simulation: seed a shock at one node, let it spread along directed
edges with probability proportional to edge weight, nodes recover over time.
the point is to get *simulated* information travel times and check them
against *empirical* lead-lag from the return series. if the network is real,
those two should agree.
"""

import numpy as np
import networkx as nx
from scipy.stats import spearmanr


def simulate_shock(G, seed_node, n_steps=60, beta=0.5, gamma=0.1,
                   n_sims=200, seed=42):
    """run SIR dynamics, return mean first-infection time per node.

    beta scales with edge weight (stronger info link = faster spread),
    gamma is the per-step recovery probability.
    states: 0=susceptible, 1=infected, 2=recovered
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    seed_i = idx[seed_node]

    # normalize edge weights to [0,1] so beta means something comparable
    # across different networks
    weights = np.array([d["te"] for _, _, d in G.edges(data=True)])
    wmax = weights.max() if len(weights) else 1.0

    # adjacency as successor lists with normalized weights
    succ = {n: [(v, d["te"] / wmax) for _, v, d in G.out_edges(n, data=True)]
            for n in nodes}

    infect_times = np.full((n_sims, len(nodes)), np.nan)
    for s in range(n_sims):
        state = np.zeros(len(nodes), dtype=int)
        state[seed_i] = 1
        infect_times[s, seed_i] = 0
        for t in range(1, n_steps + 1):
            newly = []
            for k, n in enumerate(nodes):
                if state[k] != 1:
                    continue
                for v, w in succ[n]:
                    j = idx[v]
                    if state[j] == 0 and rng.random() < beta * w:
                        newly.append(j)
                if rng.random() < gamma:
                    state[k] = 2  # recovered
            for j in newly:
                if state[j] == 0:
                    state[j] = 1
                    infect_times[s, j] = t
            if not np.any(state == 1):
                break

    with np.errstate(all="ignore"):
        mean_times = np.nanmean(infect_times, axis=0)
    return {n: mean_times[idx[n]] for n in nodes}


def travel_time_matrix(G, **kwargs):
    """mean infection time from every node to every other node."""
    nodes = list(G.nodes())
    mat = {src: simulate_shock(G, src, **kwargs) for src in nodes}
    return mat


def empirical_leadlag(returns, max_lag=10):
    """for each ordered pair (i, j): the lag at which i best predicts j.

    lag* = argmax_l corr(r_i[t - lag], r_j[t]). positive lag means i leads j.
    not proud of this loop but n=20 tickers so it's fine.
    """
    tickers = list(returns.columns)
    out = {}
    arr = returns.values
    for a, src in enumerate(tickers):
        for b, dst in enumerate(tickers):
            if a == b:
                continue
            best_lag, best_c = 0, -np.inf
            xs = arr[:, a]
            ys = arr[:, b]
            for lag in range(max_lag + 1):
                if lag == 0:
                    c = np.corrcoef(xs, ys)[0, 1]
                else:
                    c = np.corrcoef(xs[:-lag], ys[lag:])[0, 1]
                if not np.isnan(c) and c > best_c:
                    best_c, best_lag = c, lag
            out[(src, dst)] = (best_lag, best_c)
    return out


def validate_against_leadlag(G, returns, max_lag=10, **sim_kwargs):
    """spearman correlation between simulated travel times and empirical lags.

    this is the "is the network real" check for the diffusion part: pairs
    that are close in the simulated cascade should also show up as
    short-lag leaders empirically. returns (rho, pvalue, n_pairs).
    """
    sim = travel_time_matrix(G, **sim_kwargs)
    emp = empirical_leadlag(returns, max_lag=max_lag)

    xs, ys = [], []
    for (src, dst), (lag, _) in emp.items():
        t = sim[src][dst]
        if not np.isnan(t):
            xs.append(t)
            ys.append(lag)
    if len(xs) < 10:
        return np.nan, np.nan, len(xs)
    rho, p = spearmanr(xs, ys)
    return float(rho), float(p), len(xs)
