"""information-flow network estimation.

two independent ways of asking "does X help predict Y":
  1. transfer entropy (model-free, catches nonlinear stuff)
  2. granger causality (linear VAR-based, the classical check)

an edge X -> Y exists if the TE survives a permutation test. granger
p-values are attached as edge attributes as a second opinion.
"""

import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests


def _discretize(s, n_bins=5):
    # quantile bins so each bin has roughly equal mass. equal-width bins
    # are terrible for returns (fat tails -> everything lands in 1-2 bins)
    s = np.asarray(s, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.nanquantile(s, qs)
    # nanquantile can return duplicate edges if the series is flat-ish;
    # nudge them apart so digitize doesn't collapse bins
    edges = np.array(edges)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return np.digitize(s, edges) - 1


def _entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def _joint_entropy(*series):
    # stacks discretized series, counts co-occurrences
    stacked = np.stack(series, axis=1)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    return _entropy(counts)


def transfer_entropy(x, y, n_bins=5):
    """TE(x -> y), k=l=1, histogram estimator. returns nats.

    TE = H(y+|y) - H(y+|y,x)
       = [H(y+,y) - H(y)] - [H(y+,y,x) - H(y,x)]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    xd = _discretize(x, n_bins)
    yd = _discretize(y, n_bins)

    y_next, y_cur, x_cur = yd[1:], yd[:-1], xd[:-1]

    h_ynext_y = _joint_entropy(y_next, y_cur)
    h_y = _entropy(np.unique(y_cur, return_counts=True)[1])
    h_ynext_yx = _joint_entropy(y_next, y_cur, x_cur)
    h_yx = _joint_entropy(y_cur, x_cur)

    te = (h_ynext_y - h_y) - (h_ynext_yx - h_yx)
    # numerical noise can make this tiny-negative; TE is >= 0 by definition
    return max(te, 0.0)


def te_pvalue(x, y, n_shuffles=200, n_bins=5, seed=42):
    """permutation test: shuffle x's time order, recompute TE.

    shuffling breaks the temporal coupling but keeps x's marginal
    distribution, so this is the right null for "x carries no timing info".
    this took forever to get right - the first version shuffled (x,y) pairs
    jointly which obviously preserves the coupling. facepalm.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    te_obs = transfer_entropy(x, y, n_bins=n_bins)
    n = len(x)
    count = 0
    for _ in range(n_shuffles):
        xs = x[rng.permutation(n)]
        if transfer_entropy(xs, y, n_bins=n_bins) >= te_obs:
            count += 1
    # +1 smoothing so we never report p=0
    return (count + 1) / (n_shuffles + 1), te_obs


def granger_pvalue(x, y, maxlag=3):
    """does x granger-cause y? returns min p-value across lags (ssr F-test)."""
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < 10 * maxlag:
        return np.nan
    try:
        # grangercausalitytests expects [y, x], tests x -> y
        res = grangercausalitytests(df[["y", "x"]], maxlag=maxlag, verbose=False)
        ps = [res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]
        return float(min(ps))
    except Exception:
        # perfect collinearity etc - just call it uninformative
        return np.nan


def build_network(returns, n_shuffles=200, te_alpha=0.05, n_bins=5,
                  with_granger=True, seed=42, verbose=True):
    """directed graph of information flow.

    edge i -> j means "i's returns carry significant timing info about j's
    next return" per the TE permutation test. weight = TE in nats.
    """
    tickers = list(returns.columns)
    G = nx.DiGraph()
    G.add_nodes_from(tickers)

    n_pairs = len(tickers) * (len(tickers) - 1)
    done = 0
    for i, src in enumerate(tickers):
        for j, dst in enumerate(tickers):
            if i == j:
                continue
            x = returns[src].values
            y = returns[dst].values
            p, te = te_pvalue(x, y, n_shuffles=n_shuffles, n_bins=n_bins,
                              seed=seed + done)
            done += 1
            if verbose and done % 50 == 0:
                print(f"  ... {done}/{n_pairs} pairs")
            if p < te_alpha:
                attr = {"te": te, "te_p": p}
                if with_granger:
                    attr["granger_p"] = granger_pvalue(x, y)
                G.add_edge(src, dst, **attr)

    if verbose:
        print(f"network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def out_strength(G):
    """sum of outgoing TE weights per node - the 'leadership' score."""
    return {n: sum(d["te"] for _, _, d in G.out_edges(n, data=True))
            for n in G.nodes()}


def top_leaders(G, k=5):
    """k nodes with the highest out-strength."""
    s = out_strength(G)
    return sorted(s, key=s.get, reverse=True)[:k]


# --- old approach, keeping for reference ---
# def build_network_corr(returns, thresh=0.3):
#     # started with lag-1 cross-correlation. problem: correlation is symmetric-ish
#     # and linear, so the "network" was basically just "everything correlates
#     # with SPY". TE actually finds direction. keeping this here so i remember
#     # why i switched.
#     ...
