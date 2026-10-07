"""tests for the network estimation."""

import numpy as np
import pandas as pd

from diffuse.network import (
    transfer_entropy, te_pvalue, build_network, top_leaders, out_strength,
)


def coupled_series(n=2000, seed=0, lag=1, strength=0.6):
    """x drives y with a lag. y = strength * x[t-lag] + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(lag, n):
        y[t] = strength * x[t - lag] + rng.normal() * 0.5
    return x, y


def test_te_detects_direction():
    x, y = coupled_series()
    te_xy = transfer_entropy(x, y)
    te_yx = transfer_entropy(y, x)
    # x -> y should dominate. not even close on synthetic data.
    assert te_xy > te_yx
    assert te_xy > 0.05


def test_te_independent_series_is_small():
    rng = np.random.default_rng(7)
    x = rng.normal(size=3000)
    y = rng.normal(size=3000)
    assert transfer_entropy(x, y) < 0.02


def test_te_pvalue_significant_for_coupled():
    x, y = coupled_series()
    p, te = te_pvalue(x, y, n_shuffles=100, seed=1)
    assert p < 0.05, f"expected significance, got p={p}"


def test_te_pvalue_not_significant_for_independent():
    rng = np.random.default_rng(11)
    x = rng.normal(size=1500)
    y = rng.normal(size=1500)
    p, _ = te_pvalue(x, y, n_shuffles=100, seed=2)
    assert p > 0.05, f"false positive: p={p}"


def test_build_network_finds_planted_edge():
    # two blocks: A drives B, C is independent
    rng = np.random.default_rng(3)
    n = 2500
    a = rng.normal(size=n)
    b = np.concatenate([[0], 0.7 * a[:-1]]) + rng.normal(size=n) * 0.4
    c = rng.normal(size=n)
    rets = pd.DataFrame({"A": a, "B": b, "C": c})
    G = build_network(rets, n_shuffles=100, verbose=False)
    assert G.has_edge("A", "B"), "planted edge A->B not found"
    assert "A" in top_leaders(G, k=1)


def test_out_strength_symmetric_for_independent():
    rng = np.random.default_rng(5)
    rets = pd.DataFrame(rng.normal(size=(2000, 4)), columns=list("WXYZ"))
    G = build_network(rets, n_shuffles=50, verbose=False)
    # mostly no edges at all for pure noise
    assert G.number_of_edges() <= 2
