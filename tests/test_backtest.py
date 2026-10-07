"""tests for the backtest. the important one is the look-ahead test."""

import numpy as np
import pandas as pd
import pytest

from diffuse.backtest import (
    leader_signals, apply_costs, walk_forward, sharpe, max_drawdown,
)
from diffuse.null_model import permuted_network, compare_to_null
import networkx as nx


def _toy_prices(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # L leads F by one bar, Z is pure noise. build returns first so
    # prices stay positive (learned that the hard way - cumsum prices
    # go negative and log returns blow up).
    rL = rng.normal(size=n) * 0.01
    rF = np.zeros(n)
    rF[1:] = 0.5 * rL[:-1]
    rF += rng.normal(size=n) * 0.005
    rZ = rng.normal(size=n) * 0.01
    rets = pd.DataFrame({"L": rL, "F": rF, "Z": rZ}, index=idx)
    return 100 * np.exp(rets.cumsum())


def test_no_lookahead():
    """the money test: shuffling future returns must not change past signals.

    if the signal at time t depends on anything after t, reversing the
    tail of the series changes early signals. it shouldn't.
    """
    prices = _toy_prices()
    rets = np.log(prices / prices.shift(1)).dropna()
    G = nx.DiGraph()
    G.add_nodes_from(["L", "F", "Z"])
    G.add_edge("L", "F", te=0.5)

    sig_full = leader_signals(rets, ["L"], G)

    # reverse the second half of returns - future info destroyed
    rets_rev = rets.copy()
    half = len(rets) // 2
    rets_rev.iloc[half:] = rets_rev.iloc[half:].iloc[::-1].values
    sig_rev = leader_signals(rets_rev, ["L"], G)

    # signals are computed bar-by-bar from contemporaneous leader returns,
    # so the first half must be identical
    pd.testing.assert_frame_equal(sig_full.iloc[:half], sig_rev.iloc[:half])


def test_costs_are_positive_and_scale_with_turnover():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    pos = pd.DataFrame({"A": [1.0] * 5 + [-1.0] * 5}, index=idx)
    c1 = apply_costs(pos, 2.0)
    c2 = apply_costs(pos, 4.0)
    assert (c1 >= 0).all()
    # one flip of size 2 -> turnover 2 at the flip bar
    assert c1.iloc[5] == 2 * 2.0 / 1e4
    pd.testing.assert_series_equal(c2, c1 * 2)


def test_flat_positions_no_costs():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    pos = pd.DataFrame({"A": [1.0] * 10}, index=idx)
    assert (apply_costs(pos, 5.0) == 0).all()


def test_walk_forward_runs_and_is_oos():
    prices = _toy_prices(n=600)
    pnl = walk_forward(prices, estimate_bars=200, test_bars=50,
                       n_leaders=1, n_shuffles=20, verbose=False)
    assert len(pnl) > 0
    # pnl index should only cover test windows, i.e. start after first est window
    rets = np.log(prices / prices.shift(1)).dropna()
    assert pnl.index.min() >= rets.index[200]


def test_permuted_network_preserves_structure():
    G = nx.DiGraph()
    G.add_edge("A", "B", te=0.3)
    G.add_edge("A", "C", te=0.1)
    G.add_edge("B", "C", te=0.2)
    H = permuted_network(G, seed=0)
    assert set(H.nodes()) == set(G.nodes())
    assert H.number_of_edges() == G.number_of_edges()
    assert sorted(d["te"] for _, _, d in H.edges(data=True)) == \
        sorted(d["te"] for _, _, d in G.edges(data=True))


def test_compare_to_null_sane():
    rng = np.random.default_rng(0)
    real = pd.Series(rng.normal(0.001, 0.01, size=500))
    nulls = [pd.Series(rng.normal(0, 0.01, size=500)) for _ in range(10)]
    out = compare_to_null(real, nulls, periods_per_year=252)
    assert 0 <= out["p_value"] <= 1
    assert out["n_nulls"] == 10


def test_sharpe_nan_on_flat():
    s = pd.Series([0.0] * 100)
    assert np.isnan(sharpe(s, 252))


def test_max_drawdown_known():
    # +10%, then -50%: trough 0.55 vs peak 1.1 -> dd = -0.5
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
