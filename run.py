"""end-to-end pipeline: fetch -> network -> diffusion check -> backtest -> null.

usage:
    python run.py                    # full run, hourly bars, 3 months
    python run.py --quick            # small universe, fewer shuffles (smoke test)
    python run.py --interval 1d --period 2y   # daily bars, longer history

results land in results/: network.png, diffusion.png, results.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from diffuse import data as D
from diffuse import network as N
from diffuse import diffusion as DI
from diffuse import backtest as B
from diffuse import null_model as NM

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def periods_per_year(interval):
    # rough trading-calendar mapping. hourly = 6.5h * 252d.
    return {"1h": 252 * 6.5, "1d": 252, "15m": 252 * 6.5 * 4}.get(interval, 252)


def plot_network(G, path):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    strengths = N.out_strength(G)
    sizes = [300 + 3000 * strengths.get(n, 0) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="steelblue", alpha=0.8)
    widths = [1 + 4 * d["te"] for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title("information-flow network (edge width = transfer entropy)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="3mo")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-leaders", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--n-nulls", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tickers = D.UNIVERSE if not args.quick else D.UNIVERSE[:8]
    n_shuffles = 50 if args.quick else 200

    print("== 1. fetching data ==")
    prices = D.fetch_prices(tickers, interval=args.interval, period=args.period)
    D.validate_panel(prices)
    rets = D.to_returns(prices)
    print(f"{len(prices)} bars x {len(prices.columns)} tickers")

    print("== 2. estimating network ==")
    G = N.build_network(rets, n_shuffles=n_shuffles)
    leaders = N.top_leaders(G, k=args.n_leaders)
    print("leaders:", leaders)
    plot_network(G, os.path.join(RESULTS, "network.png"))

    print("== 3. diffusion vs empirical lead-lag ==")
    rho, p, n_pairs = DI.validate_against_leadlag(G, rets, n_sims=100)
    print(f"spearman(sim travel time, empirical lag) = {rho:.3f} (p={p:.3f}, n={n_pairs})")

    print("== 4. walk-forward backtest ==")
    ppy = periods_per_year(args.interval)
    # scale windows to the bar frequency: ~30d est / ~5d test
    bars_per_day = {"1h": 6.5, "1d": 1, "15m": 26}.get(args.interval, 6.5)
    est_bars = int(30 * bars_per_day)
    test_bars = int(5 * bars_per_day)
    pnl = B.walk_forward(prices, estimate_bars=est_bars, test_bars=test_bars,
                         n_leaders=args.n_leaders, cost_bps=args.cost_bps,
                         n_shuffles=n_shuffles)
    summary = B.summarize(pnl, ppy, args.cost_bps)
    print(summary)

    print("== 5. null model ==")
    null_pnls = NM.null_walk_forward(
        prices, n_perms=args.n_nulls, estimate_bars=est_bars,
        test_bars=test_bars, n_leaders=args.n_leaders,
        cost_bps=args.cost_bps, n_shuffles=n_shuffles)
    cmp = NM.compare_to_null(pnl, null_pnls, ppy)
    print(cmp)

    out = {
        "tickers": list(prices.columns),
        "interval": args.interval,
        "period": args.period,
        "n_bars": len(prices),
        "leaders": leaders,
        "n_edges": G.number_of_edges(),
        "diffusion_validation": {"spearman_rho": rho, "p": p, "n_pairs": n_pairs},
        "backtest": summary,
        "null_comparison": cmp,
    }
    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote results/results.json")


if __name__ == "__main__":
    main()
