"""fetching + cleaning price data.

all the point-in-time discipline lives here: one clean panel of prices,
no forward filling across the as-of boundary, tickers with too much
missing data get dropped instead of patched up.
"""

import pandas as pd
import numpy as np
import yfinance as yf

# liquid ETFs, all trade basically every minute the market is open.
# kept it to ETFs on purpose - single names have way more gaps and
# corporate action noise. might add a few mega-cap stocks later.
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
    "EFA", "EEM", "TLT", "GLD", "USO",
]


def fetch_prices(tickers=None, interval="1h", period="3mo", min_coverage=0.9):
    """download close prices, return a clean dataframe.

    columns are tickers, index is tz-aware timestamps sorted ascending.
    tickers missing more than (1 - min_coverage) of bars are dropped.
    """
    tickers = tickers or UNIVERSE

    # yfinance auto_adjust=False keeps the raw closes; we want total-return-ish
    # prices though, so adjusted close is actually what we want here
    raw = yf.download(
        tickers, interval=interval, period=period,
        auto_adjust=True, progress=False, threads=True,
    )
    # yf returns multiindex columns when >1 ticker, plain when 1. annoying.
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]
        name = tickers if isinstance(tickers, str) else tickers[0]
        closes.columns = [name]

    closes = closes.sort_index()
    closes.index = pd.to_datetime(closes.index, utc=True)

    # drop the thin stuff instead of interpolating garbage
    coverage = closes.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = [t for t in closes.columns if t not in keep]
    if dropped:
        print(f"dropping {len(dropped)} tickers with < {min_coverage:.0%} coverage: {dropped}")
    closes = closes[keep]

    # small gaps only (a bar or two). anything bigger and the ticker
    # would've been dropped above.
    closes = closes.ffill(limit=2)

    n_missing = closes.isna().sum().sum()
    if n_missing:
        # shouldn't happen after the ffill, but just in case
        closes = closes.dropna()
        print(f"dropped {n_missing} remaining NaN rows")

    return closes


def to_returns(prices, kind="log"):
    """bar-to-bar returns. log by default, keeps things additive."""
    if kind == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")


def validate_panel(prices):
    """sanity checks before anything touches the data. raises on problems."""
    assert isinstance(prices.index, pd.DatetimeIndex), "index must be datetimes"
    assert prices.index.is_monotonic_increasing, "index not sorted - fix this upstream"
    assert not prices.isna().all(axis=1).any(), "fully-empty bars in panel"
    dupes = prices.index.duplicated().sum()
    assert dupes == 0, f"{dupes} duplicated timestamps"
    # weekend / holiday gaps are fine, but a >5 day hole means something broke
    gaps = prices.index.to_series().diff().dropna()
    max_gap = gaps.max()
    assert max_gap < pd.Timedelta(days=6), f"suspicious gap in data: {max_gap}"
    return True
