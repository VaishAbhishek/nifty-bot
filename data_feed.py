"""
Free Nifty 50 index candle data via yfinance.
This covers the underlying price data our signal engine needs.
It does NOT provide option premiums - see option_chain.py for that.
"""

import yfinance as yf
import pandas as pd
import config


def get_recent_candles(period="5d"):
    """
    Returns a DataFrame of recent 5-min Nifty candles, indexed by IST datetime,
    with columns: open, high, low, close.

    yfinance only allows up to ~60 days of 5m history - 5d is plenty for our
    rolling EMA/ATR/angle calculations.
    """
    df = yf.download(
        config.UNDERLYING_TICKER,
        period=period,
        interval=config.CANDLE_INTERVAL,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise RuntimeError(
            "yfinance returned no data. Check your internet connection or "
            "whether '^NSEI' is still the correct ticker."
        )

    # yfinance sometimes returns MultiIndex columns when fetching a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]

    # Convert to IST for session/force-close logic
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")

    return df
