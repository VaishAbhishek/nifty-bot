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

    IMPORTANT: yfinance includes the current in-progress candle as the last row.
    We filter it out to ensure only fully closed candles are processed — matching
    how Pine Script handles calc_on_every_tick=false (bar close only).
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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")

    # Drop the last candle if it's still in progress
    # A 5-min candle opened at T is closed at T+5min
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    candle_close_times = df.index + pd.Timedelta(minutes=5)
    df = df[candle_close_times <= now_ist]

    if df.empty:
        raise RuntimeError("No fully closed candles available yet.")

    return df