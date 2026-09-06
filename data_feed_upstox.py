"""
Real-time Nifty 50 5-min candle data via Upstox API v2.
Replaces yfinance entirely. Uses Analytics Token — no daily refresh needed.

Two endpoints used:
- Historical candle API: previous days data (for EMA seeding)
- Intraday candle API:   today's live 5-min candles

Both are merged into one clean DataFrame, partial candles filtered out.
"""

import requests
import pandas as pd
import os
from datetime import date, timedelta
from zoneinfo import ZoneInfo
import config

IST = ZoneInfo("Asia/Kolkata")
BASE_URL = "https://api.upstox.com/v2"

# Nifty 50 index instrument key — confirmed working in your Phase 5 test
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
ENCODED_KEY    = "NSE_INDEX%7CNifty%2050"  # URL-encoded for path params


def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError(
            "UPSTOX_ANALYTICS_TOKEN not set. Run: source ~/.nifty_bot_env"
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _parse_candles(response_json):
    """
    Upstox candle format: [timestamp, open, high, low, close, volume, oi]
    Converts to a clean IST-indexed DataFrame.
    """
    candles = response_json.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Upstox returns IST timestamps — localize or convert
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df.sort_index()
    return df


def _get_historical(from_date_str, to_date_str):
    """
    Fetch multi-day historical 5-min candles.
    from_date, to_date: 'YYYY-MM-DD'
    """
    url = (
        f"{BASE_URL}/historical-candle"
        f"/{ENCODED_KEY}/minute/5"
        f"/{to_date_str}/{from_date_str}"
    )
    resp = requests.get(url, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return _parse_candles(resp.json())


def _get_intraday():
    """Fetch today's live 5-min candles (updates in real time)."""
    url = f"{BASE_URL}/historical-candle/intraday/{ENCODED_KEY}/minute/5"
    resp = requests.get(url, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return _parse_candles(resp.json())


def get_recent_candles(lookback_days=7):
    """
    Returns fully-closed 5-min Nifty candles covering today + lookback_days
    of history. Partial (in-progress) candles are filtered out.

    lookback_days=7 gives enough history to seed EMA(9), EMA(15), ATR(14)
    without taking too long to fetch.
    """
    today     = date.today()
    from_date = today - timedelta(days=lookback_days + 3)  # +3 buffer for weekends
    yesterday = today - timedelta(days=1)

    # Fetch historical (previous days) — needed for EMA seeding
    try:
        hist_df = _get_historical(
            from_date.strftime("%Y-%m-%d"),
            yesterday.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"[Upstox] Historical fetch warning: {e}")
        hist_df = pd.DataFrame(columns=["open", "high", "low", "close"])

    # Fetch today's live intraday candles
    try:
        intra_df = _get_intraday()
    except Exception as e:
        print(f"[Upstox] Intraday fetch warning: {e}")
        intra_df = pd.DataFrame(columns=["open", "high", "low", "close"])

    # Merge historical + intraday
    parts = [df for df in [hist_df, intra_df] if not df.empty]
    if not parts:
        raise RuntimeError("No candle data returned from Upstox — check token and connectivity.")

    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Filter to only fully closed candles
    # A 5-min candle opened at T is closed at T + 5min
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    df = df[df.index + pd.Timedelta(minutes=5) <= now_ist]

    if df.empty:
        raise RuntimeError("No fully closed candles available yet — too early in session?")

    return df


def get_ltp(instrument_key):
    """
    Fetch current Last Traded Price for any instrument.
    Used by position monitor to track live option premium.
    """
    resp = requests.get(
        f"{BASE_URL}/market-quote/ltp",
        params={"instrument_key": instrument_key},
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    # Response: {"NSE_INDEX|Nifty 50": {"last_price": 24123.45}}
    key = list(data.keys())[0] if data else None
    if key:
        return data[key].get("last_price")
    return None
