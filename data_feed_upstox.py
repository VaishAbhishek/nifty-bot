"""
Nifty 50 5-min candle data via Upstox API.

Two confirmed-working endpoints:
  Intraday (today):  GET /v3/historical-candle/intraday/{key}/minutes/5
  Historical (prev): GET /v2/historical-candle/{key}/1minute/{to}/{from}
                     → resampled to 5-min

Both merged into one clean DataFrame for the signal engine.
"""

import requests
import pandas as pd
import os
from datetime import date, timedelta
from zoneinfo import ZoneInfo

IST         = ZoneInfo("Asia/Kolkata")
BASE_V2     = "https://api.upstox.com/v2"
BASE_V3     = "https://api.upstox.com/v3"
ENCODED_KEY = "NSE_INDEX%7CNifty%2050"    # URL-encoded NSE_INDEX|Nifty 50


def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError(
            "UPSTOX_ANALYTICS_TOKEN not set. Run: source ~/.nifty_bot_env"
        )
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _last_trading_day():
    """Returns the most recent weekday — skips Saturday/Sunday."""
    d = date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _parse_candles(response_json, columns=None):
    """
    Parse Upstox candle list into a sorted IST-indexed DataFrame.
    Each candle: [timestamp, open, high, low, close, volume, oi]
    """
    candles = response_json.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    cols = columns or ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    df = pd.DataFrame(candles, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close"]].astype(float)
    return df.sort_index()


def _resample_to_5min(df_1min):
    """
    Resample 1-min OHLC → 5-min OHLC.
    NSE bar boundaries (9:15, 9:20 … 3:25, 3:30) align to 5-min grid naturally.
    """
    if df_1min.empty:
        return df_1min
    df_5min = df_1min.resample("5min").agg(
        open=("open",  "first"),
        high=("high",  "max"),
        low= ("low",   "min"),
        close=("close","last"),
    ).dropna()
    return df_5min.between_time("09:15", "15:30")


def _get_intraday_5min():
    """
    Fetch today's live 5-min candles via v3 intraday endpoint.
    Confirmed working: /v3/historical-candle/intraday/{key}/minutes/5
    Returns empty DataFrame gracefully outside market hours.
    """
    url = f"{BASE_V3}/historical-candle/intraday/{ENCODED_KEY}/minutes/5"
    try:
        resp = requests.get(url, headers=_headers(), timeout=15)
        if resp.status_code == 200:
            return _parse_candles(resp.json())
        # Outside market hours returns non-200 — that's fine, return empty
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    except Exception as e:
        print(f"[Upstox] Intraday fetch skipped: {e}")
        return pd.DataFrame(columns=["open", "high", "low", "close"])


def _get_historical_5min(lookback_days=7):
    """
    Fetch previous sessions' 1-min candles and resample to 5-min.
    Confirmed working: /v2/historical-candle/{key}/1minute/{to}/{from}
    """
    last_trade = _last_trading_day()
    from_date  = (date.today() - timedelta(days=lookback_days + 4)).strftime("%Y-%m-%d")
    to_date    = last_trade.strftime("%Y-%m-%d")
    url = f"{BASE_V2}/historical-candle/{ENCODED_KEY}/1minute/{to_date}/{from_date}"

    resp = requests.get(url, headers=_headers(), timeout=20)
    resp.raise_for_status()
    df_1min = _parse_candles(resp.json())

    if df_1min.empty:
        return df_1min
    return _resample_to_5min(df_1min)


def get_recent_candles(lookback_days=7):
    """
    Returns fully-closed 5-min Nifty candles covering the last lookback_days
    plus today's live session.

    Steps:
      1. Fetch historical 5-min candles (prev days via 1-min API + resample)
      2. Fetch today's live 5-min candles (v3 intraday endpoint, direct 5-min)
      3. Merge, deduplicate, drop any partial in-progress candle
    """
    # Step 1: historical (previous completed sessions)
    try:
        df_hist = _get_historical_5min(lookback_days)
    except Exception as e:
        raise RuntimeError(f"Upstox historical fetch failed: {e}")

    # Step 2: today's live 5-min candles
    df_intra = _get_intraday_5min()

    # Step 3: merge
    parts = [df for df in [df_hist, df_intra] if not df.empty]
    if not parts:
        raise RuntimeError(
            "No candle data from Upstox. "
            "Check UPSTOX_ANALYTICS_TOKEN and market hours."
        )

    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Keep only NSE market hours across all days
    df = df.between_time("09:15", "15:30")

    # Drop partial (in-progress) candle — bar at T is closed at T+5min
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    df = df[df.index + pd.Timedelta(minutes=5) <= now_ist]

    if df.empty:
        raise RuntimeError(
            "No fully closed 5-min candles yet. "
            "Market may not have opened, or today has no intraday data yet."
        )

    return df


def get_ltp(instrument_key):
    """Fetch current Last Traded Price for any Upstox instrument key."""
    resp = requests.get(
        f"{BASE_V2}/market-quote/ltp",
        params={"instrument_key": instrument_key},
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    key  = list(data.keys())[0] if data else None
    return data[key].get("last_price") if key else None