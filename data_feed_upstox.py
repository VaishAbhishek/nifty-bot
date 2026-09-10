"""
Real-time Nifty 50 candle data via Upstox API v2.

Key findings from diagnostics:
- v2 historical candle API supports 1minute interval (confirmed working)
- The historical API only returns COMPLETED sessions — today's live data is absent
- Today's live data is fetched via market-quote/ohlc with interval=I1 (1-min bar)
  and accumulated in memory by main.py polling every 30 seconds

Architecture:
  Historical 1-min (API) + Today's 1-min (_intraday_bars dict) → resample to 5-min
"""

import requests
import pandas as pd
import os
from datetime import date, timedelta
from zoneinfo import ZoneInfo
import config

IST         = ZoneInfo("Asia/Kolkata")
BASE_URL    = "https://api.upstox.com/v2"
ENCODED_KEY = "NSE_INDEX%7CNifty%2050"   # URL-encoded NSE_INDEX|Nifty 50

# ── Intraday accumulator ──────────────────────────────────────────────────────
# main.py polls Upstox I1 OHLC every 30s and calls update_intraday_bar().
# This builds today's 1-min bars in memory throughout the session.
_intraday_bars = {}    # {pd.Timestamp (IST, tz-aware): {'open','high','low','close'}}
_intraday_date = None  # date the bars belong to (resets on new session)


def update_intraday_bar(bar_time, ohlc):
    """
    Store/update a 1-min bar from the live I1 OHLC feed.
    bar_time: pd.Timestamp, minute-floored, IST timezone-aware
    ohlc:     dict with keys open / high / low / close
    """
    global _intraday_bars, _intraday_date
    today = bar_time.date()
    if _intraday_date != today:
        _intraday_bars = {}
        _intraday_date = today
    _intraday_bars[bar_time] = ohlc


# ── Internal helpers ──────────────────────────────────────────────────────────

def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError("UPSTOX_ANALYTICS_TOKEN not set. Run: source ~/.nifty_bot_env")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _last_trading_day():
    """Most recent weekday — skips Saturday/Sunday."""
    d = date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _parse_candles_1min(response_json):
    """Parse Upstox candle response into a sorted IST-indexed DataFrame."""
    candles = response_json.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
    )
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
    Resample 1-min OHLC to 5-min OHLC.
    NSE bar boundaries (9:15, 9:20 … 3:25, 3:30) align naturally to 5-min grid.
    """
    if df_1min.empty:
        return df_1min
    df_5min = df_1min.resample("5min").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low= ("low",    "min"),
        close=("close", "last"),
    ).dropna()
    return df_5min.between_time("09:15", "15:30")


# ── Public API ────────────────────────────────────────────────────────────────

def get_recent_candles(lookback_days=7):
    """
    Returns fully-closed 5-min Nifty candles covering the last lookback_days.

    Merges two sources:
      1. Historical 1-min candles from Upstox API (completed sessions only)
      2. Today's live 1-min bars accumulated in _intraday_bars by main.py polling
    Both are resampled to 5-min before returning.
    Partial (in-progress) 5-min candles are always dropped.
    """
    # ── Source 1: historical (completed sessions) ─────────────────────────────
    last_trade = _last_trading_day()
    from_date  = (date.today() - timedelta(days=lookback_days + 4)).strftime("%Y-%m-%d")
    to_date    = last_trade.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/historical-candle/{ENCODED_KEY}/1minute/{to_date}/{from_date}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=20)
        resp.raise_for_status()
        df_hist = _parse_candles_1min(resp.json())
    except Exception as e:
        raise RuntimeError(f"Upstox historical fetch failed: {e}\nURL: {url}")

    # ── Source 2: today's live bars from intraday accumulator ─────────────────
    if _intraday_bars:
        rows = [
            {"timestamp": ts,
             "open":  v["open"], "high": v["high"],
             "low":   v["low"],  "close": v["close"]}
            for ts, v in _intraday_bars.items()
        ]
        df_intra = pd.DataFrame(rows).set_index("timestamp")
        df_intra.index = pd.DatetimeIndex(df_intra.index)
        if df_intra.index.tz is None:
            df_intra.index = df_intra.index.tz_localize("Asia/Kolkata")
        df_intra = df_intra[["open", "high", "low", "close"]].astype(float)
        df_intra = df_intra.sort_index().between_time("09:15", "15:30")
    else:
        df_intra = pd.DataFrame(columns=["open", "high", "low", "close"])

    # ── Merge → resample → filter ─────────────────────────────────────────────
    parts = [df for df in [df_hist, df_intra] if not df.empty]
    if not parts:
        raise RuntimeError("No candle data available from Upstox.")

    df_1min = pd.concat(parts)
    df_1min = df_1min[~df_1min.index.duplicated(keep="last")].sort_index()

    df_5min = _resample_to_5min(df_1min)
    if df_5min.empty:
        raise RuntimeError("No 5-min candles after resampling.")

    # Drop any partial (in-progress) 5-min candle
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    df_5min = df_5min[df_5min.index + pd.Timedelta(minutes=5) <= now_ist]

    if df_5min.empty:
        raise RuntimeError(
            "No fully closed 5-min candles yet. "
            "If market is open, I1 polling must be running in main.py."
        )
    return df_5min


def get_ltp(instrument_key):
    """Fetch current Last Traded Price for any Upstox instrument key."""
    resp = requests.get(
        f"{BASE_URL}/market-quote/ltp",
        params={"instrument_key": instrument_key},
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    key  = list(data.keys())[0] if data else None
    return data[key].get("last_price") if key else None