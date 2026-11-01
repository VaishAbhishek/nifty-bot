"""
Nifty 50 5-min candle data via Upstox API.

Architecture (simple):
  1. Yesterday's 5-min candles  → v2 1-min API + resample to 5-min
     (needed to seed EMA/ATR correctly — without this, EMAs are wrong at session open)
  2. Today's live 5-min candles → v3 intraday/minutes/5 (direct, no resample)
     (returns empty outside 9:15–3:30 IST, that's fine — handled gracefully)

Both merged into one DataFrame. Partial/in-progress candles always dropped.
"""

import requests
import pandas as pd
import os
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import config
import market_calendar

IST        = ZoneInfo("Asia/Kolkata")
BASE_V2     = "https://api.upstox.com/v2"
BASE_V3     = "https://api.upstox.com/v3"
ENCODED_KEY = "NSE_INDEX%7CNifty%2050"


def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError(
            "UPSTOX_ANALYTICS_TOKEN not set. Run: source ~/.nifty_bot_env"
        )
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _today_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata").date()


def _last_trading_day():
    """Most recent trading day before today (history excludes today's session)."""
    return market_calendar.prev_trading_day(_today_ist())


def _prev_trading_day(from_date):
    """Trading day before from_date (skips weekends and NSE holidays)."""
    return market_calendar.prev_trading_day(from_date)


def _parse_candles(response_json):
    """Parse Upstox candle list into a sorted IST-indexed DataFrame."""
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
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def _resample_to_5min(df_1min):
    """Resample 1-min OHLC → 5-min OHLC aligned to NSE bar boundaries."""
    if df_1min.empty:
        return df_1min
    df_5min = df_1min.resample("5min").agg(
        open=("open",  "first"),
        high=("high",  "max"),
        low= ("low",   "min"),
        close=("close","last"),
    ).dropna()
    return df_5min.between_time("09:15", "15:30")


def _get_history_candles():
    """
    Fetch last 10 trading days of 1-min candles and resample to 5-min.

    WHY 10 DAYS (not 2):
    With only 2 days, EMA(9) and EMA(15) are seeded from ~150 bars.
    TradingView seeds from months of data. The resulting EMA offset causes:
      - Phantom micro-crossovers (fast briefly above slow while trending down)
      - fastEMA values slightly below TradingView's → retest threshold never triggers
    10 days = ~750 bars. EMAs converge to TradingView values within the first
    20-30 bars of each session, well before any trade signal can fire.
    """
    last_trade = _last_trading_day()
    # Walk back 10 trading days
    start = last_trade
    for _ in range(9):
        start = _prev_trading_day(start)

    to_date   = last_trade.strftime("%Y-%m-%d")
    from_date = start.strftime("%Y-%m-%d")
    url = f"{BASE_V2}/historical-candle/{ENCODED_KEY}/1minute/{to_date}/{from_date}"

    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    df_1min = _parse_candles(resp.json())
    return _resample_to_5min(df_1min)


def _get_today_candles():
    """
    Fetch today's live 5-min candles via v3 intraday endpoint.
    Returns empty DataFrame outside 9:15–3:30 IST — handled gracefully.
    Confirmed working: /v3/historical-candle/intraday/{key}/minutes/5
    """
    url = f"{BASE_V3}/historical-candle/intraday/{ENCODED_KEY}/minutes/5"
    try:
        resp = requests.get(url, headers=_headers(), timeout=15)
        if resp.status_code == 200:
            return _parse_candles(resp.json())
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    except Exception as e:
        print(f"[Upstox] Intraday fetch skipped: {e}")
        return pd.DataFrame(columns=["open", "high", "low", "close"])


_hist_cache = {"day": None, "df": None}


def _get_history_cached():
    """
    History (previous 10 trading days) doesn't change during the session, so
    fetch it once per day instead of on every 30-second poll.
    """
    today = _today_ist()
    if _hist_cache["day"] != today or _hist_cache["df"] is None:
        _hist_cache["df"] = _get_history_candles()
        _hist_cache["day"] = today
    return _hist_cache["df"]


def get_recent_candles():
    """
    Returns fully-closed 5-min Nifty candles for EMA seeding + today's live session.

    Outside market hours: returns previous sessions' candles only.
    During market hours:  returns history + today's candles merged.
    Partial in-progress candles are always dropped.
    """
    # Historical candles for EMA seeding (10 trading days, cached per day)
    try:
        df_hist = _get_history_cached()
    except Exception as e:
        raise RuntimeError(f"Historical candle fetch failed: {e}")

    # Today (live session — empty if market closed, that's fine)
    df_today = _get_today_candles()

    # Merge
    parts = [df for df in [df_hist, df_today] if not df.empty]
    if not parts:
        raise RuntimeError("No candle data available from Upstox.")

    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time("09:15", "15:30")

    # Drop partial in-progress candle (bar at T closes at T+5min)
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    df = df[df.index + pd.Timedelta(minutes=5) <= now_ist]

    if df.empty:
        raise RuntimeError(
            "No fully closed 5-min candles. "
            "Market not open yet or today is a holiday."
        )
    return df


NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


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


def get_nifty_ltp():
    """
    Fetch the current Nifty 50 index Last Traded Price in real-time.
    Used for intrabar EMA-touch detection — much faster than polling
    the intraday candle endpoint, which only updates at bar boundaries.
    Returns None on any error (caller should handle gracefully).
    """
    try:
        return get_ltp(NIFTY_INSTRUMENT_KEY)
    except Exception:
        return None


def get_vix():
    """Current India VIX, or None if unavailable."""
    try:
        v = get_ltp(config.VIX_INSTRUMENT_KEY)
        return float(v) if v else None
    except Exception:
        return None