"""
Real-time Nifty 50 candle data via Upstox API v2.

Key finding from diagnostics:
- v2 historical candle API supports 1minute and day intervals (NOT 5minute)
- Solution: fetch 1-minute candles and resample to 5-minute using pandas
- This produces mathematically identical OHLC to TradingView's 5-min bars

Confirmed working endpoint:
GET /v2/historical-candle/{instrumentKey}/1minute/{to_date}/{from_date}
"""

import requests
import pandas as pd
import os
from datetime import date, timedelta
from zoneinfo import ZoneInfo
import config

IST          = ZoneInfo("Asia/Kolkata")
BASE_URL     = "https://api.upstox.com/v2"
ENCODED_KEY  = "NSE_INDEX%7CNifty%2050"   # NSE_INDEX|Nifty 50, URL-encoded


def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError("UPSTOX_ANALYTICS_TOKEN not set. Run: source ~/.nifty_bot_env")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _last_trading_day():
    """Returns the most recent weekday (skips Saturday/Sunday)."""
    d = date.today()
    while d.weekday() >= 5:   # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def _parse_candles_1min(response_json):
    """Parse Upstox 1-minute candle response into a clean DataFrame."""
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
    df = df.sort_index()
    return df


def _resample_to_5min(df_1min):
    """
    Resample 1-minute OHLC candles to 5-minute OHLC candles.

    NSE 5-min bar boundaries: 9:15, 9:20, 9:25 ... 3:25, 3:30
    9:15 is divisible by 5 (555 min / 5 = 111) so no offset needed.

    Resampling rule:
      open  = first 1-min open in the 5-min window
      high  = max of all 1-min highs
      low   = min of all 1-min lows
      close = last 1-min close
    """
    if df_1min.empty:
        return df_1min

    df_5min = df_1min.resample("5min").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low= ("low",    "min"),
        close=("close", "last"),
    ).dropna()

    # Keep only NSE market hours (9:15 AM – 3:30 PM)
    df_5min = df_5min.between_time("09:15", "15:30")
    return df_5min


def get_recent_candles(lookback_days=7):
    """
    Returns fully-closed 5-min Nifty candles for the last lookback_days.
    Fetches 1-minute data from Upstox and resamples to 5-minute.
    Partial (in-progress) candles are always filtered out.
    """
    last_trade  = _last_trading_day()
    from_date   = (date.today() - timedelta(days=lookback_days + 4)).strftime("%Y-%m-%d")
    to_date     = last_trade.strftime("%Y-%m-%d")

    # Single API call covers both historical and today's data
    url = f"{BASE_URL}/historical-candle/{ENCODED_KEY}/1minute/{to_date}/{from_date}"

    try:
        resp = requests.get(url, headers=_headers(), timeout=20)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Upstox candle fetch failed: {e}\nURL: {url}")

    df_1min = _parse_candles_1min(resp.json())
    if df_1min.empty:
        raise RuntimeError("Upstox returned empty candle data — market might be closed.")

    # If market is open today, fetch today's 1-min candles separately
    # (historical endpoint may not include today's live data)
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    today_ist = date.today()
    market_open  = pd.Timestamp(today_ist).tz_localize("Asia/Kolkata").replace(hour=9, minute=15)
    market_close = pd.Timestamp(today_ist).tz_localize("Asia/Kolkata").replace(hour=15, minute=30)
    is_weekday   = today_ist.weekday() < 5

    if is_weekday and market_open <= now_ist <= market_close:
        today_str = today_ist.strftime("%Y-%m-%d")
        url_today = f"{BASE_URL}/historical-candle/{ENCODED_KEY}/1minute/{today_str}/{today_str}"
        try:
            resp_today = requests.get(url_today, headers=_headers(), timeout=15)
            if resp_today.status_code == 200:
                df_today_1min = _parse_candles_1min(resp_today.json())
                if not df_today_1min.empty:
                    df_1min = pd.concat([df_1min, df_today_1min])
                    df_1min = df_1min[~df_1min.index.duplicated(keep="last")]
                    df_1min = df_1min.sort_index()
        except Exception as e:
            print(f"[Upstox] Today's intraday fetch skipped: {e}")

    # Resample 1-min → 5-min
    df_5min = _resample_to_5min(df_1min)

    if df_5min.empty:
        raise RuntimeError("No 5-min candles after resampling — check market hours.")

    # Drop partial (in-progress) candles — candle at T closes at T+5min
    df_5min = df_5min[df_5min.index + pd.Timedelta(minutes=5) <= now_ist]

    if df_5min.empty:
        raise RuntimeError("No fully closed 5-min candles yet — too early in session?")

    return df_5min


def get_ltp(instrument_key):
    """Fetch current Last Traded Price for any instrument key."""
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