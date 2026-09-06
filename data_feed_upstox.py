"""
Real-time Nifty 50 5-min candle data via Upstox API.
Tries v3 first (newer format), falls back to v2.

Key fixes:
- Uses last trading weekday as to_date (avoids Saturday/Sunday 404)
- Handles intraday gracefully on weekends/holidays
- Tries both API versions automatically
"""

import requests
import pandas as pd
import os
from datetime import date, timedelta
from zoneinfo import ZoneInfo
import config

IST = ZoneInfo("Asia/Kolkata")

# Upstox instrument key for Nifty 50 index
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
ENCODED_KEY    = "NSE_INDEX%7CNifty%2050"


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


def _last_trading_day(from_date=None):
    """
    Returns the most recent weekday (Mon-Fri).
    If from_date is None, starts from today.
    Needed because Upstox rejects Saturday/Sunday as to_date.
    """
    d = from_date or date.today()
    # weekday(): 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _is_market_open():
    """Returns True only during NSE trading hours on weekdays."""
    from datetime import datetime, time as dtime
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)


def _parse_candles(response_json):
    """
    Upstox candle format: [timestamp, open, high, low, close, volume, oi]
    Handles both v2 and v3 response structures.
    """
    # v2 structure: data.candles
    # v3 structure: data.candles (same key, different nesting sometimes)
    data = response_json.get("data", {})
    if isinstance(data, dict):
        candles = data.get("candles", [])
    else:
        candles = []

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


def _fetch_with_fallback(url_v3, url_v2):
    """Try v3 first, fall back to v2 if it fails."""
    for url in [url_v3, url_v2]:
        try:
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            continue
    # Both failed — raise the v2 error for diagnostics
    resp = requests.get(url_v2, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


def _get_historical(from_date_str, to_date_str):
    """Fetch multi-day historical 5-min candles."""
    url_v3 = (
        f"https://api.upstox.com/v3/historical-candle"
        f"/{ENCODED_KEY}/5minute/{to_date_str}/{from_date_str}"
    )
    url_v2 = (
        f"https://api.upstox.com/v2/historical-candle"
        f"/{ENCODED_KEY}/minute/5/{to_date_str}/{from_date_str}"
    )
    return _parse_candles(_fetch_with_fallback(url_v3, url_v2))


def _get_intraday():
    """
    Fetch today's live 5-min candles.
    Returns empty DataFrame gracefully on weekends or outside market hours.
    """
    if not _is_market_open():
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    url_v3 = (
        f"https://api.upstox.com/v3/historical-candle"
        f"/intraday/{ENCODED_KEY}/5minute"
    )
    url_v2 = (
        f"https://api.upstox.com/v2/historical-candle"
        f"/intraday/{ENCODED_KEY}/minute/5"
    )
    try:
        return _parse_candles(_fetch_with_fallback(url_v3, url_v2))
    except Exception as e:
        print(f"[Upstox] Intraday fetch skipped: {e}")
        return pd.DataFrame(columns=["open", "high", "low", "close"])


def get_recent_candles(lookback_days=7):
    """
    Returns fully-closed 5-min Nifty candles for the last lookback_days
    of trading days, merged with today's intraday data if market is open.
    Partial (in-progress) candles are always filtered out.
    """
    today          = date.today()
    last_trade_day = _last_trading_day()                    # last weekday
    from_date      = today - timedelta(days=lookback_days + 4)  # +4 buffer for weekends

    # Historical: from N days ago up to last trading weekday
    try:
        hist_df = _get_historical(
            from_date.strftime("%Y-%m-%d"),
            last_trade_day.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"[Upstox] Historical fetch warning: {e}")
        hist_df = pd.DataFrame(columns=["open", "high", "low", "close"])

    # Intraday: today's live candles (empty if weekend/holiday)
    intra_df = _get_intraday()

    # Merge
    parts = [df for df in [hist_df, intra_df] if not df.empty]
    if not parts:
        raise RuntimeError(
            "No candle data from Upstox. "
            "If market is open, check UPSTOX_ANALYTICS_TOKEN. "
            "If weekend/holiday, this is expected — bot sleeps until Monday."
        )

    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Keep only last lookback_days worth
    cutoff = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=lookback_days + 4)
    df = df[df.index >= cutoff]

    # Drop partial (in-progress) candles — candle at T closes at T+5min
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    df = df[df.index + pd.Timedelta(minutes=5) <= now_ist]

    if df.empty:
        raise RuntimeError(
            "No fully closed candles available. "
            "Market might not have opened yet today."
        )

    return df


def get_ltp(instrument_key):
    """Fetch current Last Traded Price for any instrument key."""
    resp = requests.get(
        "https://api.upstox.com/v2/market-quote/ltp",
        params={"instrument_key": instrument_key},
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    key  = list(data.keys())[0] if data else None
    return data[key].get("last_price") if key else None