"""
Real Nifty weekly option chain via Upstox API v2.
Replaces the Black-Scholes synthetic estimate entirely.
Returns actual LTP (Last Traded Price) from NSE — the real premium
you'd pay if you placed a market order right now.

Uses Analytics Token — no daily refresh needed.
"""

import requests
import os
import datetime
from zoneinfo import ZoneInfo
import config

IST = ZoneInfo("Asia/Kolkata")
BASE_URL = "https://api.upstox.com/v2"
NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError("UPSTOX_ANALYTICS_TOKEN not set in environment")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _try_expiry_candidates():
    """
    Generate candidate expiry dates to try, in chronological order.
    Covers today through the next 14 calendar days.
    NSE weekly expiry is normally Tuesday; on holidays it shifts to Monday
    (or occasionally Wednesday). Rather than maintaining a holiday calendar,
    we probe dates and take the first one with live LTP data.
    """
    today = datetime.date.today()
    # Weekdays only (Mon-Fri) from today through next 14 days
    candidates = []
    for offset in range(15):
        d = today + datetime.timedelta(days=offset)
        if d.weekday() < 5:   # Mon=0 ... Fri=4
            candidates.append(d.strftime("%Y-%m-%d"))
    return candidates


def discover_nearest_expiry(direction=1):
    """
    Finds the nearest Nifty weekly expiry that actually has live LTP data.

    Handles all cases automatically:
      - Normal Tuesday expiry
      - Holiday shift to Monday
      - Holiday shift to Wednesday
      - Weekend / after-hours (returns None if market is fully closed)

    Returns (expiry_date_str, candidates_list) or raises RuntimeError.
    """
    side_key = "call_options" if direction == 1 else "put_options"

    for expiry_date in _try_expiry_candidates():
        try:
            data = fetch_option_chain(expiry_date)
        except Exception:
            continue

        # Check if this expiry has any strikes with real LTP
        for row in data.get("data", []):
            side = row.get(side_key, {})
            if side and side.get("market_data", {}).get("ltp", 0) > 0:
                return expiry_date   # found a live expiry

    raise RuntimeError(
        "Could not find any Nifty weekly expiry with live LTP data. "
        "Market is likely closed or Upstox data is unavailable."
    )


def get_current_week_expiry():
    """
    Returns the nearest upcoming expiry as a 'YYYY-MM-DD' string.
    Now a thin wrapper around discover_nearest_expiry() so holiday
    shifts (e.g. Tuesday holiday → Monday expiry) are handled automatically.
    Falls back to static Tuesday calculation when market is closed
    (e.g. weekends) since discover_nearest_expiry needs live data.
    """
    today   = datetime.date.today()
    now_ist = datetime.datetime.now(IST)
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    is_market_hours = (
        today.weekday() < 5 and
        market_open <= now_ist.time() <= market_close
    )

    if is_market_hours:
        try:
            return discover_nearest_expiry()
        except Exception:
            pass  # fall through to static calculation

    # Static fallback: nearest Tuesday (used on weekends / after-hours)
    days_to_tuesday = (1 - today.weekday()) % 7
    if days_to_tuesday == 0 and (now_ist.hour > 15 or (now_ist.hour == 15 and now_ist.minute >= 30)):
        days_to_tuesday = 7
    expiry = today + datetime.timedelta(days=days_to_tuesday)
    return expiry.strftime("%Y-%m-%d")


def get_next_week_expiry():
    """Returns the Tuesday 7 days after get_current_week_expiry()."""
    curr = datetime.datetime.strptime(get_current_week_expiry(), "%Y-%m-%d").date()
    return (curr + datetime.timedelta(days=7)).strftime("%Y-%m-%d")


def fetch_option_chain(expiry_date=None):
    """
    Fetch full Nifty option chain from Upstox for the given expiry.
    Returns raw Upstox JSON response.
    """
    if expiry_date is None:
        expiry_date = get_current_week_expiry()

    resp = requests.get(
        f"{BASE_URL}/option/chain",
        params={
            "instrument_key": NIFTY_INSTRUMENT_KEY,
            "expiry_date": expiry_date,
        },
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def find_strike_near_premium(direction, target_premium=None):
    """
    Find the CE or PE strike whose current LTP is closest to target_premium.

    direction: 1 = bullish (buy CE), -1 = bearish (buy PE)
    target_premium: defaults to config.TARGET_PREMIUM

    Dynamically discovers the actual expiry date that has live LTP data —
    handles Tuesday holidays (Monday expiry), Wednesday shifts, etc.

    Returns dict:
        strike, premium (real LTP), option_type, expiry,
        instrument_key (for live LTP tracking after entry)
    """
    if target_premium is None:
        target_premium = config.TARGET_PREMIUM

    option_type = "CE" if direction == 1 else "PE"
    side_key    = "call_options" if direction == 1 else "put_options"

    # Try expiry dates dynamically: Mon/Tue/Wed of current + next week
    for expiry_date in _try_expiry_candidates():
        try:
            data = fetch_option_chain(expiry_date)
        except Exception:
            continue

        candidates = []
        for row in data.get("data", []):
            side = row.get(side_key)
            if not side:
                continue
            market_data = side.get("market_data", {})
            ltp = market_data.get("ltp", 0)
            if not ltp or ltp <= 0:
                continue
            candidates.append({
                "strike":         row.get("strike_price"),
                "premium":        round(float(ltp), 2),
                "option_type":    option_type,
                "expiry":         expiry_date,
                "instrument_key": side.get("instrument_key", ""),
                "bid":            market_data.get("bid_price", 0),
                "ask":            market_data.get("ask_price", 0),
            })

        if candidates:
            best = min(candidates, key=lambda c: abs(c["premium"] - target_premium))
            return best

    raise RuntimeError(
        f"No valid {option_type} strikes with LTP > 0 found in the next 14 days. "
        f"Market is likely closed or Upstox data is unavailable."
    )


def get_option_ltp(instrument_key):
    """
    Fetch current LTP for an open option position.
    Called every 30s by position monitor after entry.
    instrument_key: Upstox key returned by find_strike_near_premium()
    """
    if not instrument_key:
        return None

    resp = requests.get(
        f"{BASE_URL}/market-quote/ltp",
        params={"instrument_key": instrument_key},
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    key  = list(data.keys())[0] if data else None
    if key:
        return data[key].get("last_price")
    return None
