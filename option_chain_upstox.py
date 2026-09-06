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


def get_current_week_expiry():
    """
    Nifty weekly options expire every TUESDAY (effective September 1, 2025).
    Returns this week's Tuesday as 'YYYY-MM-DD'.
    If today is Tuesday and market has closed (after 3:30 PM IST),
    returns next Tuesday.
    """
    today    = datetime.date.today()
    now_ist  = datetime.datetime.now(IST)
    days_to  = (1 - today.weekday()) % 7   # 1 = Tuesday
    if days_to == 0 and now_ist.hour >= 15 and now_ist.minute >= 30:
        days_to = 7
    expiry = today + datetime.timedelta(days=days_to)
    return expiry.strftime("%Y-%m-%d")


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
    target_premium: defaults to config.TARGET_PREMIUM (₹95)

    Returns dict:
        strike, premium (real LTP), option_type, expiry,
        instrument_key (for live LTP tracking after entry)
    """
    if target_premium is None:
        target_premium = config.TARGET_PREMIUM

    expiry_date = get_current_week_expiry()
    data        = fetch_option_chain(expiry_date)

    side_key    = "call_options" if direction == 1 else "put_options"
    option_type = "CE"           if direction == 1 else "PE"

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

    if not candidates:
        raise RuntimeError(
            f"No valid {option_type} strikes found for expiry {expiry_date}. "
            f"Market might be closed or data unavailable."
        )

    best = min(candidates, key=lambda c: abs(c["premium"] - target_premium))
    return best


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
