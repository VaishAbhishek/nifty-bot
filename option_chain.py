"""
Fetches the live Nifty weekly option chain from NSE's public website endpoint.

IMPORTANT: This is NOT an official/documented API. NSE can change, block, or
rate-limit this at any time without notice. It requires a believable browser
User-Agent and a two-step request (homepage first, to pick up cookies, then
the API call) or NSE will reject the request.
"""

import requests
import time

BASE_URL = "https://www.nseindia.com"
OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    # Hitting the homepage first sets the cookies NSE checks for on the API call
    session.get(BASE_URL, timeout=10)
    time.sleep(1)
    return session


def fetch_option_chain(symbol="NIFTY", retries=3):
    """
    Returns the raw parsed JSON option chain from NSE, or raises an exception
    if all retries fail.
    """
    session = _get_session()
    last_error = None
    for attempt in range(retries):
        try:
            resp = session.get(OPTION_CHAIN_URL, params={"symbol": symbol}, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}"
        except Exception as e:
            last_error = str(e)
        time.sleep(2)
    raise RuntimeError(f"Failed to fetch NSE option chain after {retries} attempts: {last_error}")


def get_current_week_expiry(option_chain_json):
    """First expiry in NSE's list is always the nearest one."""
    return option_chain_json["records"]["expiryDates"][0]


def find_strike_near_premium(option_chain_json, direction, target_premium=95.0):
    """
    direction: 1 for bullish (find CE), -1 for bearish (find PE)
    Returns dict: {strike, premium, option_type, expiry}
    """
    expiry = get_current_week_expiry(option_chain_json)
    side_key = "CE" if direction == 1 else "PE"

    candidates = []
    for row in option_chain_json["records"]["data"]:
        if row.get("expiryDate") != expiry:
            continue
        side = row.get(side_key)
        if not side:
            continue
        premium = side.get("lastPrice")
        if premium is None or premium <= 0:
            continue
        candidates.append({
            "strike": row["strikePrice"],
            "premium": premium,
            "option_type": side_key,
            "expiry": expiry,
        })

    if not candidates:
        raise RuntimeError(f"No valid {side_key} strikes found for expiry {expiry}")

    best = min(candidates, key=lambda c: abs(c["premium"] - target_premium))
    return best
