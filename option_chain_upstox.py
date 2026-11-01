"""
Real Nifty weekly option chain via Upstox API v2.

- find_strike_near_premium(): picks the strike nearest the target premium among
  LIQUID strikes (tight bid-ask, optional OI floor) and returns its delta, IV,
  OI, bid/ask plus chain-wide sentiment stats (PCR, max-OI strikes).
- get_option_quote(): live bid / ask / LTP for an open position.
- get_chain_stats(): sentiment snapshot for the journal.

Uses the Analytics Token — read-only, no daily refresh needed.
"""

import datetime
import os
from zoneinfo import ZoneInfo

import requests

import config
import market_calendar

IST = ZoneInfo("Asia/Kolkata")
BASE_URL = "https://api.upstox.com/v2"
NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError("UPSTOX_ANALYTICS_TOKEN not set in environment")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _today():
    return datetime.datetime.now(IST).date()


# ── Expiry discovery ─────────────────────────────────────────────────────────

_expiry_cache = {"day": None, "expiry": None}


def _try_expiry_candidates():
    """
    Candidate expiry dates in order: the calendar-based expected expiry first,
    then every trading day over the next 14 days (covers unexpected shifts).
    """
    today = _today()
    expected = market_calendar.weekly_expiry_for(today).strftime("%Y-%m-%d")
    candidates = [expected]
    for offset in range(15):
        d = today + datetime.timedelta(days=offset)
        s = d.strftime("%Y-%m-%d")
        if market_calendar.is_trading_day(d) and s not in candidates:
            candidates.append(s)
    return candidates


def _chain_has_live_data(data):
    for row in data.get("data", []) or []:
        for side_key in ("call_options", "put_options"):
            side = row.get(side_key) or {}
            if (side.get("market_data") or {}).get("ltp", 0) > 0:
                return True
    return False


def discover_nearest_expiry():
    """Nearest expiry with live data. Cached for the day once found."""
    today = _today()
    if _expiry_cache["day"] == today and _expiry_cache["expiry"]:
        return _expiry_cache["expiry"]
    for expiry_date in _try_expiry_candidates():
        try:
            data = fetch_option_chain(expiry_date)
        except Exception:
            continue
        if _chain_has_live_data(data):
            _expiry_cache.update(day=today, expiry=expiry_date)
            return expiry_date
    raise RuntimeError("Could not find any Nifty expiry with live data.")


def get_current_week_expiry():
    """Nearest live expiry; falls back to the holiday-aware calendar rule."""
    try:
        return discover_nearest_expiry()
    except Exception:
        return market_calendar.weekly_expiry_for(_today()).strftime("%Y-%m-%d")


def fetch_option_chain(expiry_date=None):
    """Raw Upstox option chain JSON for the given expiry."""
    if expiry_date is None:
        expiry_date = get_current_week_expiry()
    resp = requests.get(
        f"{BASE_URL}/option/chain",
        params={"instrument_key": NIFTY_INSTRUMENT_KEY, "expiry_date": expiry_date},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── Chain statistics (sentiment for the journal) ─────────────────────────────

def chain_stats(data):
    """
    PCR, highest call / put OI strikes and ATM IV from a raw chain response.
    Highest call OI strike above spot is often read as resistance, highest put
    OI strike below spot as support.
    """
    rows = data.get("data", []) or []
    if not rows:
        return {}
    spot = None
    total_call_oi = total_put_oi = 0.0
    max_call = (None, -1.0)
    max_put = (None, -1.0)
    atm_row, atm_dist = None, None
    for row in rows:
        strike = row.get("strike_price")
        spot = spot or row.get("underlying_spot_price")
        c = (row.get("call_options") or {}).get("market_data") or {}
        p = (row.get("put_options") or {}).get("market_data") or {}
        c_oi, p_oi = float(c.get("oi") or 0), float(p.get("oi") or 0)
        total_call_oi += c_oi
        total_put_oi += p_oi
        if c_oi > max_call[1]:
            max_call = (strike, c_oi)
        if p_oi > max_put[1]:
            max_put = (strike, p_oi)
        if spot and strike is not None:
            dist = abs(strike - spot)
            if atm_dist is None or dist < atm_dist:
                atm_row, atm_dist = row, dist

    atm_iv = None
    if atm_row:
        ivs = [((atm_row.get(k) or {}).get("option_greeks") or {}).get("iv")
               for k in ("call_options", "put_options")]
        ivs = [float(v) for v in ivs if v]
        atm_iv = round(sum(ivs) / len(ivs), 2) if ivs else None

    return {
        "spot": spot,
        "pcr": round(total_put_oi / total_call_oi, 3) if total_call_oi else None,
        "max_call_oi_strike": max_call[0],
        "max_put_oi_strike": max_put[0],
        "atm_iv": atm_iv,
    }


def get_chain_stats():
    """Sentiment snapshot for the nearest expiry (empty dict on failure)."""
    try:
        data = fetch_option_chain(get_current_week_expiry())
        stats = chain_stats(data)
        stats["expiry"] = get_current_week_expiry()
        return stats
    except Exception as e:
        print(f"[option_chain] stats unavailable: {e}")
        return {}


# ── Strike selection ─────────────────────────────────────────────────────────

def _spread_pct(bid, ask):
    if not bid or not ask or bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2
    return round((ask - bid) / mid * 100, 2)


def find_strike_near_premium(direction, target_premium=None):
    """
    Pick the liquid CE (direction=1) or PE (direction=-1) strike whose premium is
    closest to target_premium.

    Liquid = two-sided quote with spread <= MAX_SPREAD_PCT and OI >= MIN_STRIKE_OI.
    Returns a dict with strike, premium (LTP), bid, ask, spread_pct, delta, iv, oi,
    option_type, expiry, instrument_key and chain_stats.
    Raises RuntimeError("no liquid strike ...") if nothing qualifies.
    """
    if target_premium is None:
        target_premium = config.TARGET_PREMIUM
    option_type = "CE" if direction == 1 else "PE"
    side_key = "call_options" if direction == 1 else "put_options"

    expiry_date = get_current_week_expiry()
    data = fetch_option_chain(expiry_date)
    if not _chain_has_live_data(data):
        raise RuntimeError(f"No live option data for expiry {expiry_date}")

    liquid, illiquid = [], 0
    for row in data.get("data", []) or []:
        side = row.get(side_key)
        if not side:
            continue
        md = side.get("market_data") or {}
        greeks = side.get("option_greeks") or {}
        ltp = float(md.get("ltp") or 0)
        if ltp <= 0:
            continue
        bid, ask = float(md.get("bid_price") or 0), float(md.get("ask_price") or 0)
        spread = _spread_pct(bid, ask)
        oi = float(md.get("oi") or 0)
        if spread is None or spread > config.MAX_SPREAD_PCT or oi < config.MIN_STRIKE_OI:
            illiquid += 1
            continue
        liquid.append({
            "strike": row.get("strike_price"),
            "premium": round(ltp, 2),
            "bid": bid,
            "ask": ask,
            "spread_pct": spread,
            "delta": abs(float(greeks["delta"])) if greeks.get("delta") else None,
            "iv": float(greeks["iv"]) if greeks.get("iv") else None,
            "oi": oi,
            "option_type": option_type,
            "expiry": expiry_date,
            "instrument_key": side.get("instrument_key", ""),
        })

    if not liquid:
        raise RuntimeError(
            f"no liquid strike: {illiquid} {option_type} strikes failed the "
            f"spread/OI filter (max spread {config.MAX_SPREAD_PCT}%)"
        )
    best = min(liquid, key=lambda c: abs(c["premium"] - target_premium))
    best["chain_stats"] = chain_stats(data)
    return best


# ── Live quotes for open positions ───────────────────────────────────────────

def get_option_quote(instrument_key):
    """
    Live quote for an option: {"bid", "ask", "ltp", "oi"}.
    Uses the full market quote (with depth); falls back to LTP-only.
    Returns None if nothing could be fetched.
    """
    if not instrument_key:
        return None
    try:
        resp = requests.get(
            f"{BASE_URL}/market-quote/quotes",
            params={"instrument_key": instrument_key},
            headers=_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {}) or {}
        if data:
            q = next(iter(data.values()))
            depth = q.get("depth") or {}
            buys, sells = depth.get("buy") or [], depth.get("sell") or []
            bid = float(buys[0].get("price") or 0) if buys else 0.0
            ask = float(sells[0].get("price") or 0) if sells else 0.0
            ltp = float(q.get("last_price") or 0)
            if ltp > 0 or bid > 0:
                return {"bid": bid, "ask": ask, "ltp": ltp, "oi": q.get("oi")}
    except Exception as e:
        print(f"[option_chain] full quote failed, falling back to LTP: {e}")

    ltp = get_option_ltp(instrument_key)
    if ltp:
        return {"bid": 0.0, "ask": 0.0, "ltp": float(ltp), "oi": None}
    return None


def get_option_ltp(instrument_key):
    """LTP only (fallback)."""
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
    key = list(data.keys())[0] if data else None
    return data[key].get("last_price") if key else None
