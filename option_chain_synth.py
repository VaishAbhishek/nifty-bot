"""
Black-Scholes synthetic option premium calculator.
Used as fallback when NSE's option chain endpoint is blocked (e.g. from Oracle Cloud IPs).

Limitations:
- Uses a fixed volatility estimate (VIX proxy) — real IV varies by strike/expiry
- Premium is approximate, not the real tradable price
- Good enough to validate entry/exit TIMING during paper trading
- Replace with real NSE data before going live with real money

For paper trading purposes, the entry/exit logic timing is what matters most —
the actual premium numbers will differ slightly from real market prices.
"""

import math
import datetime
import yfinance as yf
import config


def get_india_vix():
    """
    Fetch India VIX as volatility estimate.
    Falls back to 15% if fetch fails.
    """
    try:
        vix = yf.Ticker("^INDIAVIX")
        hist = vix.history(period="1d", interval="5m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100
    except Exception:
        pass
    return 0.15  # 15% fallback — conservative estimate for Nifty


def black_scholes(S, K, T, r, sigma, option_type="CE"):
    """
    Standard Black-Scholes formula.
    S: spot price
    K: strike price
    T: time to expiry in years
    r: risk-free rate (use 0.065 for India ~6.5% repo rate)
    sigma: annualized volatility (e.g. 0.15 for 15%)
    option_type: "CE" or "PE"
    """
    if T <= 0:
        # Expired — intrinsic value only
        if option_type == "CE":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    def N(x):
        """Standard normal CDF"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if option_type == "CE":
        price = S * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        price = K * math.exp(-r * T) * N(-d2) - S * N(-d1)

    return round(price, 2)


def get_time_to_expiry(expiry_date):
    """
    Returns time to expiry in years from now.
    expiry_date: datetime.date object (Thursday of current week)
    """
    now = datetime.datetime.now()
    expiry_dt = datetime.datetime.combine(expiry_date, datetime.time(15, 30))  # 3:30 PM IST
    delta = expiry_dt - now
    return max(delta.total_seconds() / (365.25 * 24 * 3600), 1e-6)


def get_current_week_expiry_date():
    """
    Nifty weekly options expire every Thursday.
    Returns this week's Thursday date.
    """
    today = datetime.date.today()
    days_to_thursday = (3 - today.weekday()) % 7  # 3 = Thursday
    if days_to_thursday == 0 and datetime.datetime.now().hour >= 15:
        days_to_thursday = 7  # Today is Thursday but market closed, use next week
    return today + datetime.timedelta(days=days_to_thursday)


def find_strike_near_premium_synthetic(spot_price, direction, target_premium=95.0, sigma=None):
    """
    Scans strikes around ATM to find the one whose B-S premium is closest to target.

    direction: 1 for CE, -1 for PE
    Returns dict matching the same format as option_chain.find_strike_near_premium()
    """
    if sigma is None:
        sigma = get_india_vix()

    option_type = "CE" if direction == 1 else "PE"
    expiry_date = get_current_week_expiry_date()
    T = get_time_to_expiry(expiry_date)
    r = 0.065  # RBI repo rate proxy

    # Round spot to nearest 50 (Nifty strike interval)
    atm = round(spot_price / 50) * 50

    # Scan strikes from deep ITM to deep OTM
    strikes = [atm + (i * 50) for i in range(-20, 21)]

    best = None
    best_diff = float("inf")

    for K in strikes:
        premium = black_scholes(spot_price, K, T, r, sigma, option_type)
        diff = abs(premium - target_premium)
        if diff < best_diff:
            best_diff = diff
            best = {
                "strike": K,
                "premium": premium,
                "option_type": option_type,
                "expiry": expiry_date.strftime("%d-%b-%Y"),
                "source": "synthetic_bs",
                "sigma_used": round(sigma * 100, 1),
            }

    return best
