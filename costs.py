"""
Transaction cost model for one round trip (buy then sell) of index options.

Rates live in config.py — check them against an actual Upstox contract note
and adjust if they differ.
"""

import config


def round_trip_charges(buy_premium, sell_premium, lots=1):
    """Return total ₹ charges for buying and selling `lots` lots."""
    qty = config.LOT_SIZE * lots
    buy_value = buy_premium * qty
    sell_value = sell_premium * qty
    turnover = buy_value + sell_value

    brokerage = 2 * config.BROKERAGE_PER_ORDER
    stt = sell_value * config.STT_SELL_PCT / 100
    exchange = turnover * config.EXCHANGE_TXN_PCT / 100
    sebi = turnover * config.SEBI_FEE_PER_CRORE / 1e7
    stamp = buy_value * config.STAMP_DUTY_BUY_PCT / 100
    gst = (brokerage + exchange + sebi) * config.GST_PCT / 100

    return round(brokerage + stt + exchange + sebi + stamp + gst, 2)


def cost_per_unit(premium):
    """Approximate round-trip cost per unit, used to set a true breakeven stop."""
    return round_trip_charges(premium, premium) / config.LOT_SIZE
