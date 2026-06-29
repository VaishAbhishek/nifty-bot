"""
Main loop for the Nifty paper trading bot.

Run with: python main.py

Runs continuously during market hours, polling for new closed 5-min candles.
On a signal: fetches the option chain, picks the strike nearest your target
premium, asks for Telegram approval (auto-executes after the timeout), then
simulates the paper trade and tracks it candle-by-candle until SL/target/EOD.
"""

import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import config
import data_feed
import signal_engine
import option_chain
import telegram_bot
from paper_trader import PaperTrader

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def is_market_hours(now):
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE and now.weekday() < 5


def is_force_close_time(now):
    return now.time() >= dtime(config.FORCE_CLOSE_HOUR, config.FORCE_CLOSE_MINUTE)


def get_current_premium(chain_json, strike, option_type):
    for row in chain_json["records"]["data"]:
        side = row.get(option_type)
        if side and row["strikePrice"] == strike:
            return side.get("lastPrice")
    return None


def main():
    engine = signal_engine.SignalEngine(
        min_angle_deg=config.MIN_ANGLE_DEG,
        angle_confirm_bars=config.ANGLE_CONFIRM_BARS,
        retest_buffer_pct=config.RETEST_BUFFER_PCT,
        use_wick_for_retest=config.USE_WICK_FOR_RETEST,
        max_wait_bars=config.MAX_WAIT_BARS,
    )
    trader = PaperTrader()

    telegram_bot.send_message(
        f"🤖 Nifty paper trading bot started. PAPER_TRADING={config.PAPER_TRADING}, "
        f"angle≥{config.MIN_ANGLE_DEG}°, RR={config.RISK_REWARD_RATIO}"
    )

    last_seen_candle_time = None

    while True:
        now = datetime.now(IST)

        if not is_market_hours(now):
            print(f"[{now}] Outside market hours, sleeping...")
            time.sleep(60)
            continue

        if trader.is_paused(now.date()):
            print(f"[{now}] Paused for the day after consecutive losses. Will resume next trading day.")
            time.sleep(60)
            continue

        try:
            df = data_feed.get_recent_candles()
        except Exception as e:
            print(f"[{now}] Data fetch failed: {e}")
            time.sleep(config.POLL_SECONDS)
            continue

        latest_candle_time = df.index[-1]
        if latest_candle_time == last_seen_candle_time:
            # Same candle still open — but check wick retest mid-bar if armed
            if not trader.is_flat():
                _monitor_open_position(trader, now)
            elif engine.armed and config.USE_WICK_FOR_RETEST:
                _check_intrabar_retest(engine, df, trader, now)
            time.sleep(config.POLL_SECONDS)
            continue

        last_seen_candle_time = latest_candle_time
        df = signal_engine.compute_indicators(df, config.FAST_EMA_LEN, config.SLOW_EMA_LEN, config.ATR_LEN)

        events = engine.process(df, flat=trader.is_flat())

        for event in events:
            _handle_event(event, trader, now)

        if not trader.is_flat():
            _monitor_open_position(trader, now)

        time.sleep(config.POLL_SECONDS)


def _check_intrabar_retest(engine, df, trader, now):
    """
    When USE_WICK_FOR_RETEST=True, check if the live candle's running
    low (bullish) or high (bearish) has touched the Fast EMA mid-bar.
    yfinance includes the current partial candle as the last row.
    """
    fast_ema = df["fast_ema"].iloc[-1]
    low = df["low"].iloc[-1]
    high = df["high"].iloc[-1]
    close = df["close"].iloc[-1]
    slow_ema = df["slow_ema"].iloc[-1]
    buffer = fast_ema * config.RETEST_BUFFER_PCT / 100

    retest = False
    if engine.armed_dir == 1 and low <= fast_ema + buffer:
        retest = True
    elif engine.armed_dir == -1 and high >= fast_ema - buffer:
        retest = True

    if retest:
        # Use current live price as entry (closer to real touch than candle close)
        entry_price = float(close)
        sl_price = float(slow_ema)
        events = [("entry", engine.armed_dir, entry_price, sl_price)]
        engine.armed = False
        engine.armed_dir = 0
        for event in events:
            _handle_event(event, trader, now)


def _handle_event(event, trader, now):
    kind = event[0]

    if kind == "angle_check":
        _, direction, angle_deg, passed = event
        dir_str = "Bullish" if direction == 1 else "Bearish"
        status = "PASSED ✅" if passed else "rejected"
        print(f"[{now}] {dir_str} crossover, angle={angle_deg:.1f}°, {status}")

    elif kind == "armed":
        _, direction = event
        dir_str = "Bullish (CE)" if direction == 1 else "Bearish (PE)"
        print(f"[{now}] Armed: {dir_str}, waiting for retest")
        telegram_bot.send_message(f"👀 {dir_str} setup armed, waiting for retest near Fast EMA.")

    elif kind == "invalidated":
        _, reason = event
        print(f"[{now}] Setup invalidated: {reason}")

    elif kind == "skipped_gap":
        print(f"[{now}] Crossover skipped - gap candle")

    elif kind == "entry":
        _, direction, underlying_entry, underlying_sl = event
        _process_entry(direction, underlying_entry, underlying_sl, trader, now)


def _process_entry(direction, underlying_entry, underlying_sl, trader, now):
    dir_str = "CE (Buy)" if direction == 1 else "PE (Sell)"
    try:
        chain = option_chain.fetch_option_chain(config.NSE_OPTION_SYMBOL)
        strike_info = option_chain.find_strike_near_premium(chain, direction, config.TARGET_PREMIUM)
    except Exception as e:
        print(f"[{now}] Option chain fetch failed, skipping this signal: {e}")
        telegram_bot.send_message(f"⚠️ Entry signal fired ({dir_str}) but option chain fetch failed: {e}")
        return

    msg = (
        f"📍 Entry signal: {dir_str}\n"
        f"Strike: {strike_info['strike']} {strike_info['option_type']}\n"
        f"Premium: ₹{strike_info['premium']}\n"
        f"Expiry: {strike_info['expiry']}\n"
        f"(Paper trade - approve or it auto-executes in {config.APPROVAL_TIMEOUT_SECONDS}s)"
    )
    approved = telegram_bot.request_approval(msg)

    if not approved:
        print(f"[{now}] Entry rejected via Telegram")
        return

    trader.open_position(direction, strike_info, underlying_entry, underlying_sl,
                          config.RISK_REWARD_RATIO, now)
    telegram_bot.send_message(
        f"✅ PAPER trade opened: {strike_info['strike']} {strike_info['option_type']} "
        f"@ ₹{strike_info['premium']}"
    )


def _monitor_open_position(trader, now):
    pos = trader.position
    try:
        chain = option_chain.fetch_option_chain(config.NSE_OPTION_SYMBOL)
        current_premium = get_current_premium(chain, pos["strike"], pos["option_type"])
    except Exception as e:
        print(f"[{now}] Could not fetch current premium: {e}")
        return

    if current_premium is None:
        return

    force_close = is_force_close_time(now)
    exit_info = trader.check_exit(current_premium, now, force_close=force_close)

    if exit_info:
        telegram_bot.send_message(
            f"🔔 PAPER trade closed: {exit_info['result']} ({exit_info['exit_reason']})\n"
            f"{exit_info['strike']} {exit_info['option_type']}: "
            f"₹{exit_info['entry_premium']} → ₹{exit_info['exit_premium']}\n"
            f"P&L per lot: ₹{exit_info['pnl_per_lot']}"
        )


if __name__ == "__main__":
    main()
