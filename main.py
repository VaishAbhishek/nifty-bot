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
import option_chain_synth
import telegram_bot
from paper_trader import PaperTrader

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def is_market_hours(now):
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE and now.weekday() < 5


def is_force_close_time(now):
    return now.time() >= dtime(config.FORCE_CLOSE_HOUR, config.FORCE_CLOSE_MINUTE)


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

        # Reset signal engine state at the start of each new trading day
        current_day = now.date()
        if not hasattr(main, '_last_trading_day') or main._last_trading_day != current_day:
            engine.reset_day()
            main._last_trading_day = current_day
            print(f"[{now}] New trading day {current_day} — signal engine state reset")

        try:
            df = data_feed.get_recent_candles()
        except Exception as e:
            print(f"[{now}] Data fetch failed: {e}")
            time.sleep(config.POLL_SECONDS)
            continue

        df = signal_engine.compute_indicators(df, config.FAST_EMA_LEN, config.SLOW_EMA_LEN, config.ATR_LEN)

        latest_candle_time = df.index[-1]
        if latest_candle_time == last_seen_candle_time:
            # Same candle still open — but check wick retest mid-bar if armed
            if not trader.is_flat():
                _monitor_open_position(trader, now)
            elif engine.armed and config.USE_WICK_FOR_RETEST:
                _check_intrabar_retest(engine, trader, now)
            time.sleep(config.POLL_SECONDS)
            continue

        last_seen_candle_time = latest_candle_time

        events = engine.process(df, flat=trader.is_flat())

        for event in events:
            _handle_event(event, trader, now)

        if not trader.is_flat():
            _monitor_open_position(trader, now)

        time.sleep(config.POLL_SECONDS)


def _check_intrabar_retest(engine, trader, now):
    """
    When USE_WICK_FOR_RETEST=True, check if the live candle's running
    low (bullish) or high (bearish) has touched the Fast EMA mid-bar.

    Note: data_feed.get_recent_candles() now filters out partial candles,
    so we fetch raw data here to get the current in-progress candle's wick.
    """
    try:
        import yfinance as yf
        import pandas as pd
        raw = yf.download(config.UNDERLYING_TICKER, period="1d",
                          interval="5m", progress=False, auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert("Asia/Kolkata")

        # Get closed candles for EMA calculation
        df = data_feed.get_recent_candles()
        df = signal_engine.compute_indicators(df, config.FAST_EMA_LEN,
                                              config.SLOW_EMA_LEN, config.ATR_LEN)

        fast_ema = float(df["fast_ema"].iloc[-1])
        slow_ema = float(df["slow_ema"].iloc[-1])

        # Use LIVE partial candle's wick for retest detection
        live_low  = float(raw["low"].iloc[-1])
        live_high = float(raw["high"].iloc[-1])
        live_close = float(raw["close"].iloc[-1])
        buffer = fast_ema * config.RETEST_BUFFER_PCT / 100

        retest = False
        if engine.armed_dir == 1 and live_low <= fast_ema + buffer:
            retest = True
        elif engine.armed_dir == -1 and live_high >= fast_ema - buffer:
            retest = True

        if retest:
            events = [("entry", engine.armed_dir, live_close, slow_ema)]
            engine.armed = False
            engine.armed_dir = 0
            engine.wait_start_ts = None
            for event in events:
                _handle_event(event, trader, now)

    except Exception as e:
        print(f"[{now}] Intrabar retest check failed: {e}")


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
        df = data_feed.get_recent_candles()
        spot_price = float(df["close"].iloc[-1])
        strike_info = option_chain_synth.find_strike_near_premium_synthetic(
            spot_price, direction, config.TARGET_PREMIUM
        )
    except Exception as e:
        print(f"[{now}] Synthetic option pricing failed: {e}")
        telegram_bot.send_message(
            f"⚠️ Entry signal fired ({dir_str}) but synthetic pricing failed: {e}"
        )
        return

    msg = (
        f"📍 Entry signal: {dir_str}\n"
        f"Strike: {strike_info['strike']} {strike_info['option_type']}\n"
        f"Premium: ₹{strike_info['premium']} (B-S estimate, σ={strike_info['sigma_used']}%)\n"
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
        df = data_feed.get_recent_candles()
        spot_price = float(df["close"].iloc[-1])
        # Re-price the open position using current spot + B-S
        current_premium = option_chain_synth.black_scholes(
            S=spot_price,
            K=pos["strike"],
            T=option_chain_synth.get_time_to_expiry(
                option_chain_synth.get_current_week_expiry_date()
            ),
            r=0.065,
            sigma=option_chain_synth.get_india_vix(),
            option_type=pos["option_type"]
        )
    except Exception as e:
        print(f"[{now}] Could not fetch current premium: {e}")
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