"""
Main loop for the Nifty paper trading bot.
Now uses Upstox API for real NSE data:
  - data_feed_upstox  : real 5-min Nifty candles (replaces yfinance)
  - option_chain_upstox: real option chain LTP (replaces synthetic B-S)

Run with: python3 main.py
"""

import os
import time
import requests
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import config
import data_feed_upstox as data_feed
import signal_engine
import option_chain_upstox as option_chain
import telegram_bot
from paper_trader import PaperTrader

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def is_market_hours(now):
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE and now.weekday() < 5


def is_force_close_time(now):
    return now.time() >= dtime(config.FORCE_CLOSE_HOUR, config.FORCE_CLOSE_MINUTE)


def main():
    engine = signal_engine.SignalEngine(
        min_angle_deg      = config.MIN_ANGLE_DEG,
        angle_confirm_bars = config.ANGLE_CONFIRM_BARS,
        retest_buffer_pct  = config.RETEST_BUFFER_PCT,
        use_wick_for_retest= config.USE_WICK_FOR_RETEST,
        max_wait_bars      = config.MAX_WAIT_BARS,
    )
    trader = PaperTrader()

    telegram_bot.send_message(
        f"🤖 Nifty bot started (Upstox data). "
        f"PAPER_TRADING={config.PAPER_TRADING}, "
        f"angle≥{config.MIN_ANGLE_DEG}°, RR={config.RISK_REWARD_RATIO}"
    )

    last_seen_candle_time = None

    while True:
        now = datetime.now(IST)

        if not is_market_hours(now):
            print(f"[{now:%H:%M:%S}] Outside market hours, sleeping...")
            time.sleep(60)
            continue

        if trader.is_paused(now.date()):
            print(f"[{now:%H:%M:%S}] Paused after consecutive losses. Resumes tomorrow.")
            time.sleep(60)
            continue

        # ── Daily state reset at market open ──────────────────────────
        current_day = now.date()
        if not hasattr(main, '_last_day') or main._last_day != current_day:
            engine.reset_day()
            main._last_day = current_day
            print(f"[{now:%H:%M:%S}] New day {current_day} — engine state reset")

        # ── Fetch candles ──────────────────────────────────────────────
        # get_recent_candles() now merges:
        #   - historical 5-min (v2 1-min API resampled)
        #   - today's live 5-min (v3 intraday endpoint, no polling needed)
        try:
            df = data_feed.get_recent_candles()
        except Exception as e:
            print(f"[{now:%H:%M:%S}] Candle fetch failed: {e}")
            time.sleep(config.POLL_SECONDS)
            continue

        df = signal_engine.compute_indicators(
            df, config.FAST_EMA_LEN, config.SLOW_EMA_LEN, config.ATR_LEN
        )

        latest_candle_time = df.index[-1]

        # ── Same candle still running — check intrabar retest & monitor ──
        if latest_candle_time == last_seen_candle_time:
            if not trader.is_flat():
                _monitor_open_position(trader, now)
            elif engine.armed and config.USE_WICK_FOR_RETEST:
                _check_intrabar_retest(engine, df, trader, now)
            time.sleep(config.POLL_SECONDS)
            continue

        # ── New closed candle ──────────────────────────────────────────
        last_seen_candle_time = latest_candle_time
        print(f"[{now:%H:%M:%S}] New candle: {latest_candle_time:%H:%M} "
              f"close={df['close'].iloc[-1]:.2f}")

        events = engine.process(df, flat=trader.is_flat())
        for event in events:
            _handle_event(event, trader, now)

        if not trader.is_flat():
            _monitor_open_position(trader, now)

        time.sleep(config.POLL_SECONDS)



def _check_intrabar_retest(engine, df, trader, now):
    """
    Check if the live (partial) candle's wick has touched the Fast EMA.
    Uses v3 intraday endpoint (minutes/5) — includes the current in-progress bar.
    """
    try:
        token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
        resp = requests.get(
            "https://api.upstox.com/v3/historical-candle/intraday"
            "/NSE_INDEX%7CNifty%2050/minutes/5",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return
        candles = resp.json().get("data", {}).get("candles", [])
        if not candles:
            return

        # Last candle is the current in-progress bar
        live = candles[-1]   # [ts, open, high, low, close, vol, oi]
        live_high  = float(live[2])
        live_low   = float(live[3])
        live_close = float(live[4])

        fast_ema = float(df["fast_ema"].iloc[-1])
        slow_ema = float(df["slow_ema"].iloc[-1])
        buffer   = fast_ema * config.RETEST_BUFFER_PCT / 100

        retest = False
        if engine.armed_dir == 1 and live_low <= fast_ema + buffer:
            retest = True
        elif engine.armed_dir == -1 and live_high >= fast_ema - buffer:
            retest = True

        if retest:
            direction = engine.armed_dir
            engine.armed         = False
            engine.armed_dir     = 0
            engine.wait_start_ts = None
            _handle_event(("entry", direction, live_close, slow_ema), trader, now)

    except Exception as e:
        print(f"[{now:%H:%M:%S}] Intrabar retest error: {e}")


# ── Event handler ──────────────────────────────────────────────────────────

def _handle_event(event, trader, now):
    kind = event[0]

    if kind == "angle_check":
        _, direction, angle_deg, passed = event
        dir_str = "Bullish" if direction == 1 else "Bearish"
        status  = "PASSED ✅" if passed else "rejected"
        print(f"[{now:%H:%M:%S}] {dir_str} crossover, angle={angle_deg:.1f}°, {status}")

    elif kind == "armed":
        _, direction = event
        dir_str = "Bullish (CE)" if direction == 1 else "Bearish (PE)"
        print(f"[{now:%H:%M:%S}] Armed: {dir_str}, waiting for retest")
        telegram_bot.send_message(
            f"👀 {dir_str} setup armed — waiting for retest near Fast EMA."
        )

    elif kind == "invalidated":
        _, reason = event
        print(f"[{now:%H:%M:%S}] Setup invalidated: {reason}")

    elif kind == "skipped_gap":
        print(f"[{now:%H:%M:%S}] Crossover skipped — gap candle")

    elif kind == "entry":
        _, direction, underlying_entry, underlying_sl = event
        _process_entry(direction, underlying_entry, underlying_sl, trader, now)


def _process_entry(direction, underlying_entry, underlying_sl, trader, now):
    dir_str = "CE (Buy Call)" if direction == 1 else "PE (Buy Put)"

    try:
        strike_info = option_chain.find_strike_near_premium(
            direction, config.TARGET_PREMIUM
        )
    except Exception as e:
        msg = f"⚠️ Entry signal ({dir_str}) — option chain fetch failed: {e}"
        print(f"[{now:%H:%M:%S}] {msg}")
        telegram_bot.send_message(msg)
        return

    bid_ask = ""
    if strike_info.get("bid") and strike_info.get("ask"):
        bid_ask = f"\nBid/Ask: ₹{strike_info['bid']} / ₹{strike_info['ask']}"

    msg = (
        f"📍 Entry signal: {dir_str}\n"
        f"Strike: {strike_info['strike']} {strike_info['option_type']}\n"
        f"Premium (LTP): ₹{strike_info['premium']}{bid_ask}\n"
        f"Expiry: {strike_info['expiry']}\n"
        f"(Paper trade — approve or auto-executes in {config.APPROVAL_TIMEOUT_SECONDS}s)"
    )
    approved = telegram_bot.request_approval(msg)

    if not approved:
        print(f"[{now:%H:%M:%S}] Entry rejected via Telegram")
        return

    position = trader.open_position(
        direction, strike_info, underlying_entry,
        underlying_sl, config.RISK_REWARD_RATIO, now
    )
    telegram_bot.send_message(
        f"✅ PAPER trade opened:\n"
        f"{strike_info['strike']} {strike_info['option_type']} @ ₹{strike_info['premium']}\n"
        f"SL: ₹{position['sl_premium']:.2f} | Target: ₹{position['target_premium']:.2f}"
    )


def _monitor_open_position(trader, now):
    pos = trader.position
    if not pos:
        return

    # Use real Upstox LTP for the open option position
    try:
        instrument_key  = pos.get("instrument_key", "")
        current_premium = option_chain.get_option_ltp(instrument_key)
        if current_premium is None:
            print(f"[{now:%H:%M:%S}] Could not fetch current premium (None returned)")
            return
    except Exception as e:
        print(f"[{now:%H:%M:%S}] Premium fetch failed: {e}")
        return

    force_close = is_force_close_time(now)
    exit_info   = trader.check_exit(current_premium, now, force_close=force_close)

    if exit_info:
        emoji = "🟢" if exit_info["result"] == "WIN" else "🔴"
        telegram_bot.send_message(
            f"{emoji} PAPER trade closed: {exit_info['result']} ({exit_info['exit_reason']})\n"
            f"{exit_info['strike']} {exit_info['option_type']}: "
            f"₹{exit_info['entry_premium']} → ₹{exit_info['exit_premium']}\n"
            f"P&L per lot: ₹{exit_info['pnl_per_lot']:,.2f}"
        )


if __name__ == "__main__":
    main()