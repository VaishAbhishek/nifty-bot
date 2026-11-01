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
        # Normalize to UTC microseconds for reliable equality comparison
        # (avoids tz-aware vs naive drift causing duplicate processing)
        latest_candle_utc = df.index[-1].tz_convert("UTC") if hasattr(df.index[-1], 'tz_convert') else df.index[-1]

        # ── Same candle still running — check intrabar retest & monitor ──
        if latest_candle_utc == last_seen_candle_time:
            if not trader.is_flat():
                # Position open — check exits fast (SL/target can hit within seconds)
                _monitor_all_positions(trader, now)
                sleep_secs = config.POSITION_POLL_SECONDS
            elif engine.armed:
                # Armed — real-time LTP-based retest: fires at the exact tick price touches EMA
                _check_ltp_retest(engine, df, trader, now)
                # After _check_ltp_retest, engine.armed may now be False (entry fired).
                # Use fast poll if still armed, normal poll if entry fired or not armed.
                sleep_secs = config.LTP_POLL_SECONDS if engine.armed else config.POLL_SECONDS
            elif trader.is_flat() and not engine.armed and engine.pending_dir == 0:
                # Not armed yet — check if a crossover is forming intrabar (informational only)
                _check_intrabar_crossover(engine, df, now)
                sleep_secs = config.POLL_SECONDS
            else:
                sleep_secs = config.POLL_SECONDS
            time.sleep(sleep_secs)
            continue

        # ── New closed candle ──────────────────────────────────────────
        last_seen_candle_time = latest_candle_utc
        fast_ema_val = df["fast_ema"].iloc[-1]
        slow_ema_val = df["slow_ema"].iloc[-1]
        print(f"[{now:%H:%M:%S}] New candle: {latest_candle_time:%H:%M} "
              f"close={df['close'].iloc[-1]:.2f} "
              f"fast={fast_ema_val:.2f} slow={slow_ema_val:.2f}")

        events = engine.process(df, flat=True)   # flat=True: engine signals always; positions managed separately
        for event in events:
            _handle_event(event, trader, now)

        # ── EOD: clear armed state at force-close time ───────────────────
        if is_force_close_time(now) and engine.armed:
            print(f"[{now:%H:%M:%S}] EOD: armed {('CE' if engine.armed_dir == 1 else 'PE')} "
                  f"setup cleared (force-close time — no retest occurred)")
            telegram_bot.send_message(
                f"⏰ EOD: Armed {'CE' if engine.armed_dir == 1 else 'PE'} setup expired at force-close. No trade taken."
            )
            engine.reset_day()

        # Monitor ALL open positions (supports multiple simultaneous trades)
        _monitor_all_positions(trader, now)

        time.sleep(config.POLL_SECONDS)





# ── Intrabar early-warning crossover check ────────────────────────────────────

_intrabar_warned_bar_ts = None   # module-level guard — warn once per live bar

def _check_intrabar_crossover(engine, df, now):
    """
    Called every 30s during the current live bar (before it closes).
    Appends the partial candle to df, recomputes EMAs, and checks if a
    crossover is forming right now.

    If yes: sends a Telegram early-warning (up to ~4.5 min before bar close).
    This is purely informational — no trade fires until bar close + angle check.
    Fires at most once per bar (guarded by _intrabar_warned_bar_ts).
    """
    global _intrabar_warned_bar_ts
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

        # Last candle is the current in-progress (partial) bar
        live = candles[-1]   # [ts, open, high, low, close, vol, oi]
        live_ts    = live[0]
        live_close = float(live[4])

        # Only warn once per live bar
        if _intrabar_warned_bar_ts == live_ts:
            return

        # Append partial candle to df and recompute EMAs
        import pandas as pd
        live_index = pd.Timestamp(live_ts)
        if live_index.tzinfo is None:
            live_index = live_index.tz_localize("Asia/Kolkata")
        else:
            live_index = live_index.tz_convert("Asia/Kolkata")

        live_row = pd.DataFrame(
            {"open": [float(live[1])], "high": [float(live[2])],
             "low": [float(live[3])], "close": [live_close]},
            index=[live_index]
        )
        df_live = pd.concat([df, live_row])
        df_live = signal_engine.compute_indicators(
            df_live, config.FAST_EMA_LEN, config.SLOW_EMA_LEN, config.ATR_LEN
        )

        fast = df_live["fast_ema"]
        slow = df_live["slow_ema"]
        n = len(df_live) - 1

        cross_up   = fast.iloc[n-1] < slow.iloc[n-1] and fast.iloc[n] > slow.iloc[n]
        cross_down = fast.iloc[n-1] > slow.iloc[n-1] and fast.iloc[n] < slow.iloc[n]

        if cross_up or cross_down:
            _intrabar_warned_bar_ts = live_ts
            direction = "Bullish (CE)" if cross_up else "Bearish (PE)"
            fast_now  = fast.iloc[n]
            slow_now  = slow.iloc[n]
            # Compute minutes remaining in the current 5-min bar
            bar_open = live_index.replace(second=0, microsecond=0)
            elapsed_sec = max(0, (now - bar_open.replace(tzinfo=now.tzinfo)).total_seconds())
            mins_left = max(0, 5 - int(elapsed_sec // 60))

            msg = (
                f"⚡ Early warning: {direction} crossover forming!\n"
                f"Live price: {live_close:.2f}\n"
                f"Fast EMA: {fast_now:.2f} | Slow EMA: {slow_now:.2f}\n"
                f"~{mins_left} min until bar close — wait for angle confirmation."
            )
            print(f"[{now:%H:%M:%S}] ⚡ INTRABAR crossover developing: {direction} "
                  f"live={live_close:.2f} fast={fast_now:.2f} slow={slow_now:.2f}")
            telegram_bot.send_message(msg)

    except Exception as e:
        print(f"[{now:%H:%M:%S}] Intrabar crossover check error: {e}")


def _check_ltp_retest(engine, df, trader, now):
    """
    Real-time EMA touch detection using the Nifty 50 LTP (tick price).

    Why this is better than reading the partial candle's high/low:
    - The LTP endpoint reflects the *current* market price, not a bar aggregate.
    - We fire the entry at the exact moment price touches the EMA, not at bar close.
    - Fast EMA is fixed for the entire live bar (computed from closed candles only),
      so we can compare LTP against it directly without recomputing anything.

    Guard: requires at least 1 fully closed candle since arming — prevents entering
    on the same candle that generated the crossover before price has moved away.
    """
    # Enforce minimum 1-bar wait after arming
    if engine.wait_start_ts is not None:
        wait_pos = None
        try:
            wait_pos = df.index.get_loc(engine.wait_start_ts)
        except KeyError:
            pass
        bars_since_arm = (len(df) - 1 - wait_pos) if wait_pos is not None else 0
        if bars_since_arm < 1:
            return  # too soon — arm candle hasn't closed yet

    try:
        ltp = data_feed.get_nifty_ltp()
        if ltp is None:
            return

        # EMA values are fixed for the current live bar — computed from closed candles
        fast_ema = float(df["fast_ema"].iloc[-1])
        slow_ema = float(df["slow_ema"].iloc[-1])
        buffer   = fast_ema * config.RETEST_BUFFER_PCT / 100

        retest = False
        if engine.armed_dir == 1 and ltp <= fast_ema + buffer:
            # Bullish setup: price pulled back down to touch the Fast EMA
            retest = True
        elif engine.armed_dir == -1 and ltp >= fast_ema - buffer:
            # Bearish setup: price bounced up to touch the Fast EMA
            retest = True

        if retest:
            direction = engine.armed_dir
            print(
                f"[{now:%H:%M:%S}] 🎯 LTP retest detected! "
                f"LTP={ltp:.2f} fast_ema={fast_ema:.2f} "
                f"({'bullish' if direction == 1 else 'bearish'})"
            )
            engine.armed         = False
            engine.armed_dir     = 0
            engine.wait_start_ts = None
            # Entry price = actual LTP at touch (not candle close)
            _handle_event(("entry", direction, ltp, slow_ema), trader, now)

    except Exception as e:
        print(f"[{now:%H:%M:%S}] LTP retest check error: {e}")


# ── Event handler ──────────────────────────────────────────────────────────

def _handle_event(event, trader, now):
    kind = event[0]

    if kind == "angle_check":
        # angle_check tuple: (kind, direction, angle_deg, passed, ema_reversed)
        _, direction, angle_deg, passed = event[0], event[1], event[2], event[3]
        ema_reversed = event[4] if len(event) > 4 else False
        dir_str = "Bullish" if direction == 1 else "Bearish"
        status  = "PASSED ✅" if passed else "rejected"
        reversal_note = " (EMA reversed after cross — choppy)" if ema_reversed and not passed else ""
        print(f"[{now:%H:%M:%S}] {dir_str} crossover, angle={angle_deg:.1f}°, {status}{reversal_note}")

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

    elif kind == "skipped_crossover":
        _, direction, reason = event
        dir_str = "Bullish (CE)" if direction == 1 else "Bearish (PE)"
        reason_map = {
            "already_armed":  "a setup is already armed",
            "pending_exists": "waiting on earlier crossover angle",
        }
        reason_str = reason_map.get(reason, reason)
        print(f"[{now:%H:%M:%S}] {dir_str} crossover SKIPPED — {reason_str}")


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


def _monitor_all_positions(trader, now):
    """Monitor ALL open positions and handle exits. Supports multiple simultaneous trades."""
    if trader.is_flat():
        return

    force_close = is_force_close_time(now)
    count = len(trader.positions)
    if count > 1:
        print(f"[{now:%H:%M:%S}] Monitoring {count} open positions")

    closed = trader.check_all_exits(option_chain.get_option_ltp, now, force_close=force_close)
    for exit_info in closed:
        emoji = "🟢" if exit_info["result"] == "WIN" else "🔴"
        telegram_bot.send_message(
            f"{emoji} PAPER trade closed: {exit_info['result']} ({exit_info['exit_reason']})\n"
            f"{exit_info['strike']} {exit_info['option_type']}: "
            f"₹{exit_info['entry_premium']} → ₹{exit_info['exit_premium']}\n"
            f"P&L per lot: ₹{exit_info['pnl_per_lot']:,.2f}"
        )


if __name__ == "__main__":
    main()