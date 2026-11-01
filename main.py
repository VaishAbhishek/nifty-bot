"""
Main loop for the Nifty paper trading bot (Upstox data).

Each cycle:
  1. Monitor open positions FIRST (stops, trailing, targets, 15:15 force-close),
     regardless of pauses or pending approvals.
  2. Fetch candles, run the signal engine, log every signal with market context.
  3. Every SNAPSHOT_EVERY_MINUTES, record a market context snapshot.
  4. After the close, write the daily journal (and the weekly review on the
     last trading day of the week) and send a Telegram summary.

Run with: python3 main.py
"""

import os
import time
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config
import data_feed_upstox as data_feed
import journal
import market_calendar
import option_chain_upstox as option_chain
import signal_engine
import telegram_bot
import weekly_review
from paper_trader import PaperTrader, premium_risk_points, structure_stop

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

ctx_builder = journal.ContextBuilder()
_chain_cache = {"at": None, "stats": {}}


def is_market_hours(now):
    return (market_calendar.is_trading_day(now.date())
            and MARKET_OPEN <= now.time() <= MARKET_CLOSE)


def is_force_close_time(now):
    return now.time() >= dtime(config.FORCE_CLOSE_HOUR, config.FORCE_CLOSE_MINUTE)


# ── Market context ───────────────────────────────────────────────────────────

def market_context(df, now, chain=None, refresh_chain=False):
    """Context dict for the journal. Option-chain stats are cached ~10 min."""
    if chain is None:
        stale = (_chain_cache["at"] is None
                 or (now - _chain_cache["at"]).total_seconds() > 600)
        if refresh_chain or stale:
            _chain_cache["stats"] = option_chain.get_chain_stats()
            _chain_cache["at"] = now
        chain = _chain_cache["stats"]
    return ctx_builder.build(df, now, vix=data_feed.get_vix(), chain=chain)


def _safe_context(df, now, **kw):
    try:
        return market_context(df, now, **kw)
    except Exception as e:
        print(f"[{now:%H:%M:%S}] context error: {e}")
        return {}


def _dir_str(direction):
    return "CE" if direction == 1 else "PE"


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    engine = signal_engine.SignalEngine(
        min_angle_deg=config.MIN_ANGLE_DEG,
        angle_confirm_bars=config.ANGLE_CONFIRM_BARS,
        retest_buffer_pct=config.RETEST_BUFFER_PCT,
        use_wick_for_retest=config.USE_WICK_FOR_RETEST,
        max_wait_bars=config.MAX_WAIT_BARS,
    )
    trader = PaperTrader()
    _close_stale_positions(trader)

    telegram_bot.send_message(
        f"🤖 Nifty bot started. PAPER_TRADING={config.PAPER_TRADING}, "
        f"angle≥{config.MIN_ANGLE_DEG}°, RR={config.RISK_REWARD_RATIO}, "
        f"SL {config.SL_MIN_POINTS:.0f}-{config.SL_MAX_POINTS:.0f} pts, "
        f"max {config.MAX_LOSSES_PER_DAY} losses/day, lot {config.LOT_SIZE}"
    )

    last_seen_candle = None
    last_day = None
    last_snapshot = None
    journal_done_day = None
    df = None

    while True:
        now = datetime.now(IST)
        today = now.date()

        # ── After the close: daily journal (once per trading day) ─────────
        if (market_calendar.is_trading_day(today) and journal_done_day != today
                and now.time() >= dtime(config.JOURNAL_HOUR, config.JOURNAL_MINUTE)):
            _write_journals(today)
            journal_done_day = today

        if not is_market_hours(now):
            time.sleep(60)
            continue

        # ── Daily reset ──────────────────────────────────────────────────
        if last_day != today:
            engine.reset_day()
            trader.new_day(today)
            last_day = today
            print(f"[{now:%H:%M:%S}] New day {today} — engine state reset")

        # ── 1. Always monitor open positions first ───────────────────────
        _monitor_all_positions(trader, now)

        # ── 2. Candles ───────────────────────────────────────────────────
        try:
            df = data_feed.get_recent_candles()
        except Exception as e:
            print(f"[{now:%H:%M:%S}] Candle fetch failed: {e}")
            time.sleep(config.POSITION_POLL_SECONDS if not trader.is_flat() else config.POLL_SECONDS)
            continue
        df = signal_engine.compute_indicators(
            df, config.FAST_EMA_LEN, config.SLOW_EMA_LEN, config.ATR_LEN)

        # ── 3. Periodic market snapshot ──────────────────────────────────
        if last_snapshot is None or (now - last_snapshot) >= timedelta(minutes=config.SNAPSHOT_EVERY_MINUTES):
            journal.record_snapshot(now, _safe_context(df, now, refresh_chain=True))
            last_snapshot = now

        latest = df.index[-1].tz_convert("UTC")

        # ── Same candle still forming ────────────────────────────────────
        if latest == last_seen_candle:
            if engine.armed:
                _check_ltp_retest(engine, df, trader, now)
            elif trader.is_flat() and engine.pending_dir == 0:
                _check_intrabar_crossover(df, now)
            if not trader.is_flat():
                sleep_secs = config.POSITION_POLL_SECONDS
            elif engine.armed:
                sleep_secs = config.LTP_POLL_SECONDS
            else:
                sleep_secs = config.POLL_SECONDS
            time.sleep(sleep_secs)
            continue

        # ── New closed candle ────────────────────────────────────────────
        last_seen_candle = latest
        print(f"[{now:%H:%M:%S}] New candle {df.index[-1]:%H:%M} close={df['close'].iloc[-1]:.2f} "
              f"fast={df['fast_ema'].iloc[-1]:.2f} slow={df['slow_ema'].iloc[-1]:.2f}")

        for event in engine.process(df, flat=True):
            _handle_event(event, trader, now, df)

        if is_force_close_time(now) and engine.armed:
            journal.record_signal("invalidated", engine.armed_dir, now, "eod_force_close",
                                  _safe_context(df, now))
            telegram_bot.send_message(
                f"⏰ EOD: armed {_dir_str(engine.armed_dir)} setup expired at force-close.")
            engine.reset_day()

        time.sleep(config.POSITION_POLL_SECONDS if not trader.is_flat() else config.POLL_SECONDS)


def _close_stale_positions(trader):
    """Positions left over from a previous day (bot was down at 15:15)."""
    now = datetime.now(IST)
    for pos in list(trader.positions):
        if datetime.fromisoformat(pos["entry_time"]).date() < now.date():
            info = trader._close(pos, pos["last_price"], now, "STALE_RESTART")
            telegram_bot.send_message(
                f"⚠️ Closed stale paper position {info['strike']} {info['option_type']} "
                f"from {info['date']} at last seen ₹{info['exit_premium']}.")


def _write_journals(today):
    try:
        try:
            df = data_feed.get_recent_candles()
        except Exception:
            df = None
        summary = journal.write_daily(today, df)
        print(summary)
        telegram_bot.send_message(summary)
        nxt = today + timedelta(days=1)
        while not market_calendar.is_trading_day(nxt):
            nxt += timedelta(days=1)
        if nxt.isocalendar()[1] != today.isocalendar()[1]:
            path = weekly_review.write_weekly(today)
            telegram_bot.send_message(f"📊 Weekly review written: {path}")
    except Exception as e:
        print(f"Journal error: {e}")
        telegram_bot.send_message(f"⚠️ Journal failed: {e}")


# ── Intrabar early-warning crossover check ──────────────────────────────────

_intrabar_warned_bar_ts = None


def _check_intrabar_crossover(df, now):
    """Informational Telegram alert when a crossover is forming in the live bar."""
    global _intrabar_warned_bar_ts
    try:
        token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
        resp = requests.get(
            "https://api.upstox.com/v3/historical-candle/intraday/NSE_INDEX%7CNifty%2050/minutes/5",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return
        candles = resp.json().get("data", {}).get("candles", [])
        if not candles:
            return
        live = max(candles, key=lambda c: c[0])
        if _intrabar_warned_bar_ts == live[0]:
            return
        idx = pd.Timestamp(live[0])
        idx = idx.tz_localize("Asia/Kolkata") if idx.tzinfo is None else idx.tz_convert("Asia/Kolkata")
        if idx <= df.index[-1]:
            return   # not a newer, in-progress bar
        row = pd.DataFrame({"open": [float(live[1])], "high": [float(live[2])],
                            "low": [float(live[3])], "close": [float(live[4])]}, index=[idx])
        d = signal_engine.compute_indicators(pd.concat([df[["open", "high", "low", "close"]], row]),
                                             config.FAST_EMA_LEN, config.SLOW_EMA_LEN, config.ATR_LEN)
        f, s = d["fast_ema"], d["slow_ema"]
        up = f.iloc[-2] < s.iloc[-2] and f.iloc[-1] > s.iloc[-1]
        down = f.iloc[-2] > s.iloc[-2] and f.iloc[-1] < s.iloc[-1]
        if up or down:
            _intrabar_warned_bar_ts = live[0]
            telegram_bot.send_message(
                f"⚡ Early warning: {'Bullish (CE)' if up else 'Bearish (PE)'} crossover forming. "
                f"Live {float(live[4]):.2f} — wait for bar close + angle check.")
    except Exception as e:
        print(f"[{now:%H:%M:%S}] Intrabar crossover check error: {e}")


def _check_ltp_retest(engine, df, trader, now):
    """Fire the entry the moment the live Nifty price touches the fast EMA."""
    if engine.wait_start_ts is not None:
        try:
            wait_pos = df.index.get_loc(engine.wait_start_ts)
        except KeyError:
            wait_pos = None
        if wait_pos is None or (len(df) - 1 - wait_pos) < 1:
            return   # arm candle hasn't been followed by a closed candle yet
    ltp = data_feed.get_nifty_ltp()
    if ltp is None:
        return
    fast_ema = float(df["fast_ema"].iloc[-1])
    slow_ema = float(df["slow_ema"].iloc[-1])
    buffer = fast_ema * config.RETEST_BUFFER_PCT / 100
    direction = engine.armed_dir
    if (direction == 1 and ltp <= fast_ema + buffer) or (direction == -1 and ltp >= fast_ema - buffer):
        print(f"[{now:%H:%M:%S}] 🎯 LTP retest: LTP={ltp:.2f} fast={fast_ema:.2f} ({_dir_str(direction)})")
        engine.armed, engine.armed_dir, engine.wait_start_ts = False, 0, None
        _process_entry(direction, float(ltp), slow_ema, trader, now, df, source="ltp_touch")


# ── Events ───────────────────────────────────────────────────────────────────

def _handle_event(event, trader, now, df):
    kind = event[0]

    if kind == "angle_check":
        _, direction, angle, passed = event[:4]
        reversed_ = event[4] if len(event) > 4 else False
        detail = f"angle {angle:.1f}°" + (" (EMA reversed)" if reversed_ and not passed else "")
        print(f"[{now:%H:%M:%S}] {_dir_str(direction)} crossover {detail} "
              f"{'PASSED' if passed else 'rejected'}")
        journal.record_signal("angle_pass" if passed else "angle_fail", direction, now, detail,
                              _safe_context(df, now))

    elif kind == "armed":
        direction = event[1]
        journal.record_signal("armed", direction, now, "waiting for fast-EMA retest",
                              _safe_context(df, now))
        telegram_bot.send_message(f"👀 {_dir_str(direction)} setup armed — waiting for retest.")

    elif kind == "invalidated":
        journal.record_signal("invalidated", 0, now, event[1], _safe_context(df, now))

    elif kind == "skipped_gap":
        journal.record_signal("skipped_gap", event[1], now, "first candle gapped", {})

    elif kind == "skipped_crossover":
        journal.record_signal("skipped_crossover", event[1], now, event[2], {})

    elif kind == "entry":
        _, direction, underlying_entry, slow_ema = event
        _process_entry(direction, underlying_entry, slow_ema, trader, now, df, source="candle_close")


def _process_entry(direction, underlying_entry, slow_ema, trader, now, df, source):
    dir_str = "CE (Buy Call)" if direction == 1 else "PE (Buy Put)"

    ok, reason = trader.can_open()
    if ok and is_force_close_time(now):
        ok, reason = False, "after_force_close_time"
    if not ok:
        journal.record_signal("entry_skipped", direction, now, reason, _safe_context(df, now))
        print(f"[{now:%H:%M:%S}] Entry signal {dir_str} skipped: {reason}")
        if reason != "position_open":
            telegram_bot.send_message(f"⏸ {dir_str} signal skipped — {reason}.")
        return

    try:
        strike = option_chain.find_strike_near_premium(direction, config.TARGET_PREMIUM)
    except Exception as e:
        journal.record_signal("entry_skipped", direction, now, f"strike: {e}", _safe_context(df, now))
        telegram_bot.send_message(f"⚠️ {dir_str} signal — no trade: {e}")
        return

    chain = dict(strike.get("chain_stats") or {}, expiry=strike.get("expiry"))
    ctx = _safe_context(df, now, chain=chain)
    signal_id = journal.record_signal(
        "entry_signal", direction, now,
        f"{source}; Nifty {underlying_entry:.2f}; {strike['strike']} {strike['option_type']} "
        f"@{strike['premium']} spread {strike['spread_pct']}%", ctx)

    u_stop = structure_stop(direction, df, slow_ema)
    risk, raw = premium_risk_points(direction, underlying_entry, u_stop, strike.get("delta"))
    msg = (f"📍 Entry signal: {dir_str}\n"
           f"Strike: {strike['strike']} {strike['option_type']} (exp {strike['expiry']})\n"
           f"Premium ₹{strike['premium']} | Bid/Ask ₹{strike['bid']}/₹{strike['ask']} "
           f"({strike['spread_pct']}%)\n"
           f"Delta {strike.get('delta')} | Stop {risk} pts (structure {raw}) | Target {risk * config.RISK_REWARD_RATIO:.1f} pts\n"
           f"(Paper trade — approve or auto-executes in {config.APPROVAL_TIMEOUT_SECONDS}s)")
    approved = telegram_bot.request_approval(
        msg, on_tick=lambda: _monitor_all_positions(trader, datetime.now(IST)))
    if not approved:
        journal.record_signal("entry_rejected", direction, now, "rejected via Telegram", ctx)
        return

    # Fresh quote at the moment of entry — the chain price may be a minute old
    quote = option_chain.get_option_quote(strike["instrument_key"]) or {
        "bid": strike["bid"], "ask": strike["ask"], "ltp": strike["premium"]}
    bid, ask = quote.get("bid") or 0, quote.get("ask") or 0
    if bid and ask and (ask - bid) / ((ask + bid) / 2) * 100 > config.MAX_SPREAD_PCT:
        journal.record_signal("entry_skipped", direction, now,
                              f"spread widened to {bid}/{ask} before fill", ctx)
        telegram_bot.send_message(f"⏸ {dir_str} skipped — spread widened ({bid}/{ask}).")
        return

    entry_now = datetime.now(IST)
    pos = trader.open_position(direction, strike, quote, underlying_entry, u_stop,
                               entry_now, context=ctx, signal_id=signal_id)
    journal.record_signal("trade_open", direction, entry_now,
                          f"{pos['strike']} {pos['option_type']} fill ₹{pos['entry_fill']} "
                          f"SL ₹{pos['sl_premium']} TGT ₹{pos['target_premium']}", ctx)
    telegram_bot.send_message(
        f"✅ PAPER trade opened: {pos['strike']} {pos['option_type']} @ ₹{pos['entry_fill']} (ask)\n"
        f"SL ₹{pos['sl_premium']} ({pos['risk_points']} pts) | Target ₹{pos['target_premium']}\n"
        f"Trailing: BE at +1R, +0.5R locked at +1.5R")


def _monitor_all_positions(trader, now):
    if trader.is_flat():
        return
    closed = trader.check_all_exits(option_chain.get_option_quote, now,
                                    force_close=is_force_close_time(now))
    for x in closed:
        emoji = {"WIN": "🟢", "LOSS": "🔴"}.get(x["result"], "⚪")
        journal.record_signal("trade_exit", x["direction"], now,
                              f"{x['exit_reason']} @ ₹{x['exit_premium']} net ₹{x['net_pnl']}", {})
        telegram_bot.send_message(
            f"{emoji} PAPER trade closed: {x['result']} ({x['exit_reason']})\n"
            f"{x['strike']} {x['option_type']}: ₹{x['entry_fill']} → ₹{x['exit_premium']} "
            f"in {x['minutes_held']} min\n"
            f"Gross ₹{x['gross_pnl']:,.0f} − costs ₹{x['charges']:,.0f} = net ₹{x['net_pnl']:,.0f}")
    if closed and trader.is_paused():
        telegram_bot.send_message(
            f"🛑 Daily limit reached ({trader.losses_today} losses, ₹{trader.pnl_today:,.0f}). "
            f"No new trades today; still monitoring.")


if __name__ == "__main__":
    main()
