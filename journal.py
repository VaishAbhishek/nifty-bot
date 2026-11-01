"""
Daily trading journal.

Records WHY things happened, not just what:
  - journal/signals.csv          every signal event (taken or not) + market context
  - paper_trades.csv             every closed trade + its entry context + MFE/MAE
  - journal/market_snapshots.csv context every SNAPSHOT_EVERY_MINUTES
  - journal/YYYY-MM-DD.md        readable daily page with rule-based explanations

Context captured (all computed from data, no AI):
  market structure  gap, prior-day high/low/close, opening range, 5m & 15m trend, day range
  volatility        India VIX (and change since open), ATR
  sentiment         put-call ratio, highest call/put OI strikes, ATM IV
  liquidity         bid-ask spread and OI of the traded strike (on trade rows)
  calendar          weekday, time bucket, days to expiry, scheduled events (events.csv)
"""

import csv
import os
from datetime import datetime, time as dtime

import pandas as pd

import config
import market_calendar

CONTEXT_FIELDS = [
    "weekday", "time_bucket", "spot", "gap_pct", "prev_close", "prev_high", "prev_low",
    "vs_prev_range", "dist_pdh_pct", "dist_pdl_pct", "or_high", "or_low", "vs_opening_range",
    "first30_dir", "trend_5m", "trend_15m", "atr", "day_range", "vix", "vix_change",
    "pcr", "max_call_oi_strike", "max_put_oi_strike", "atm_iv", "expiry", "days_to_expiry",
    "events",
]

SIGNAL_FIELDS = ["signal_id", "date", "time", "kind", "direction", "detail"] + CONTEXT_FIELDS

TRADE_FIELDS = [
    "trade_id", "signal_id", "date", "entry_time", "exit_time", "minutes_held",
    "direction", "option_type", "strike", "expiry", "instrument_key",
    "entry_ltp", "entry_fill", "entry_spread_pct", "delta", "iv", "oi",
    "underlying_entry", "underlying_stop", "risk_points", "initial_sl", "final_sl",
    "target_premium", "trail_stage", "exit_premium", "exit_reason",
    "mfe_points", "mae_points", "mfe_r", "gross_pnl", "charges", "net_pnl", "result",
] + CONTEXT_FIELDS

SNAPSHOT_FIELDS = ["date", "time"] + CONTEXT_FIELDS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _path(name):
    os.makedirs(config.JOURNAL_DIR, exist_ok=True)
    return os.path.join(config.JOURNAL_DIR, name)


def _append(path, fields, row):
    """Append a row; if an older file has a different header, archive it first."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header != fields:
            base, ext = os.path.splitext(path)
            os.replace(path, f"{base}_old_{datetime.now():%Y%m%d%H%M%S}{ext}")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def time_bucket(t: dtime):
    if t < dtime(9, 30):
        return "09:15-09:30"
    if t < dtime(11, 0):
        return "09:30-11:00"
    if t < dtime(13, 30):
        return "11:00-13:30"
    if t < dtime(14, 45):
        return "13:30-14:45"
    return "14:45-15:30"


def _r(x, n=2):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ── Market context ───────────────────────────────────────────────────────────

class ContextBuilder:
    """Builds the context dict. Keeps today's opening VIX to report the change."""

    def __init__(self):
        self.vix_open = None
        self.vix_day = None

    def build(self, df, now, vix=None, chain=None):
        ctx = {k: None for k in CONTEXT_FIELDS}
        today = now.date()
        ctx["weekday"] = now.strftime("%a")
        ctx["time_bucket"] = time_bucket(now.time())
        chain = chain or {}

        if vix:
            if self.vix_day != today:
                self.vix_day, self.vix_open = today, vix
            ctx["vix"] = _r(vix)
            ctx["vix_change"] = _r(vix - self.vix_open) if self.vix_open else None

        for k in ("pcr", "max_call_oi_strike", "max_put_oi_strike", "atm_iv"):
            ctx[k] = chain.get(k)

        expiry = chain.get("expiry")
        if expiry:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        else:
            exp_date = market_calendar.weekly_expiry_for(today)
        ctx["expiry"] = exp_date.isoformat()
        ctx["days_to_expiry"] = market_calendar.days_to_expiry(today, exp_date)
        ctx["events"] = "; ".join(market_calendar.events_for(today))

        if df is None or df.empty:
            return ctx

        dates = df.index.date
        today_df = df[dates == today]
        prev_df = df[dates < today]
        last_close = float(df["close"].iloc[-1])
        spot = float(chain.get("spot") or last_close)
        ctx["spot"] = _r(spot)
        if "atr" in df:
            ctx["atr"] = _r(df["atr"].iloc[-1])
        if "fast_ema" in df and "slow_ema" in df:
            ctx["trend_5m"] = "up" if df["fast_ema"].iloc[-1] > df["slow_ema"].iloc[-1] else "down"

        closes_15 = df["close"].resample("15min", origin="start_day", offset="15min").last().dropna()
        if len(closes_15) >= config.SLOW_EMA_LEN:
            f15 = closes_15.ewm(span=config.FAST_EMA_LEN, adjust=False).mean().iloc[-1]
            s15 = closes_15.ewm(span=config.SLOW_EMA_LEN, adjust=False).mean().iloc[-1]
            ctx["trend_15m"] = "up" if f15 > s15 else "down"

        if not prev_df.empty:
            last_prev_day = prev_df.index.date[-1]
            pday = prev_df[prev_df.index.date == last_prev_day]
            pc, ph, pl = float(pday["close"].iloc[-1]), float(pday["high"].max()), float(pday["low"].min())
            ctx.update(prev_close=_r(pc), prev_high=_r(ph), prev_low=_r(pl))
            ctx["vs_prev_range"] = "above PDH" if spot > ph else "below PDL" if spot < pl else "inside"
            ctx["dist_pdh_pct"] = _r((ph - spot) / spot * 100, 3)
            ctx["dist_pdl_pct"] = _r((spot - pl) / spot * 100, 3)
            if not today_df.empty:
                ctx["gap_pct"] = _r((float(today_df["open"].iloc[0]) - pc) / pc * 100, 3)

        if not today_df.empty:
            ctx["day_range"] = _r(today_df["high"].max() - today_df["low"].min())
            if len(today_df) >= 6:
                first30 = today_df.iloc[:6]
                oh, ol = float(first30["high"].max()), float(first30["low"].min())
                ctx.update(or_high=_r(oh), or_low=_r(ol))
                ctx["vs_opening_range"] = "above" if spot > oh else "below" if spot < ol else "inside"
                ref = ctx["prev_close"] or float(first30["open"].iloc[0])
                ctx["first30_dir"] = "up" if float(first30["close"].iloc[-1]) > ref else "down"
        return ctx


# ── Logs ─────────────────────────────────────────────────────────────────────

def record_signal(kind, direction, now, detail="", context=None):
    sid = f"S{now:%Y%m%d-%H%M%S}-{kind[:6]}"
    row = {"signal_id": sid, "date": now.date().isoformat(), "time": now.strftime("%H:%M:%S"),
           "kind": kind, "direction": {1: "CE", -1: "PE"}.get(direction, ""),
           "detail": detail}
    row.update(context or {})
    _append(_path("signals.csv"), SIGNAL_FIELDS, row)
    return sid


def record_trade(row):
    _append(config.TRADE_LOG_PATH, TRADE_FIELDS, row)


def record_snapshot(now, context):
    row = {"date": now.date().isoformat(), "time": now.strftime("%H:%M")}
    row.update(context)
    _append(_path("market_snapshots.csv"), SNAPSHOT_FIELDS, row)


# ── Explanations ─────────────────────────────────────────────────────────────

def explain_trade(t):
    """
    Rule-based reasons for a trade's outcome. Each line is tagged:
    [+] usually supportive for this direction, [-] usually a headwind, [i] info.
    """
    out = []
    long = t.get("option_type") == "CE"
    want = "up" if long else "down"
    net, gross, charges = _f(t.get("net_pnl")), _f(t.get("gross_pnl")), _f(t.get("charges"))
    mfe_r = _f(t.get("mfe_r"))
    spot = _f(t.get("spot"))

    reason = t.get("exit_reason")
    mins = t.get("minutes_held")
    out.append(f"[i] Exit: {reason} after {mins} min; best point +{t.get('mfe_points')} pts "
               f"({mfe_r}R), worst -{t.get('mae_points')} pts.")
    if reason == "TRAIL_SL":
        out.append(f"[+] Trailing stop ({t.get('trail_stage')}) locked the result instead of a full stop-out.")
    if mfe_r is not None and mfe_r >= 1 and t.get("result") != "WIN":
        out.append(f"[-] Trade reached +{mfe_r}R before reversing.")
    if mfe_r is not None and mfe_r < 0.3 and reason == "SL":
        out.append("[-] Went against the entry almost immediately (never reached +0.3R).")

    for label, key in (("15-min trend", "trend_15m"), ("First 30 min direction", "first30_dir")):
        v = t.get(key)
        if v:
            out.append(f"[{'+' if v == want else '-'}] {label} was {v} "
                       f"({'with' if v == want else 'against'} the trade).")

    vs = t.get("vs_prev_range")
    pdh, pdl = _f(t.get("dist_pdh_pct")), _f(t.get("dist_pdl_pct"))
    if long and pdh is not None and 0 <= pdh < 0.15:
        out.append("[-] Bought just under the prior-day high (common resistance).")
    elif not long and pdl is not None and 0 <= pdl < 0.15:
        out.append("[-] Sold just above the prior-day low (common support).")
    elif vs in ("above PDH", "below PDL"):
        out.append(f"[{'+' if (vs == 'above PDH') == long else '-'}] Price was {vs} "
                   f"(breakout zone).")

    wall = _f(t.get("max_call_oi_strike") if long else t.get("max_put_oi_strike"))
    if wall and spot:
        gap = (wall - spot) / spot * 100 if long else (spot - wall) / spot * 100
        if 0 <= gap < 0.5:
            kind = "call OI wall (resistance)" if long else "put OI wall (support)"
            out.append(f"[-] Biggest {kind} at {int(wall)}, only {gap:.2f}% away.")

    pcr = _f(t.get("pcr"))
    if pcr:
        bias = "put-heavy (often read as supportive)" if pcr > 1.2 else \
               "call-heavy (often read as capping)" if pcr < 0.8 else "neutral"
        out.append(f"[i] PCR {pcr} — {bias}.")

    sp = _f(t.get("entry_spread_pct"))
    if sp is not None and sp > 2:
        out.append(f"[-] Wide bid-ask spread at entry ({sp}%).")
    vix = _f(t.get("vix"))
    if vix:
        if vix < 12:
            out.append(f"[-] Low VIX ({vix}): small moves, targets harder to reach.")
        elif vix > 20:
            out.append(f"[-] High VIX ({vix}): premiums expensive, IV can deflate.")
        else:
            out.append(f"[i] VIX {vix} (change {t.get('vix_change')}).")
    if t.get("time_bucket") == "11:00-13:30":
        out.append("[-] Midday entry (often choppy).")
    if str(t.get("days_to_expiry")) == "0":
        out.append("[-] Expiry day: fastest time decay.")
    if t.get("events"):
        out.append(f"[i] Event day: {t['events']}.")
    if charges and gross:
        out.append(f"[i] Costs ₹{charges:.0f} = {abs(charges / gross) * 100:.0f}% of gross P&L.")
    return out


# ── Daily page ───────────────────────────────────────────────────────────────

def _day_type(today_df, prev_ranges):
    if today_df.empty:
        return "n/a"
    o, c = float(today_df["open"].iloc[0]), float(today_df["close"].iloc[-1])
    h, l = float(today_df["high"].max()), float(today_df["low"].min())
    rng = h - l
    if rng <= 0:
        return "flat"
    body = abs(c - o) / rng
    wide = prev_ranges and rng > 1.3 * (sum(prev_ranges) / len(prev_ranges))
    if body >= 0.6:
        return f"Trend day ({'up' if c > o else 'down'})" + (", wide range" if wide else "")
    if body <= 0.3:
        return "Two-sided / range day" + (", wide range" if wide else "")
    return f"Mixed day, leaning {'up' if c > o else 'down'}"


def write_daily(day, df=None):
    """Write journal/YYYY-MM-DD.md and return a short Telegram summary."""
    iso = day.isoformat()
    signals = [r for r in _read(_path("signals.csv")) if r["date"] == iso]
    trades = [r for r in _read(config.TRADE_LOG_PATH) if r.get("date") == iso]
    snaps = [r for r in _read(_path("market_snapshots.csv")) if r["date"] == iso]

    L = [f"# Journal {iso} ({day:%A})", ""]

    # Market summary
    L.append("## Market")
    if df is not None and not df.empty:
        dates = df.index.date
        tdf = df[dates == day]
        prev_days = sorted({d for d in dates if d < day})[-5:]
        prev_ranges = [float(df[dates == d]["high"].max() - df[dates == d]["low"].min())
                       for d in prev_days]
        if not tdf.empty:
            o, h = float(tdf["open"].iloc[0]), float(tdf["high"].max())
            l, c = float(tdf["low"].min()), float(tdf["close"].iloc[-1])
            pc = float(df[dates == prev_days[-1]]["close"].iloc[-1]) if prev_days else None
            chg = f"{(c - pc) / pc * 100:+.2f}%" if pc else "n/a"
            gap = f"{(o - pc) / pc * 100:+.2f}%" if pc else "n/a"
            L.append(f"- Nifty O {o:.0f} · H {h:.0f} · L {l:.0f} · C {c:.0f} ({chg}); gap {gap}; "
                     f"range {h - l:.0f} pts")
            L.append(f"- Day type: {_day_type(tdf, prev_ranges)}")
    if snaps:
        first, last = snaps[0], snaps[-1]
        L.append(f"- India VIX {first.get('vix') or '?'} → {last.get('vix') or '?'}; "
                 f"PCR {first.get('pcr') or '?'} → {last.get('pcr') or '?'}; "
                 f"ATM IV {first.get('atm_iv') or '?'} → {last.get('atm_iv') or '?'}")
        L.append(f"- Biggest OI at close: calls {last.get('max_call_oi_strike') or '?'} (resistance), "
                 f"puts {last.get('max_put_oi_strike') or '?'} (support)")
        L.append(f"- Expiry {last.get('expiry')} ({last.get('days_to_expiry')} trading days away)")
    events = market_calendar.events_for(day)
    L.append(f"- Scheduled events: {'; '.join(events) if events else 'none'}")
    L.append("")

    # Results
    net = sum(_f(t.get("net_pnl")) or 0 for t in trades)
    wins = sum(t.get("result") == "WIN" for t in trades)
    losses = sum(t.get("result") == "LOSS" for t in trades)
    L.append("## Results")
    L.append(f"- Trades: {len(trades)} (wins {wins}, losses {losses}); net P&L ₹{net:,.0f} per lot after costs")

    kinds = {}
    for s in signals:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    funnel = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    L.append(f"- Signal funnel: {funnel or 'no signals'}")
    L.append("")

    for i, t in enumerate(trades, 1):
        L.append(f"## Trade {i}: {t['option_type']} {t['strike']} — {t['result']} ₹{_f(t['net_pnl']):,.0f}")
        L.append(f"- {t['entry_time']} → {t['exit_time']} · fill ₹{t['entry_fill']} "
                 f"(LTP {t['entry_ltp']}, spread {t['entry_spread_pct']}%) → exit ₹{t['exit_premium']}")
        L.append(f"- Risk {t['risk_points']} pts, SL {t['initial_sl']} → {t['final_sl']}, "
                 f"target {t['target_premium']}, delta {t['delta']}, IV {t['iv']}")
        L.append("- Why:")
        for line in explain_trade(t):
            L.append(f"    - {line}")
        L.append("")

    L.append("## Signal timeline")
    if signals:
        for s in signals:
            L.append(f"- {s['time']} {s['kind']} {s['direction']} {s['detail']}".rstrip())
    else:
        L.append("- none")
    L.append("")

    # Preserve hand-written notes on regeneration
    md_path = _path(f"{iso}.md")
    notes = "- (add anything you noticed: news, how the day felt, mistakes)\n"
    if os.path.exists(md_path):
        with open(md_path) as f:
            old = f.read()
        if "## My notes" in old:
            notes = old.split("## My notes", 1)[1].lstrip("\n")
    L.append("## My notes")
    text = "\n".join(L) + "\n" + notes
    with open(md_path, "w") as f:
        f.write(text)

    summary = (f"📓 Journal {iso}: {len(trades)} trade(s), W{wins}/L{losses}, net ₹{net:,.0f}. "
               f"Signals: {funnel or 'none'}.")
    return summary
