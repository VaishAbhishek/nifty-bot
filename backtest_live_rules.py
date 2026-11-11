"""
Live-Rules Backtest — does the bot's CURRENT exit logic make money?
===================================================================
backtest_filters.py showed that a stop at the Slow EMA (about 6 Nifty
points) loses. The live bot no longer trades that way: it uses a swing
stop clamped to 10-15 premium points, a 2R target and a trailing stop.
Nothing had backtested those rules. This script does, and compares them
with a small, fixed set of alternatives.

For every variant it reports rupees per lot, net of the real cost model
in costs.py, month by month.

Usage (on the VM, after market hours):
  cd ~/nifty-bot
  set -a && source ~/.nifty_bot_env && set +a
  python3 backtest_live_rules.py

It reuses nifty_5min_cache.csv if backtest_filters.py already made it.

Outputs:
  live_rules_results.csv   every variant, ranked by net rupees per lot
  live_rules_monthly.csv   month-by-month P&L of the live config and the best variant
  live_rules_trades.csv    every trade of the live config

Entry signals come from the bot's own SignalEngine and config.py, replayed
the way main.py runs it (candle-close entries plus live-price touches), so
the trades should line up with paper_trades.csv. If paper_trades.csv is in
the folder, a trade-by-trade reality check is printed at the end.

Simplifications (read before trusting the numbers):
  * Option premium is modelled as entry premium + delta x index move,
    minus THETA_PTS_PER_HOUR of time decay. Real options also have gamma
    and IV changes, so individual trades will differ.
  * Entry premium is fixed at config.TARGET_PREMIUM, delta at DEFAULT_DELTA.
  * Within one 5-min candle the stop is assumed to be hit before the
    target or a trail step (worst case).
  * No slippage beyond the cost model. SLIPPAGE_PTS adds some.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

import config
import costs
from backtest_filters import load_data, add_features, is_fc, TRAIN_FRACTION
from signal_engine import SignalEngine, compute_indicators

DELTA               = config.DEFAULT_DELTA
ENTRY_PREMIUM       = config.TARGET_PREMIUM
LOT                 = config.LOT_SIZE
THETA_PTS_PER_HOUR  = 1.0    # assumed premium decay while holding (per unit)
SLIPPAGE_PTS        = 0.5    # assumed premium slippage per round trip (per unit)

# ── Variants tested (fixed in advance; do not keep adding until one wins) ──
STOP_BANDS = {"live 10-15": (config.SL_MIN_POINTS, config.SL_MAX_POINTS),
              "tighter 7-10": (7.0, 10.0),
              "wider 15-20": (15.0, 20.0)}
RR_VALUES  = [1.0, 1.5, 2.0, 3.0]
TRAILS     = {"trail on": config.TRAIL_STEPS, "trail off": []}
FILTERS    = {
    "no filter": lambda f: True,
    "ADX>=20":   lambda f: f["adx"] >= 20,
    "ADX>=20, right side of day avg, before 14:30":
        lambda f: f["adx"] >= 20 and f["sess_aligned"] == 1 and f["entry_hour"] < 14.5,
}
LIVE = ("live 10-15", config.RISK_REWARD_RATIO, "trail on", "no filter")


def find_entries(df):
    """
    Drive the bot's own SignalEngine over history and list every entry signal
    it would raise, the same two ways main.py does:
      * candle_close: engine.process() fires 'entry' when a closed candle's wick
        touched its Fast EMA (+ buffer). Entry at that candle's close.
      * ltp_touch: while armed, the live price touching the last closed Fast EMA
        (+ buffer) during the next candle. Entry at the touch price.
    Signals do not depend on whether a position is open (main.py passes
    flat=True), so they are computed once and reused for every variant.
    """
    ind = compute_indicators(df[["open", "high", "low", "close"]], config.FAST_EMA_LEN,
                             config.SLOW_EMA_LEN, config.ATR_LEN)
    engine = SignalEngine(config.MIN_ANGLE_DEG, config.ANGLE_CONFIRM_BARS,
                          config.RETEST_BUFFER_PCT, config.USE_WICK_FOR_RETEST,
                          config.MAX_WAIT_BARS)
    o, h, l = ind["open"].values, ind["high"].values, ind["low"].values
    fast, slow = ind["fast_ema"].values, ind["slow_ema"].values
    idx, dates = ind.index, ind.index.date
    n_sw = config.SWING_LOOKBACK_BARS
    lo_n = ind["low"].rolling(n_sw).min().values
    hi_n = ind["high"].rolling(n_sw).max().values
    adx, sess = df["adx"].values, df["sess_side"].values
    five = pd.Timedelta(minutes=5)

    def stop_level(d, p, slow_val):
        if d == 1:
            return min(lo_n[p], slow_val) - config.SWING_BUFFER_POINTS
        return max(hi_n[p], slow_val) + config.SWING_BUFFER_POINTS

    def feat(d, p, when):
        return {"adx": adx[p], "sess_aligned": int(sess[p] == d),
                "entry_hour": when.hour + (0.5 if when.minute >= 30 else 0.0)}

    out = []
    for i in range(80, len(ind)):
        ts = idx[i]
        if dates[i] != dates[i - 1]:
            engine.reset_day()

        # live price touch during candle i, against the last CLOSED candle (i-1)
        if engine.armed and engine.wait_start_ts is not None and dates[i] == dates[i - 1]:
            wait_pos = idx.get_loc(engine.wait_start_ts)
            if (i - 1 - wait_pos) >= 1 and not is_fc(ts):
                d, p = engine.armed_dir, i - 1
                level = fast[p] + d * fast[p] * config.RETEST_BUFFER_PCT / 100
                touched = l[i] <= level if d == 1 else h[i] >= level
                if touched:
                    entry = min(level, o[i]) if d == 1 else max(level, o[i])
                    out.append({"bar": i, "intrabar": True, "time": ts, "dir": d,
                                "entry": entry, "und_stop": stop_level(d, p, slow[p]),
                                "feat": feat(d, p, ts), "source": "ltp_touch"})
                    engine.armed, engine.armed_dir, engine.wait_start_ts = False, 0, None

        # candle i closes
        window = ind.iloc[max(0, i - 300): i + 1]
        close_time = ts + five
        for ev in engine.process(window, flat=True):
            if ev[0] == "entry" and not is_fc(close_time):
                _, d, entry, slow_val = ev
                out.append({"bar": i, "intrabar": False, "time": close_time, "dir": d,
                            "entry": entry, "und_stop": stop_level(d, i, slow_val),
                            "feat": feat(d, i, close_time), "source": "candle_close"})
        if is_fc(close_time) and engine.armed:
            engine.reset_day()
    return out


def run(df, entries, band, rr, trail_steps, gate):
    o, h, l = (df[k].values for k in ["open", "high", "low"])
    idx = df.index
    dates = idx.date
    sl_min, sl_max = band
    cost_unit = costs.cost_per_unit(ENTRY_PREMIUM)
    by_bar = {}
    for e in entries:
        by_bar.setdefault(e["bar"], []).append(e)

    trades = []
    pos = None
    losses_today, pnl_today = 0, 0.0

    def close(ts, prem_move, reason):
        nonlocal pos, losses_today, pnl_today
        hours = max((ts - pos["time"]).total_seconds(), 0) / 3600
        exit_prem = max(ENTRY_PREMIUM + prem_move - THETA_PTS_PER_HOUR * hours - SLIPPAGE_PTS, 0.05)
        pnl = (exit_prem - ENTRY_PREMIUM) * LOT - costs.round_trip_charges(ENTRY_PREMIUM, exit_prem)
        trades.append({"entry_time": pos["time"], "exit_time": ts,
                       "direction": "CE" if pos["dir"] == 1 else "PE",
                       "source": pos["source"], "risk_pts": round(pos["R"], 2),
                       "exit_reason": reason, "pnl_rupees": round(pnl, 2),
                       "r_multiple": round(pnl / (pos["R"] * LOT), 2)})
        pnl_today += pnl
        if pnl < -config.SCRATCH_RUPEES:
            losses_today += 1
        pos = None

    def manage(i, ts):
        """Stop / target / trail on candle i. Stop is checked first (worst case)."""
        d, R = pos["dir"], pos["R"]
        worst = ((l[i] if d == 1 else h[i]) - pos["entry"]) * d * DELTA
        best = ((h[i] if d == 1 else l[i]) - pos["entry"]) * d * DELTA
        if worst <= pos["stop"]:
            close(ts, pos["stop"], "SL" if pos["stop"] < 0 else "Trail")
        elif best >= rr * R:
            close(ts, rr * R, "Target")
        else:
            for at_r, lock_r in trail_steps:
                if best >= at_r * R:
                    pos["stop"] = max(pos["stop"], lock_r * R + (cost_unit if lock_r == 0 else 0))

    def try_open(e):
        nonlocal pos
        if pos is not None:
            return
        if losses_today >= config.MAX_LOSSES_PER_DAY or pnl_today <= -config.DAILY_MAX_LOSS_RUPEES:
            return
        if not gate(e["feat"]):
            return
        R = min(max(abs(e["entry"] - e["und_stop"]) * DELTA, sl_min), sl_max)
        pos = {"dir": e["dir"], "entry": e["entry"], "R": R, "stop": -R,
               "time": e["time"], "source": e["source"]}

    for i in range(80, len(df)):
        ts = idx[i]
        if dates[i] != dates[i - 1]:
            losses_today, pnl_today = 0, 0.0
        if pos is not None:
            if is_fc(ts):
                close(ts, (o[i] - pos["entry"]) * pos["dir"] * DELTA, "EOD")
            else:
                manage(i, ts + pd.Timedelta(minutes=5))
        for e in by_bar.get(i, []):
            if e["intrabar"]:
                try_open(e)
                if pos is not None and pos["time"] == e["time"]:
                    # rest of the entry candle: only the stop can be judged safely
                    worst = ((l[i] if e["dir"] == 1 else h[i]) - e["entry"]) * e["dir"] * DELTA
                    if worst <= -pos["R"]:
                        close(ts + pd.Timedelta(minutes=5), -pos["R"], "SL")
            else:
                try_open(e)

    return pd.DataFrame(trades)


def reconcile(bt, path=None):
    """Line the backtest up against the bot's real paper trades, trade by trade."""
    path = path or config.TRADE_LOG_PATH
    if not os.path.exists(path) or bt.empty:
        return
    try:
        p = pd.read_csv(path, on_bad_lines="skip")
        p = p[pd.to_numeric(p["net_pnl"], errors="coerce").notna()].copy()
        p["t"] = pd.to_datetime(p["date"].astype(str) + " " + p["entry_time"].astype(str)
                                ).dt.tz_localize("Asia/Kolkata")
        p["dir"] = p["option_type"]
    except Exception as e:
        print(f"\n(Could not read {path} for reconciliation: {e})")
        return
    if p.empty:
        return
    b = bt.copy()
    b["t"] = pd.to_datetime(b["entry_time"])
    last_data = b["t"].max().normalize() + pd.Timedelta(days=1)
    b = b[b["t"] >= p["t"].min().normalize()]
    used, rows = set(), []
    for _, r in p.iterrows():
        cand = b[(b["direction"] == r["dir"]) & ((b["t"] - r["t"]).abs() <= pd.Timedelta(minutes=20))
                 & (~b.index.isin(used))]
        if len(cand):
            k = (cand["t"] - r["t"]).abs().idxmin()
            used.add(k)
            rows.append([f"{r['t']:%m-%d %H:%M}", r["dir"], r["exit_reason"], round(float(r["net_pnl"])),
                         f"{b.loc[k, 't']:%H:%M}", b.loc[k, "exit_reason"], round(b.loc[k, "pnl_rupees"])])
        else:
            note = "after backtest data ends" if r["t"] >= last_data else "NOT IN BACKTEST"
            rows.append([f"{r['t']:%m-%d %H:%M}", r["dir"], r["exit_reason"], round(float(r["net_pnl"])),
                         "-", note, 0])
    for k, r in b[~b.index.isin(used)].iterrows():
        rows.append(["-", r["direction"], "BACKTEST ONLY", 0, f"{r['t']:%m-%d %H:%M}",
                     r["exit_reason"], round(r["pnl_rupees"])])
    rec = pd.DataFrame(rows, columns=["paper_entry", "type", "paper_exit", "paper_net",
                                      "backtest_entry", "backtest_exit", "backtest_net"])
    rec.to_csv("live_rules_reconcile.csv", index=False)
    print("\n" + "=" * 110)
    print("  REALITY CHECK — live config backtest vs your actual paper trades")
    print("=" * 110)
    print(rec.to_string(index=False))
    both = rec[(rec.paper_entry != "-") & (rec.backtest_entry != "-")]
    print(f"\n  Matched {len(both)} of {int((rec.paper_entry != '-').sum())} paper trades. "
          f"On matched trades: paper ₹{both.paper_net.sum():,.0f} vs backtest ₹{both.backtest_net.sum():,.0f}.")


def monthly(t):
    m = pd.to_datetime(t["entry_time"]).dt.strftime("%Y-%m")
    return t.groupby(m)["pnl_rupees"].sum().round(0)


def stats(t, n_months, cutoff):
    if len(t) == 0:
        return None
    win = t[t.pnl_rupees > config.SCRATCH_RUPEES]
    loss = t[t.pnl_rupees < -config.SCRATCH_RUPEES]
    gw, gl = win.pnl_rupees.sum(), -loss.pnl_rupees.sum()
    cum = t.pnl_rupees.cumsum()
    mo = monthly(t)
    d = pd.to_datetime(t["entry_time"]).dt.date
    early, late = t[d <= cutoff].pnl_rupees.sum(), t[d > cutoff].pnl_rupees.sum()
    # longest run of losing trades
    streak = best = 0
    for x in t.pnl_rupees:
        streak = streak + 1 if x < -config.SCRATCH_RUPEES else 0
        best = max(best, streak)
    return {"trades": len(t), "win_rate": round(len(win) / len(t) * 100, 1),
            "profit_factor": round(gw / gl, 2) if gl > 0 else np.inf,
            "net_rupees_per_lot": round(t.pnl_rupees.sum(), 0),
            "avg_per_month": round(t.pnl_rupees.sum() / n_months, 0),
            "worst_month": mo.min(), "losing_months": int((mo < 0).sum()),
            "months": len(mo), "max_drawdown": round((cum - cum.cummax()).min(), 0),
            "longest_losing_streak": best,
            "early_period": round(early, 0), "late_period": round(late, 0)}


def main():
    df = add_features(load_data())
    days = sorted(set(df.index.date))
    cutoff = days[int(len(days) * TRAIN_FRACTION) - 1]
    n_months = len(days) / 21.0
    print(f"{len(days)} trading days (~{n_months:.1f} months). "
          f"Early period ends {cutoff}.\n")

    entries = find_entries(df)
    print(f"{len(entries)} entry signals from the bot's own SignalEngine "
          f"({sum(e['source'] == 'ltp_touch' for e in entries)} live-price touches, "
          f"{sum(e['source'] == 'candle_close' for e in entries)} candle closes).\n")
    rows, kept = [], {}
    for (bn, band), rr, (tn, steps), (fn, gate) in itertools.product(
            STOP_BANDS.items(), RR_VALUES, TRAILS.items(), FILTERS.items()):
        t = run(df, entries, band, rr, steps, gate)
        s = stats(t, n_months, cutoff)
        if s is None:
            continue
        key = (bn, rr, tn, fn)
        kept[key] = t
        rows.append({"stop_band": bn, "rr": rr, "trail": tn, "filter": fn,
                     "is_live_config": key == LIVE, **s,
                     "holds_up": "yes" if s["early_period"] > 0 and s["late_period"] > 0 else "no"})
    if not rows:
        print("No trades. Check data."); sys.exit(1)

    res = pd.DataFrame(rows).sort_values("net_rupees_per_lot", ascending=False)
    res.to_csv("live_rules_results.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cols = ["stop_band", "rr", "trail", "filter", "trades", "win_rate", "profit_factor",
            "avg_per_month", "worst_month", "losing_months", "max_drawdown",
            "longest_losing_streak", "holds_up"]
    live = res[res.is_live_config]
    print("=" * 110)
    print("  YOUR LIVE CONFIG (rupees per lot, net of costs)")
    print("=" * 110)
    print(live[cols].to_string(index=False) if len(live) else "  live config produced no trades")
    print("\n" + "=" * 110)
    print("  TOP 10 VARIANTS  —  only trust rows where holds_up = yes")
    print("=" * 110)
    print(res[cols].head(10).to_string(index=False))
    print(f"\n  {int((res.net_rupees_per_lot > 0).sum())} of {len(res)} variants were profitable; "
          f"{int(((res.net_rupees_per_lot > 0) & (res.holds_up == 'yes')).sum())} "
          f"were profitable in both the early and late period.")

    out = {}
    if LIVE in kept:
        kept[LIVE].to_csv("live_rules_trades.csv", index=False)
        out["live_config"] = monthly(kept[LIVE])
    b = res.iloc[0]
    out["best_variant"] = monthly(kept[(b.stop_band, b.rr, b.trail, b["filter"])])
    mt = pd.DataFrame(out).fillna(0)
    mt.to_csv("live_rules_monthly.csv")
    print("\n  MONTH BY MONTH (rupees per lot)")
    print(mt.to_string())
    if LIVE in kept:
        reconcile(kept[LIVE])
    print("\nSaved: live_rules_results.csv, live_rules_monthly.csv, live_rules_trades.csv, live_rules_reconcile.csv")


if __name__ == "__main__":
    main()
