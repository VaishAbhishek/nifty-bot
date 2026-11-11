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
import sys

import numpy as np
import pandas as pd

import config
import costs
from backtest_filters import load_data, add_features, is_fc, MIN_ANGLE_DEG, \
    ANGLE_CONFIRM_BARS, MAX_WAIT_BARS, TRAIN_FRACTION

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


def run(df, band, rr, trail_steps, gate):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    fast, slow, atr = df["fast"].values, df["slow"].values, df["atr"].values
    adx, sess, idx = df["adx"].values, df["sess_side"].values, df.index
    dates = idx.date
    lo_n = df["low"].rolling(config.SWING_LOOKBACK_BARS).min().values
    hi_n = df["high"].rolling(config.SWING_LOOKBACK_BARS).max().values
    sl_min, sl_max = band
    cost_unit = costs.cost_per_unit(ENTRY_PREMIUM)

    trades = []
    pending, cross_i, armed, wait_i = 0, None, 0, None
    pos = None
    losses_today, pnl_today = 0, 0.0

    def close(ts, prem_move, reason):
        nonlocal pos, losses_today, pnl_today
        hours = (ts - pos["time"]).total_seconds() / 3600
        exit_prem = max(ENTRY_PREMIUM + prem_move - THETA_PTS_PER_HOUR * hours - SLIPPAGE_PTS, 0.05)
        pnl = (exit_prem - ENTRY_PREMIUM) * LOT - costs.round_trip_charges(ENTRY_PREMIUM, exit_prem)
        trades.append({"entry_time": pos["time"], "exit_time": ts,
                       "direction": "CE" if pos["dir"] == 1 else "PE",
                       "risk_pts": pos["R"], "exit_reason": reason,
                       "pnl_rupees": round(pnl, 2),
                       "r_multiple": round(pnl / (pos["R"] * LOT), 2)})
        pnl_today += pnl
        if pnl < -config.SCRATCH_RUPEES:
            losses_today += 1
        pos = None

    for i in range(80, len(df)):
        ts = idx[i]
        new_day = dates[i] != dates[i - 1]
        if new_day:
            pending, cross_i, armed, wait_i = 0, None, 0, None
            losses_today, pnl_today = 0, 0.0

        if pos is not None:
            d, R = pos["dir"], pos["R"]
            if is_fc(ts):
                close(ts, (o[i] - pos["entry"]) * d * DELTA, "EOD")
                continue
            # premium move at this bar's worst and best index price
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
            continue

        blocked = (losses_today >= config.MAX_LOSSES_PER_DAY
                   or pnl_today <= -config.DAILY_MAX_LOSS_RUPEES)

        if armed and not new_day and not is_fc(ts) and i - 1 >= wait_i:
            p = i - 1
            limit = fast[p]
            touched = l[i] <= limit if armed == 1 else h[i] >= limit
            if touched:
                entry = min(limit, o[i]) if armed == 1 else max(limit, o[i])
                if armed == 1:
                    und_stop = min(lo_n[p], slow[p]) - config.SWING_BUFFER_POINTS
                else:
                    und_stop = max(hi_n[p], slow[p]) + config.SWING_BUFFER_POINTS
                R = min(max(abs(entry - und_stop) * DELTA, sl_min), sl_max)
                feat = {"adx": adx[p], "sess_aligned": int(sess[p] == armed),
                        "entry_hour": ts.hour + (0.5 if ts.minute >= 30 else 0.0)}
                d_armed = armed
                armed, wait_i = 0, None
                if not blocked and gate(feat):
                    pos = {"dir": d_armed, "entry": entry, "R": R, "stop": -R, "time": ts}
                    worst = ((l[i] if d_armed == 1 else h[i]) - entry) * d_armed * DELTA
                    if worst <= -R:
                        close(ts, -R, "SL")
                    continue

        up = fast[i - 1] < slow[i - 1] and fast[i] > slow[i]
        dn = fast[i - 1] > slow[i - 1] and fast[i] < slow[i]
        gap = new_day and o[i] != c[i - 1]
        if (up or dn) and not gap and not armed and pending == 0:
            pending, cross_i = (1 if up else -1), i
        if pending and i - cross_i == ANGLE_CONFIRM_BARS:
            slope = (fast[i] - fast[cross_i]) / ANGLE_CONFIRM_BARS
            ang = float(np.degrees(np.arctan(slope / atr[i]))) if atr[i] else 0.0
            if (pending == 1 and ang >= MIN_ANGLE_DEG) or (pending == -1 and ang <= -MIN_ANGLE_DEG):
                armed, wait_i = pending, i
            pending, cross_i = 0, None
        if armed:
            broken = c[i] < slow[i] if armed == 1 else c[i] > slow[i]
            if (i - wait_i) > MAX_WAIT_BARS or broken or is_fc(ts):
                armed, wait_i = 0, None

    return pd.DataFrame(trades)


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

    rows, kept = [], {}
    for (bn, band), rr, (tn, steps), (fn, gate) in itertools.product(
            STOP_BANDS.items(), RR_VALUES, TRAILS.items(), FILTERS.items()):
        t = run(df, band, rr, steps, gate)
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
    print("\nSaved: live_rules_results.csv, live_rules_monthly.csv, live_rules_trades.csv")


if __name__ == "__main__":
    main()
