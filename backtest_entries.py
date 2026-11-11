"""
Entry Lab — is a different entry signal better than the EMA crossover retest?
=============================================================================
backtest_live_rules.py showed that no stop / target / trail / filter setting
makes the current entry profitable over the test period. This script keeps
the exits and risk rules the same and swaps the ENTRY, for a short list of
ideas fixed in advance:

  1. current        EMA 9/15 crossover + angle + retest (the bot's own SignalEngine)
  2. current midday the same signal, entries 11:30-13:30 only
  3. orb            Opening-range breakout: first 5-min close outside the
                    09:15-09:45 range, between 09:45 and 14:30. One trade a day.
  4. trend pullback 15-min trend up (or down) with ADX >= 20, price dips below
                    the 9 EMA, then closes back above it and above the prior
                    candle's high (mirror for shorts). Min 6 candles between signals.

Each entry is run with the live stop band (config.SL_MIN/MAX_POINTS), the
daily loss limits and real costs, at 1.5R / 2R / 3R, trailing stop on and off.
Results are in rupees per lot. A row only counts if it made money in BOTH
the early and the late part of the data ("holds_up = yes").

Usage (on the VM, after market hours):
  cd ~/nifty-bot
  set -a && source ~/.nifty_bot_env && set +a
  python3 backtest_entries.py

Outputs:
  entries_results.csv   every entry x exit combination
  entries_monthly.csv   month-by-month P&L of the best exit for each entry

Same option-price simplifications as backtest_live_rules.py.
"""

import itertools
import sys

import numpy as np
import pandas as pd

import config
from backtest_filters import load_data, add_features, is_fc, TRAIN_FRACTION
from backtest_live_rules import find_entries, run, stats, monthly

BAND      = (config.SL_MIN_POINTS, config.SL_MAX_POINTS)
RR_VALUES = [1.5, 2.0, 3.0]
TRAILS    = {"trail on": config.TRAIL_STEPS, "trail off": []}
FIVE      = pd.Timedelta(minutes=5)
NO_GATE   = lambda f: True


def _feat(df, i, d, when):
    return {"adx": df["adx"].values[i], "sess_aligned": int(df["sess_side"].values[i] == d),
            "entry_hour": when.hour + (0.5 if when.minute >= 30 else 0.0)}


def _swing_stop(df, i, d):
    n = config.SWING_LOOKBACK_BARS
    lo = df["low"].values[max(0, i - n + 1): i + 1].min()
    hi = df["high"].values[max(0, i - n + 1): i + 1].max()
    return lo - config.SWING_BUFFER_POINTS if d == 1 else hi + config.SWING_BUFFER_POINTS


def entries_midday(base):
    return [e for e in base if 11.5 <= e["time"].hour + e["time"].minute / 60 < 13.5]


def entries_orb(df):
    """First 5-min close outside the first 30 minutes' range. One trade a day."""
    idx, dates = df.index, df.index.date
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    mins = idx.hour * 60 + idx.minute
    out, day, or_hi, or_lo, done = [], None, None, None, False
    for i in range(80, len(df)):
        if dates[i] != day:
            day, or_hi, or_lo, done = dates[i], -np.inf, np.inf, False
        if mins[i] < 585:                       # 09:15-09:40 candles build the range
            or_hi, or_lo = max(or_hi, h[i]), min(or_lo, l[i])
            continue
        if done or mins[i] >= 870 or not np.isfinite(or_hi) or not np.isfinite(or_lo):
            continue                            # one trade a day; none after 14:30
        d = 1 if c[i] > or_hi else (-1 if c[i] < or_lo else 0)
        if d:
            when = idx[i] + FIVE
            out.append({"bar": i, "intrabar": False, "time": when, "dir": d, "entry": c[i],
                        "und_stop": (or_hi + or_lo) / 2, "feat": _feat(df, i, d, when),
                        "source": "orb"})
            done = True
    return out


def entries_trend_pullback(df):
    """With the 15-min trend: dip through the 9 EMA, then a close back beyond it."""
    idx, dates = df.index, df.index.date
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    fast, htf, adx = df["fast"].values, df["htf_dir"].values, df["adx"].values
    out, last = [], -99
    for i in range(80, len(df)):
        when = idx[i] + FIVE
        if dates[i] != dates[i - 3] or is_fc(when) or i - last < 6:
            continue                            # need 3 earlier candles today; cooldown
        d = int(htf[i]) if not np.isnan(htf[i]) else 0
        if d == 0 or not adx[i] >= 20:
            continue
        if d == 1:
            dipped = any(c[j] < fast[j] for j in (i - 3, i - 2, i - 1))
            trigger = c[i] > fast[i] and c[i] > h[i - 1]
        else:
            dipped = any(c[j] > fast[j] for j in (i - 3, i - 2, i - 1))
            trigger = c[i] < fast[i] and c[i] < l[i - 1]
        if dipped and trigger:
            out.append({"bar": i, "intrabar": False, "time": when, "dir": d, "entry": c[i],
                        "und_stop": _swing_stop(df, i, d), "feat": _feat(df, i, d, when),
                        "source": "trend_pullback"})
            last = i
    return out


def main():
    df = add_features(load_data())
    days = sorted(set(df.index.date))
    cutoff = days[int(len(days) * TRAIN_FRACTION) - 1]
    n_months = len(days) / 21.0
    print(f"{len(days)} trading days (~{n_months:.1f} months). Early period ends {cutoff}.\n")

    base = find_entries(df)
    entry_sets = {
        "1 current (EMA retest)":   base,
        "2 current, midday only":   entries_midday(base),
        "3 opening-range breakout": entries_orb(df),
        "4 15-min trend pullback":  entries_trend_pullback(df),
    }
    for name, es in entry_sets.items():
        print(f"  {name}: {len(es)} signals")
    print()

    rows, kept = [], {}
    for (name, es), rr, (tn, steps) in itertools.product(
            entry_sets.items(), RR_VALUES, TRAILS.items()):
        t = run(df, es, BAND, rr, steps, NO_GATE)
        s = stats(t, n_months, cutoff)
        if s is None:
            continue
        kept[(name, rr, tn)] = t
        rows.append({"entry": name, "rr": rr, "trail": tn, **s,
                     "holds_up": "yes" if s["early_period"] > 0 and s["late_period"] > 0 else "no"})
    if not rows:
        print("No trades. Check data."); sys.exit(1)

    res = pd.DataFrame(rows).sort_values(["entry", "net_rupees_per_lot"], ascending=[True, False])
    res.to_csv("entries_results.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cols = ["entry", "rr", "trail", "trades", "win_rate", "profit_factor", "net_rupees_per_lot",
            "avg_per_month", "worst_month", "losing_months", "max_drawdown",
            "early_period", "late_period", "holds_up"]
    print("=" * 120)
    print("  ALL COMBINATIONS (rupees per lot, net of costs) — only trust holds_up = yes")
    print("=" * 120)
    print(res[cols].to_string(index=False))

    out = {}
    for name in entry_sets:
        sub = res[res.entry == name]
        if len(sub):
            b = sub.iloc[0]
            out[f"{name} ({b.rr}R, {b.trail})"] = monthly(kept[(name, b.rr, b.trail)])
    mt = pd.DataFrame(out).fillna(0)
    mt.to_csv("entries_monthly.csv")
    print("\n  MONTH BY MONTH — best exit for each entry (rupees per lot)")
    print(mt.to_string())
    ok = res[(res.holds_up == "yes")]
    print(f"\n  {len(ok)} of {len(res)} combinations made money in both periods.")
    print("Saved: entries_results.csv, entries_monthly.csv")


if __name__ == "__main__":
    main()
