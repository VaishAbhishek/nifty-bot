"""
Filter Lab — which chop / reversal filters actually raise the win ratio?
========================================================================
Runs the EMA 9/15 + angle + retest strategy on Nifty 5-min history and
measures, for every trade, the market conditions at the moment of entry.
Then answers two questions:

  1. BUCKETS: how do win rate and profit factor change as each condition
     changes?  (e.g. trades taken with ADX < 15 vs ADX > 25)
  2. RULES: if we refuse to trade unless a condition is met, what happens
     to trade count, win rate, profit factor and total points?
     Each rule is also scored on an early period and a late period
     separately, so a filter that only "worked" by luck shows up.

It does not touch the live bot. Nothing here changes config.py.

Usage (on the VM):
  cd ~/nifty-bot
  set -a && source ~/.nifty_bot_env && set +a
  python3 backtest_filters.py

Outputs:
  filter_trades.csv          every baseline trade with its entry conditions
  filter_feature_buckets.csv win rate / PF by bucket of each condition
  filter_rules_results.csv   each filter rule vs the unfiltered baseline

All P&L is in Nifty index points, net of COST_POINTS per trade.
No future data is used: conditions are read from the last CLOSED candle
before the entry candle.
"""

import os
import sys

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
FAST_LEN, SLOW_LEN, ATR_LEN = 9, 15, 14
MIN_ANGLE_DEG      = 7
ANGLE_CONFIRM_BARS = 2
MAX_WAIT_BARS      = 15
RR_VALUES          = [1.0, 1.5, 2.0]   # each RR gets its own set of results
FORCE_CLOSE        = (15, 15)

# Round-trip cost in INDEX points (brokerage + taxes + slippage on the option,
# converted at ~0.45 delta). This is an assumption — adjust to your broker.
COST_POINTS        = 2.0

LOOKBACK_MONTHS    = 6
CACHE_FILE         = "nifty_5min_cache.csv"
TRAIN_FRACTION     = 0.67   # first 67% of days = "early", rest = "late"
MIN_TRADES_BUCKET  = 8      # buckets with fewer trades are flagged, not trusted


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        print(f"Loaded {len(df)} cached candles from {CACHE_FILE} "
              f"(delete the file to re-download)")
        return df
    from backtest import fetch_historical_data
    df = fetch_historical_data(months=LOOKBACK_MONTHS)
    df.to_csv(CACHE_FILE)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS + ENTRY CONDITIONS (features)
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    day = pd.Series(df.index.date, index=df.index)

    df["fast"] = c.ewm(span=FAST_LEN, adjust=False).mean()
    df["slow"] = c.ewm(span=SLOW_LEN, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()

    # ADX(14), Wilder smoothing — trend strength, direction-agnostic
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = 1 / 14
    atr14 = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr14
    mdi = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=a, adjust=False).mean()

    # Choppiness Index(14): ~100 = pure chop, ~0 = straight-line trend
    rng = h.rolling(14).max() - l.rolling(14).min()
    df["chop"] = 100 * np.log10(tr.rolling(14).sum() / rng.replace(0, np.nan)) / np.log10(14)

    # EMA separation in ATR units — flat, braided EMAs give small values
    df["ema_spread_atr"] = (df["fast"] - df["slow"]).abs() / df["atr"]

    # Higher-timeframe trend: EMA(60) on 5-min ≈ EMA(20) on 15-min
    df["ema60"] = c.ewm(span=60, adjust=False).mean()
    df["htf_dir"] = np.sign(c - df["ema60"]) * (
        np.sign(df["ema60"] - df["ema60"].shift(6)) == np.sign(c - df["ema60"]))

    # Session average price. The index has no volume, so this is an
    # equal-weighted stand-in for VWAP (true VWAP needs Nifty futures data).
    tp = (h + l + c) / 3
    df["sess_avg"] = tp.groupby(day).expanding().mean().reset_index(level=0, drop=True)
    df["sess_side"] = np.sign(c - df["sess_avg"])

    # Opening range = first 30 minutes (09:15–09:45)
    mins = df.index.hour * 60 + df.index.minute
    in_or = (mins >= 555) & (mins < 585)
    or_hi = h.where(in_or).groupby(day).transform("max")
    or_lo = l.where(in_or).groupby(day).transform("min")
    df["or_pos"] = np.where(mins < 585, 0, np.where(c > or_hi, 1, np.where(c < or_lo, -1, 0)))

    # Whipsaw count: EMA crossovers so far today (many crosses = choppy day)
    cross = (np.sign(df["fast"] - df["slow"]).diff().abs() > 0).astype(int)
    df["crosses_today"] = cross.groupby(day).cumsum()

    # Day range used so far vs the average full-day range of the last 10 days
    day_hi = h.groupby(day).cummax()
    day_lo = l.groupby(day).cummin()
    daily = pd.DataFrame({"h": h.groupby(day).max(), "l": l.groupby(day).min(),
                          "c": c.groupby(day).last()})
    daily["adr10"] = (daily["h"] - daily["l"]).rolling(10).mean().shift(1)
    # Previous day levels: CPR width and prior high / low
    pdh, pdl, pdc = daily["h"].shift(1), daily["l"].shift(1), daily["c"].shift(1)
    pivot = (pdh + pdl + pdc) / 3
    bc = (pdh + pdl) / 2
    daily["cpr_width_pct"] = (2 * (pivot - bc)).abs() / pivot * 100
    daily["pdh"], daily["pdl"] = pdh, pdl
    for col in ["adr10", "cpr_width_pct", "pdh", "pdl"]:
        df[col] = day.map(daily[col]).values
    df["range_used"] = (day_hi - day_lo) / df["adr10"]

    # Opening gap size (% vs previous close)
    first_open = o.groupby(day).transform("first")
    df["gap_pct"] = (first_open - day.map(pdc).values).abs() / day.map(pdc).values * 100
    return df


FEATURES = ["adx", "chop", "ema_spread_atr", "angle_abs", "sl_atr", "crosses_today",
            "range_used", "cpr_width_pct", "gap_pct", "room_atr", "entry_hour",
            "htf_aligned", "sess_aligned", "or_aligned"]


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST (mirrors the Pine Script: limit order at the last closed Fast EMA)
# ─────────────────────────────────────────────────────────────────────────────
def is_fc(ts):
    return (ts.hour, ts.minute) >= FORCE_CLOSE


def run(df, rr, gate=None):
    """gate(features_dict) -> True to take the trade. None = take everything."""
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    fast, slow, atr = df["fast"].values, df["slow"].values, df["atr"].values
    idx = df.index
    dates = idx.date
    trades = []
    pending, cross_i, armed, wait_i, angle_at_arm = 0, None, 0, None, 0.0
    pos = None

    for i in range(80, len(df)):
        ts = idx[i]
        new_day = dates[i] != dates[i - 1]
        if new_day:
            pending, cross_i, armed, wait_i = 0, None, 0, None

        # ── manage open position ──
        if pos is not None:
            d = pos["dir"]
            exit_px = reason = None
            if is_fc(ts):
                exit_px, reason = o[i], "EOD"
            elif d == 1:
                if l[i] <= pos["sl"]:
                    exit_px, reason = min(pos["sl"], o[i]), "SL"
                elif h[i] >= pos["tp"]:
                    exit_px, reason = pos["tp"], "Target"
            else:
                if h[i] >= pos["sl"]:
                    exit_px, reason = max(pos["sl"], o[i]), "SL"
                elif l[i] <= pos["tp"]:
                    exit_px, reason = pos["tp"], "Target"
            if reason:
                pnl = (exit_px - pos["entry"]) * d - COST_POINTS
                trades.append({**pos["feat"], "entry_time": pos["time"], "exit_time": ts,
                               "direction": "CE" if d == 1 else "PE",
                               "entry": round(pos["entry"], 2), "exit": round(exit_px, 2),
                               "exit_reason": reason, "pnl_points": round(pnl, 2),
                               "win": int(pnl > 0)})
                pos = None
            continue

        # ── entry: limit order resting at the previous bar's Fast EMA ──
        if armed and not new_day and not is_fc(ts) and i - 1 >= wait_i:
            p = i - 1
            limit, sl = fast[p], slow[p]
            touched = l[i] <= limit if armed == 1 else h[i] >= limit
            if touched:
                entry = min(limit, o[i]) if armed == 1 else max(limit, o[i])
                dist = max(abs(limit - sl), 1.0)
                tp = limit + armed * rr * dist
                row = df.iloc[p]
                room = (row["pdh"] - entry) if armed == 1 else (entry - row["pdl"])
                feat = {
                    "adx": row["adx"], "chop": row["chop"],
                    "ema_spread_atr": row["ema_spread_atr"],
                    "angle_abs": abs(angle_at_arm), "sl_atr": dist / atr[p],
                    "crosses_today": row["crosses_today"], "range_used": row["range_used"],
                    "cpr_width_pct": row["cpr_width_pct"], "gap_pct": row["gap_pct"],
                    # distance to prior-day high (CE) / low (PE) in ATRs; negative = already beyond it
                    "room_atr": room / atr[p],
                    "entry_hour": ts.hour + (0.5 if ts.minute >= 30 else 0.0),
                    "htf_aligned": int(row["htf_dir"] == armed),
                    "sess_aligned": int(row["sess_side"] == armed),
                    "or_aligned": int(row["or_pos"] == armed),
                }
                d_armed = armed
                armed, wait_i = 0, None
                if gate is None or gate(feat):
                    pos = {"dir": d_armed, "entry": entry, "sl": sl, "tp": tp,
                           "time": ts, "feat": feat}
                    # entry candle can also stop us out (assume worst case)
                    hit = l[i] <= sl if d_armed == 1 else h[i] >= sl
                    if hit:
                        pnl = (sl - entry) * d_armed - COST_POINTS
                        trades.append({**feat, "entry_time": ts, "exit_time": ts,
                                       "direction": "CE" if d_armed == 1 else "PE",
                                       "entry": round(entry, 2), "exit": round(sl, 2),
                                       "exit_reason": "SL", "pnl_points": round(pnl, 2),
                                       "win": 0})
                        pos = None
                    continue

        # ── signal engine on the close of bar i ──
        up = fast[i - 1] < slow[i - 1] and fast[i] > slow[i]
        dn = fast[i - 1] > slow[i - 1] and fast[i] < slow[i]
        gap = new_day and o[i] != c[i - 1]
        if (up or dn) and not gap and not armed and pending == 0:
            pending, cross_i = (1 if up else -1), i

        if pending and i - cross_i == ANGLE_CONFIRM_BARS:
            slope = (fast[i] - fast[cross_i]) / ANGLE_CONFIRM_BARS
            ang = float(np.degrees(np.arctan(slope / atr[i]))) if atr[i] else 0.0
            if (pending == 1 and ang >= MIN_ANGLE_DEG) or (pending == -1 and ang <= -MIN_ANGLE_DEG):
                armed, wait_i, angle_at_arm = pending, i, ang
            pending, cross_i = 0, None

        if armed:
            broken = c[i] < slow[i] if armed == 1 else c[i] > slow[i]
            if (i - wait_i) > MAX_WAIT_BARS or broken or is_fc(ts):
                armed, wait_i = 0, None

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def stats(t):
    if t is None or len(t) == 0:
        return {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy": np.nan, "total_points": 0.0, "max_dd": 0.0}
    gw = t.loc[t.pnl_points > 0, "pnl_points"].sum()
    gl = -t.loc[t.pnl_points <= 0, "pnl_points"].sum()
    cum = t["pnl_points"].cumsum()
    return {"trades": len(t), "win_rate": round(t["win"].mean() * 100, 1),
            "profit_factor": round(gw / gl, 2) if gl > 0 else np.inf,
            "expectancy": round(t["pnl_points"].mean(), 2),
            "total_points": round(t["pnl_points"].sum(), 1),
            "max_dd": round((cum - cum.cummax()).min(), 1)}


def split(t, cutoff):
    if len(t) == 0:
        return t, t
    d = pd.to_datetime(t["entry_time"]).dt.date
    return t[d <= cutoff], t[d > cutoff]


# ─────────────────────────────────────────────────────────────────────────────
# FILTER RULES — each returns True when the trade is ALLOWED
# ─────────────────────────────────────────────────────────────────────────────
def lunch_ok(f):       # skip the 11:30–13:15 lull
    return not (11.5 <= f["entry_hour"] < 13.5)


RULES = {
    "ADX >= 20":                         lambda f: f["adx"] >= 20,
    "ADX >= 25":                         lambda f: f["adx"] >= 25,
    "Choppiness < 55":                   lambda f: f["chop"] < 55,
    "Choppiness < 50":                   lambda f: f["chop"] < 50,
    "EMA spread >= 0.25 ATR":            lambda f: f["ema_spread_atr"] >= 0.25,
    "15-min trend agrees":               lambda f: f["htf_aligned"] == 1,
    "Right side of session average":     lambda f: f["sess_aligned"] == 1,
    "Outside opening range, with trade": lambda f: f["or_aligned"] == 1,
    "<= 2 EMA crosses so far today":     lambda f: f["crosses_today"] <= 2,
    "<= 3 EMA crosses so far today":     lambda f: f["crosses_today"] <= 3,
    "Skip 11:30-13:30":                  lunch_ok,
    "No entries after 14:30":            lambda f: f["entry_hour"] < 14.5,
    "Day range used < 100% of average":  lambda f: not (f["range_used"] >= 1.0),
    "Stop >= 0.3 ATR away":              lambda f: f["sl_atr"] >= 0.3,
    ">= 1 ATR room to prior day H/L":    lambda f: not (0 <= f["room_atr"] < 1.0),
    "15-min trend + session average":    lambda f: f["htf_aligned"] == 1 and f["sess_aligned"] == 1,
    "15-min trend + ADX >= 20":          lambda f: f["htf_aligned"] == 1 and f["adx"] >= 20,
    "15-min trend + <= 3 crosses":       lambda f: f["htf_aligned"] == 1 and f["crosses_today"] <= 3,
    "15-min trend + skip lunch":         lambda f: f["htf_aligned"] == 1 and lunch_ok(f),
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def bucket_table(trades, rr):
    rows = []
    for feat in FEATURES:
        s = trades[feat]
        if s.nunique() <= 2:
            groups = s.map({0: "no", 1: "yes"})
        elif feat == "entry_hour":
            groups = pd.cut(s, [9, 10, 11.5, 13.5, 14.5, 16], right=False,
                            labels=["09:15-10:00", "10:00-11:30", "11:30-13:30",
                                    "13:30-14:30", "14:30-15:15"])
        elif feat == "crosses_today":
            groups = pd.cut(s, [-1, 1, 3, 5, 99], labels=["0-1", "2-3", "4-5", "6+"])
        else:
            try:
                groups = pd.qcut(s, 4, duplicates="drop")
            except ValueError:
                continue
        for g, sub in trades.groupby(groups, observed=True):
            st = stats(sub)
            rows.append({"rr": rr, "condition": feat, "bucket": str(g), **st,
                         "reliable": "yes" if st["trades"] >= MIN_TRADES_BUCKET else "TOO FEW"})
    return rows


def main():
    df = add_features(load_data())
    days = sorted(set(df.index.date))
    cutoff = days[int(len(days) * TRAIN_FRACTION) - 1]
    print(f"{len(days)} trading days. Early period: up to {cutoff}. Late period: after.\n")

    bucket_rows, rule_rows = [], []
    for rr in RR_VALUES:
        base = run(df, rr)
        if base.empty:
            print(f"RR {rr}: no trades."); continue
        if rr == RR_VALUES[-1]:
            base.to_csv("filter_trades.csv", index=False)
        bucket_rows += bucket_table(base, rr)

        b_all = stats(base)
        b_early, b_late = (stats(x) for x in split(base, cutoff))
        rule_rows.append({"rr": rr, "rule": "BASELINE (no filter)", **b_all,
                          "kept_pct": 100.0,
                          "early_win": b_early["win_rate"], "early_pf": b_early["profit_factor"],
                          "late_win": b_late["win_rate"], "late_pf": b_late["profit_factor"],
                          "late_trades": b_late["trades"], "verdict": "-"})
        for name, gate in RULES.items():
            t = run(df, rr, gate)
            s = stats(t)
            e, lt = (stats(x) for x in split(t, cutoff))
            better_early = (e["profit_factor"] or 0) > (b_early["profit_factor"] or 0)
            better_late = (lt["profit_factor"] or 0) > (b_late["profit_factor"] or 0)
            if s["trades"] < 20 or lt["trades"] < 8:
                verdict = "too few trades to judge"
            elif better_early and better_late and s["total_points"] >= 0.7 * b_all["total_points"]:
                verdict = "HELPS in both periods"
            elif better_early and better_late:
                verdict = "higher quality, but cuts too much profit"
            elif better_early != better_late:
                verdict = "inconsistent - likely luck"
            else:
                verdict = "hurts"
            rule_rows.append({"rr": rr, "rule": name, **s,
                              "kept_pct": round(s["trades"] / b_all["trades"] * 100, 1),
                              "early_win": e["win_rate"], "early_pf": e["profit_factor"],
                              "late_win": lt["win_rate"], "late_pf": lt["profit_factor"],
                              "late_trades": lt["trades"], "verdict": verdict})

    if not rule_rows:
        print("No trades produced. Check data."); sys.exit(1)

    rules = pd.DataFrame(rule_rows)
    buckets = pd.DataFrame(bucket_rows)
    rules.to_csv("filter_rules_results.csv", index=False)
    buckets.to_csv("filter_feature_buckets.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    show = ["rule", "trades", "kept_pct", "win_rate", "profit_factor", "total_points",
            "max_dd", "late_win", "late_pf", "verdict"]
    for rr in RR_VALUES:
        sub = rules[rules.rr == rr]
        if sub.empty:
            continue
        print("=" * 100)
        print(f"  RISK:REWARD 1:{rr}  — points are Nifty index points, net of {COST_POINTS} cost per trade")
        print("=" * 100)
        print(sub[show].to_string(index=False))
        print()
    print("Saved: filter_rules_results.csv, filter_feature_buckets.csv, filter_trades.csv")
    print("Trust a filter only if its verdict is 'HELPS in both periods'.")


if __name__ == "__main__":
    main()
