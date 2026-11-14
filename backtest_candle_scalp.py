"""
Big-Candle Reversal Scalp — backtest
====================================
The strategy as confirmed:

  Signal candle (Nifty 5-min):
    * high - low >= 40 points
    * body (|close - open|) >= 75% of that range
    * RED candle with the longer wick at the BOTTOM  -> buy a CE (bet on a bounce)
    * GREEN candle with the longer wick at the TOP   -> buy a PE (bet on a pullback)
  Entry : at the signal candle's close (the next candle's open)
  Target: 50% of the signal candle's range, in Nifty points, from entry
  Stop  : 25% of the signal candle's range, in Nifty points, from entry

Stops and targets this tight are often both inside one 5-minute candle, so
exits are simulated on 1-MINUTE candles. Within a single minute the stop is
assumed to be hit before the target (worst case).

It then tests a fixed set of nearby settings (candle size, body %, target,
stop) and the opposite idea (trade WITH the candle) for comparison.

Usage (on the VM, after market hours):
  cd ~/nifty-bot
  set -a && source ~/.nifty_bot_env && set +a
  python3 backtest_candle_scalp.py

Out-of-sample check on older data (frozen settings only, no tuning):
  python3 backtest_candle_scalp.py --older        # 24 months before April 2026
  python3 backtest_candle_scalp.py --older 36     # or as far back as you like

Outputs:
  candle_scalp_trades.csv    every trade of YOUR rules
  candle_scalp_results.csv   your rules and every variant
  candle_scalp_monthly.csv   month by month, your rules and the best variant

Simplifications:
  * Rupees = delta x Nifty move x lot size, minus costs.py charges and
    SLIPPAGE_PTS. Entry premium fixed at config.TARGET_PREMIUM.
  * One position at a time. Open trades are closed at 15:15.
    No daily loss limit, so variants can be compared like for like.
"""

import argparse
import itertools
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

import config
import costs
from backtest import _headers, _parse_candles_1min, _last_trading_day, BASE_V2, ENCODED_KEY

DELTA          = config.DEFAULT_DELTA
PREMIUM        = config.TARGET_PREMIUM
LOT            = config.LOT_SIZE
SLIPPAGE_PTS   = 0.5          # assumed premium slippage per round trip (per unit)
LOOKBACK_MONTHS = 7
CACHE_1MIN     = "nifty_1min_cache.csv"
CACHE_OLDER    = "nifty_1min_older_cache.csv"
TUNED_FROM     = date(2026, 4, 9)   # first day of the data the variants were tuned on
TRAIN_FRACTION = 0.67
FIRST_ENTRY    = (9, 20)      # no entries before the first candle has closed
LAST_ENTRY     = (15, 0)
FORCE_CLOSE    = (15, 15)

# Your rules
YOURS = {"min_range": 40, "body_pct": 0.75, "target": 0.50, "stop": 0.25,
         "wick_rule": True, "mode": "reversal"}

# The only versions run on older data (--older). Chosen from the tuned period and
# then frozen: the three that made money in both halves, the suggested middle
# setting, and your original rules for reference.
FROZEN = {
    "your original rules":            YOURS,
    "candidate (60pt, 75% body, T75, S50)":  {"min_range": 60, "body_pct": 0.75, "target": 0.75,
                                              "stop": 0.50, "wick_rule": False, "mode": "reversal"},
    "60pt, 85% body, T75, S50":       {"min_range": 60, "body_pct": 0.85, "target": 0.75,
                                       "stop": 0.50, "wick_rule": False, "mode": "reversal"},
    "60pt, 75% body, T100, S50":      {"min_range": 60, "body_pct": 0.75, "target": 1.00,
                                       "stop": 0.50, "wick_rule": False, "mode": "reversal"},
}

# Variants (fixed in advance)
GRID = {
    "min_range": [30, 40, 50, 60],
    "body_pct":  [0.60, 0.75, 0.85],
    "target":    [0.50, 0.75, 1.00],
    "stop":      [0.25, 0.50],
    "wick_rule": [True, False],
    "mode":      ["reversal", "continuation"],
}


# ── data ────────────────────────────────────────────────────────────────────
def _read_cache(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    print(f"Loaded {len(df)} cached 1-min candles from {path} "
          f"({df.index[0].date()} to {df.index[-1].date()}; delete the file to re-download)")
    return df


def download_1min(end, months, path):
    """Download `months` of 1-min candles ending at `end`, newest first, and cache them."""
    print(f"Downloading {months} months of 1-min Nifty candles ending {end} from Upstox...")
    frames, empty_run, n_chunks = [], 0, months * 30 // 7 + 1
    for n in range(n_chunks):
        start = end - timedelta(days=6)
        url = f"{BASE_V2}/historical-candle/{ENCODED_KEY}/1minute/{end:%Y-%m-%d}/{start:%Y-%m-%d}"
        got = False
        try:
            r = requests.get(url, headers=_headers(), timeout=30)
            if r.status_code == 200:
                chunk = _parse_candles_1min(r.json())
                if not chunk.empty:
                    frames.append(chunk)
                    got = True
            else:
                print(f"  {start} to {end}: HTTP {r.status_code} {r.text[:80]}")
        except Exception as e:
            print(f"  {start} to {end}: {e}")
        empty_run = 0 if got else empty_run + 1
        if empty_run >= 6:
            print(f"  No data for 6 weeks in a row before {end} - Upstox has nothing older. Stopping.")
            break
        if (n + 1) % 10 == 0:
            print(f"  ...{n + 1}/{n_chunks} weeks")
        end = start - timedelta(days=1)
        time.sleep(0.3)
    if not frames:
        raise RuntimeError("No data downloaded. Check the token, or Upstox has no data that far back.")
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index().between_time("09:15", "15:29")
    df.to_csv(path)
    print(f"Saved {len(df)} candles ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def load_1min():
    if os.path.exists(CACHE_1MIN):
        return _read_cache(CACHE_1MIN)
    return download_1min(_last_trading_day(), LOOKBACK_MONTHS, CACHE_1MIN)


def load_older(months):
    """Data from BEFORE the period the settings were tuned on."""
    if os.path.exists(CACHE_OLDER):
        return _read_cache(CACHE_OLDER)
    return download_1min(TUNED_FROM - timedelta(days=1), months, CACHE_OLDER)


def to_5min(m1):
    return m1.resample("5min").agg(open=("open", "first"), high=("high", "max"),
                                   low=("low", "min"), close=("close", "last")).dropna()


# ── signals ─────────────────────────────────────────────────────────────────
def find_signals(m5, min_range, body_pct, wick_rule, mode):
    o, h, l, c = (m5[k].values for k in ["open", "high", "low", "close"])
    rng = h - l
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    red, green = c < o, c > o
    ok = (rng >= min_range) & (body >= body_pct * rng)
    if wick_rule:
        ok &= (red & (lower > upper)) | (green & (upper > lower))
    else:
        ok &= red | green
    sig = []
    for i in np.where(ok)[0]:
        ts = m5.index[i] + pd.Timedelta(minutes=5)      # entry time = candle close
        hm = (ts.hour, ts.minute)
        if hm < FIRST_ENTRY or hm > LAST_ENTRY:
            continue
        d = 1 if red[i] else -1                         # reversal: against the colour
        if mode == "continuation":
            d = -d
        sig.append((ts, d, rng[i]))
    return sig


# ── simulation on 1-minute candles ──────────────────────────────────────────
def simulate(m1, pos_of, signals, target, stop):
    o, h, l = m1["open"].values, m1["high"].values, m1["low"].values
    idx = m1.index
    hm = idx.hour * 100 + idx.minute
    dates = idx.date
    fc = FORCE_CLOSE[0] * 100 + FORCE_CLOSE[1]
    trades, busy_until = [], -1
    for ts, d, rng in signals:
        j = pos_of.get(ts)
        if j is None or j <= busy_until:
            continue                                    # no candle at entry / still in a trade
        entry = o[j]
        tp, sl = entry + d * target * rng, entry - d * stop * rng
        k, exit_px, reason = j, None, None
        while k < len(o):
            if dates[k] != dates[j] or hm[k] >= fc:
                exit_px, reason = (o[k] if dates[k] == dates[j] else o[k - 1]), "EOD"
                break
            hit_sl = l[k] <= sl if d == 1 else h[k] >= sl
            hit_tp = h[k] >= tp if d == 1 else l[k] <= tp
            if hit_sl:
                exit_px, reason = sl, "SL"
                break
            if hit_tp:
                exit_px, reason = tp, "Target"
                break
            k += 1
        if reason is None:
            exit_px, reason, k = o[-1], "EOD", len(o) - 1
        pts = (exit_px - entry) * d
        exit_prem = max(PREMIUM + pts * DELTA - SLIPPAGE_PTS, 0.05)
        pnl = (exit_prem - PREMIUM) * LOT - costs.round_trip_charges(PREMIUM, exit_prem)
        trades.append({"entry_time": ts, "exit_time": idx[min(k, len(idx) - 1)],
                       "option": "CE" if d == 1 else "PE", "signal_range": round(rng, 1),
                       "nifty_entry": round(entry, 2), "nifty_exit": round(exit_px, 2),
                       "exit_reason": reason, "nifty_points": round(pts, 2),
                       "minutes": k - j, "pnl_rupees": round(pnl, 2)})
        busy_until = k
    return pd.DataFrame(trades)


def monthly(t):
    return t.groupby(pd.to_datetime(t["entry_time"]).dt.strftime("%Y-%m"))["pnl_rupees"].sum().round(0)


def stats(t, cutoff):
    if len(t) == 0:
        return None
    w, lo = t[t.pnl_rupees > 0], t[t.pnl_rupees <= 0]
    gl = -lo.pnl_rupees.sum()
    cum = t.pnl_rupees.cumsum()
    d = pd.to_datetime(t["entry_time"]).dt.date
    mo = monthly(t)
    return {"trades": len(t), "win_rate": round(len(w) / len(t) * 100, 1),
            "profit_factor": round(w.pnl_rupees.sum() / gl, 2) if gl > 0 else np.inf,
            "net_rupees_per_lot": round(t.pnl_rupees.sum(), 0),
            "avg_per_trade": round(t.pnl_rupees.mean(), 0),
            "nifty_points_total": round(t.nifty_points.sum(), 0),
            "losing_months": int((mo < 0).sum()), "months": len(mo),
            "max_drawdown": round((cum - cum.cummax()).min(), 0),
            "early_period": round(t[d <= cutoff].pnl_rupees.sum(), 0),
            "late_period": round(t[d > cutoff].pnl_rupees.sum(), 0)}


def run_older(months):
    """Out-of-sample check: frozen settings on data the tuning never saw."""
    m1 = load_older(months)
    m5 = to_5min(m1)
    pos_of = {ts: i for i, ts in enumerate(m1.index)}
    days = sorted(set(m1.index.date))
    cutoff = days[len(days) // 2 - 1]
    print(f"\nOLDER DATA: {days[0]} to {days[-1]} ({len(days)} trading days). "
          f"None of this was used to choose the settings.\n")
    rows, halves = [], {}
    for name, p in FROZEN.items():
        sig = find_signals(m5, p["min_range"], p["body_pct"], p["wick_rule"], p["mode"])
        t = simulate(m1, pos_of, sig, p["target"], p["stop"])
        s = stats(t, cutoff)
        if s is None:
            rows.append({"version": name, "trades": 0})
            continue
        s["first_half"], s["second_half"] = s.pop("early_period"), s.pop("late_period")
        rows.append({"version": name, **s})
        d = pd.to_datetime(t["entry_time"])
        halves[name] = t.groupby(d.dt.year.astype(str) + "-H" + ((d.dt.month - 1) // 6 + 1).astype(str)
                                 )["pnl_rupees"].sum().round(0)
        if name.startswith("candidate"):
            t.to_csv("candle_scalp_older_trades.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv("candle_scalp_older_results.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print("=" * 120)
    print("  FROZEN SETTINGS ON OLDER DATA (rupees per lot, net of costs)")
    print("=" * 120)
    print(res.to_string(index=False))
    if halves:
        hy = pd.DataFrame(halves).fillna(0)
        hy.to_csv("candle_scalp_older_halfyears.csv")
        print("\n  BY HALF-YEAR (rupees per lot)")
        print(hy.to_string())
    print("\nA version passes only if it is profitable overall AND in both halves here.")
    print("Saved: candle_scalp_older_results.csv, candle_scalp_older_halfyears.csv, "
          "candle_scalp_older_trades.csv (candidate version)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--older", nargs="?", const=24, type=int, metavar="MONTHS",
                    help="test the frozen settings on data before the tuned period (default 24 months)")
    args = ap.parse_args()
    if args.older:
        run_older(args.older)
        return
    m1 = load_1min()
    m5 = to_5min(m1)
    pos_of = {ts: i for i, ts in enumerate(m1.index)}
    days = sorted(set(m1.index.date))
    cutoff = days[int(len(days) * TRAIN_FRACTION) - 1]
    print(f"{len(days)} trading days, {len(m5)} five-minute candles. "
          f"Early period ends {cutoff}.\n")

    keys = list(GRID)
    rows, kept, cache = [], {}, {}
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(keys, combo))
        skey = (p["min_range"], p["body_pct"], p["wick_rule"], p["mode"])
        if skey not in cache:
            cache[skey] = find_signals(m5, *skey)
        t = simulate(m1, pos_of, cache[skey], p["target"], p["stop"])
        s = stats(t, cutoff)
        if s is None:
            continue
        is_yours = p == YOURS
        kept[combo] = t
        rows.append({**p, "is_your_rules": is_yours, **s,
                     "holds_up": "yes" if s["early_period"] > 0 and s["late_period"] > 0 else "no",
                     "_combo": combo})
    if not rows:
        print("No trades at all. Check data."); sys.exit(1)

    res = pd.DataFrame(rows).sort_values("net_rupees_per_lot", ascending=False)
    res.drop(columns="_combo").to_csv("candle_scalp_results.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cols = ["mode", "min_range", "body_pct", "wick_rule", "target", "stop", "trades", "win_rate",
            "profit_factor", "net_rupees_per_lot", "avg_per_trade", "losing_months",
            "max_drawdown", "early_period", "late_period", "holds_up"]
    yours = res[res.is_your_rules]
    print("=" * 120)
    print("  YOUR RULES (rupees per lot, net of costs)")
    print("=" * 120)
    if len(yours):
        print(yours[cols].to_string(index=False))
        ty = kept[yours.iloc[0]["_combo"]]
        ty.to_csv("candle_scalp_trades.csv", index=False)
        print("\n  Exits:", ty.exit_reason.value_counts().to_dict(),
              "| median minutes in trade:", int(ty.minutes.median()))
        print("  By option:", ty.groupby("option").pnl_rupees.agg(["count", "sum"]).round(0).to_dict("index"))
    else:
        print("  Your rules produced no trades in this data.")

    enough = res[res.trades >= 30]
    print("\n" + "=" * 120)
    print("  TOP 12 VARIANTS with at least 30 trades — only trust holds_up = yes")
    print("=" * 120)
    print(enough[cols].head(12).to_string(index=False))
    good = enough[(enough.net_rupees_per_lot > 0) & (enough.holds_up == "yes")]
    print(f"\n  {len(good)} of {len(enough)} variants (30+ trades) made money in both periods.")
    for mode in GRID["mode"]:
        sub = enough[enough["mode"] == mode]
        if len(sub):
            print(f"  {mode}: {int((sub.net_rupees_per_lot > 0).sum())} of {len(sub)} profitable, "
                  f"median net ₹{sub.net_rupees_per_lot.median():,.0f}")

    out = {}
    if len(yours):
        out["your_rules"] = monthly(kept[yours.iloc[0]["_combo"]])
    if len(enough):
        out["best_variant"] = monthly(kept[enough.iloc[0]["_combo"]])
    if out:
        mt = pd.DataFrame(out).fillna(0)
        mt.to_csv("candle_scalp_monthly.csv")
        print("\n  MONTH BY MONTH (rupees per lot)")
        print(mt.to_string())
    print("\nSaved: candle_scalp_results.csv, candle_scalp_trades.csv, candle_scalp_monthly.csv")


if __name__ == "__main__":
    main()
