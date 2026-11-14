"""
Research Setups — three published intraday ideas, tested on Nifty as written
============================================================================
The EMA retest, the alternative entries and the big-candle scalp were all
tuned on our own data and none survived. This script goes the other way:
it takes rules from published research, keeps their parameters, and runs
them on about three years of 1-minute Nifty data. Nothing is tuned here.

  A. NOISE-BAND MOMENTUM  (Zarattini, Aziz, Barbon 2024, "Beat the Market")
     For every minute of the day, measure the average absolute move from the
     open at that minute over the last 14 days. That defines a "noise area"
     around today's open (widened by the overnight gap). Every half hour,
     if price is above the area go long, below it go short.
       A1: hold until price crosses the opposite edge (then reverse) or 15:15.
       A2: also exit when price falls back inside the area or through the
           day's average price (the paper uses VWAP; the index has no
           volume, so an equal-weighted average stands in).

  B. OPENING-RANGE BREAKOUT, 30 minutes  (intradaylab.com spec, 2017-2026)
     Range = 09:15-09:45 high/low. First break of either side, one trade a
     day. Stop = other side of the range, target = 2x the range, time exit
     14:30.  B1: skip days with a range under 40 points.  B2: no skip.

  C. FIRST HALF-HOUR -> LAST HALF-HOUR  (Gao, Han, Li, Zhou 2018)
     Direction of the move from yesterday's close to 09:45 is traded from
     15:00 to 15:25.  C1: every day.  C2: only if 14:30-15:00 agrees.

Usage (on the VM, after market hours):
  cd ~/nifty-bot && git pull
  set -a && source ~/.nifty_bot_env && set +a
  python3 backtest_research_setups.py

It reuses nifty_1min_older_cache.csv and nifty_1min_cache.csv if present
(otherwise it downloads them).

Outputs:
  research_results.csv   one row per setup
  research_yearly.csv    rupees per lot by year
  research_trades.csv    every trade of every setup

How rupees are estimated (simplifications):
  option P&L = delta x Nifty move - THETA_PTS_PER_HOUR x hours held
               - SLIPPAGE_PTS, x lot size, minus costs.py charges.
  Real options also gain from gamma on big moves and lose more to decay
  near expiry. The "nifty_points" columns show the raw edge with no
  option model at all, and the FUTURES rows show the same trades taken
  with one lot of Nifty futures (no time decay, but about 8 points of
  cost per trade and far more margin than Rs 10k).
"""

import os
import sys

import numpy as np
import pandas as pd

import config
import costs
from backtest_candle_scalp import load_1min, load_older, CACHE_OLDER

DELTA              = config.DEFAULT_DELTA
PREMIUM            = config.TARGET_PREMIUM
LOT                = config.LOT_SIZE
THETA_PTS_PER_HOUR = 2.5     # assumed time decay of a ~Rs 95 weekly option while held
SLIPPAGE_PTS       = 0.5     # assumed premium slippage per round trip
OLDER_MONTHS       = 24
NOISE_LOOKBACK     = 14      # days, from the paper
ORB_MIN_RANGE      = 40      # points, from the source
EOD                = 15 * 60 + 15
# Nifty futures, one lot, round trip: brokerage ~Rs 40, STT 0.02% of the sell
# value (~Rs 310), exchange + SEBI + stamp + GST (~Rs 105), 1 point slippage.
# About Rs 520, i.e. 8 Nifty points. Futures need roughly Rs 1.5-2 lakh margin
# per lot, so this column is for comparison, not something Rs 10k can trade.
FUT_COST_POINTS    = 8.0


def rupees(points, minutes):
    exit_prem = max(PREMIUM + points * DELTA - THETA_PTS_PER_HOUR * minutes / 60 - SLIPPAGE_PTS, 0.05)
    return (exit_prem - PREMIUM) * LOT - costs.round_trip_charges(PREMIUM, exit_prem)


def trade(setup, t_in, t_out, d, p_in, p_out, reason):
    pts = (p_out - p_in) * d
    mins = max((t_out - t_in).total_seconds() / 60, 0)
    return {"setup": setup, "entry_time": t_in, "exit_time": t_out,
            "option": "CE" if d == 1 else "PE", "nifty_entry": round(p_in, 2),
            "nifty_exit": round(p_out, 2), "exit_reason": reason,
            "nifty_points": round(pts, 2), "minutes": round(mins),
            "pnl_rupees": round(rupees(pts, mins), 2),
            "pnl_futures": round((pts - FUT_COST_POINTS) * LOT, 2)}


def day_frames(m1):
    """Yield (date, frame, minutes-since-midnight array) for each full trading day."""
    for d, g in m1.groupby(m1.index.date):
        if len(g) >= 300:
            yield d, g, (g.index.hour * 60 + g.index.minute).values


# ── A. Noise-band momentum ──────────────────────────────────────────────────
def setup_noise(m1, use_avg_stop):
    name = "A2 noise band + day-average stop" if use_avg_stop else "A1 noise band, hold to opposite edge"
    days = list(day_frames(m1))
    # |move from open| at each minute of each day
    moves = {}
    for d, g, mins in days:
        moves[d] = pd.Series(np.abs(g["close"].values / g["open"].values[0] - 1), index=mins)
    out, prev_close = [], None
    for k, (d, g, mins) in enumerate(days):
        if k >= NOISE_LOOKBACK and prev_close is not None:
            hist = pd.concat([moves[days[j][0]] for j in range(k - NOISE_LOOKBACK, k)], axis=1)
            sigma = hist.mean(axis=1).reindex(mins).ffill().values
            o, c = g["open"].values, g["close"].values
            day_open = o[0]
            ub = max(day_open, prev_close) * (1 + sigma)
            lb = min(day_open, prev_close) * (1 - sigma)
            avg = np.cumsum((g["high"].values + g["low"].values + c) / 3) / np.arange(1, len(c) + 1)
            pos, t_in, p_in = 0, None, None
            for i in range(len(c) - 1):
                m_close = mins[i] + 1                      # this bar closes at this minute
                if m_close >= EOD:
                    break
                if m_close < 9 * 60 + 45 or m_close % 30 != 15:
                    continue                               # decisions only at :15 and :45
                if np.isnan(sigma[i]):
                    continue
                if use_avg_stop:
                    want = 1 if c[i] > max(ub[i], avg[i]) else (-1 if c[i] < min(lb[i], avg[i]) else 0)
                    if pos == 1 and c[i] > max(ub[i], avg[i]):
                        want = 1
                    if pos == -1 and c[i] < min(lb[i], avg[i]):
                        want = -1
                else:
                    want = 1 if c[i] > ub[i] else (-1 if c[i] < lb[i] else pos)
                if want != pos:
                    t_now, p_now = g.index[i + 1], o[i + 1]
                    if pos != 0:
                        out.append(trade(name, t_in, t_now, pos, p_in, p_now,
                                         "Reverse" if want == -pos else "Back inside"))
                    pos, t_in, p_in = want, t_now, p_now
            if pos != 0:
                j = int(np.searchsorted(mins, EOD))
                j = min(j, len(c) - 1)
                out.append(trade(name, t_in, g.index[j], pos, p_in, o[j], "EOD"))
        prev_close = g["close"].values[-1]
    return out


# ── B. Opening-range breakout ───────────────────────────────────────────────
def setup_orb(m1, min_range):
    name = f"B1 ORB 30-min, range >= {min_range}" if min_range else "B2 ORB 30-min, any range"
    out = []
    for d, g, mins in day_frames(m1):
        o, h, l = g["open"].values, g["high"].values, g["low"].values
        in_or = mins < 9 * 60 + 45
        if in_or.sum() < 25:
            continue
        hi, lo = h[in_or].max(), l[in_or].min()
        rng = hi - lo
        if rng < min_range or rng <= 0:
            continue
        pos = 0
        for i in range(int(in_or.sum()), len(o)):
            if mins[i] >= 14 * 60 + 30:
                if pos != 0:
                    out.append(trade(name, t_in, g.index[i], pos, p_in, o[i], "Time exit"))
                break
            if pos == 0:
                up, dn = h[i] > hi, l[i] < lo
                if up and dn:
                    break                                   # both sides in one minute: skip the day
                if up or dn:
                    pos = 1 if up else -1
                    p_in = max(hi, o[i]) if up else min(lo, o[i])
                    t_in = g.index[i]
                    sl = lo if up else hi
                    tp = p_in + pos * 2 * rng
                    # rest of the entry minute: only the stop can be judged safely
                    if (l[i] <= sl) if pos == 1 else (h[i] >= sl):
                        out.append(trade(name, t_in, g.index[i], pos, p_in, sl, "SL"))
                        break
                continue
            hit_sl = l[i] <= sl if pos == 1 else h[i] >= sl
            hit_tp = h[i] >= tp if pos == 1 else l[i] <= tp
            if hit_sl or hit_tp:
                out.append(trade(name, t_in, g.index[i], pos, p_in, sl if hit_sl else tp,
                                 "SL" if hit_sl else "Target"))
                break
    return out


# ── C. First half-hour predicts last half-hour ──────────────────────────────
def setup_first_last(m1, need_r12):
    name = "C2 first->last half-hour, 14:30-15:00 agrees" if need_r12 else "C1 first->last half-hour"
    out, prev_close = [], None
    for d, g, mins in day_frames(m1):
        o, c = g["open"].values, g["close"].values
        def at(minute):                                     # index of the bar that OPENS at `minute`
            j = int(np.searchsorted(mins, minute))
            return j if j < len(mins) and mins[j] == minute else None
        i945, i1430, i1500, i1525 = at(9 * 60 + 45), at(14 * 60 + 30), at(15 * 60), at(15 * 60 + 25)
        if prev_close is not None and None not in (i945, i1430, i1500, i1525):
            r1 = o[i945] / prev_close - 1
            r12 = o[i1500] / o[i1430] - 1
            d1 = int(np.sign(r1))
            if d1 != 0 and (not need_r12 or np.sign(r12) == d1):
                out.append(trade(name, g.index[i1500], g.index[i1525], d1, o[i1500], o[i1525], "Time exit"))
        prev_close = c[-1]
    return out


# ── reporting ───────────────────────────────────────────────────────────────
def summarise(t, n_months, mid):
    w, lo = t[t.pnl_rupees > 0], t[t.pnl_rupees <= 0]
    gl = -lo.pnl_rupees.sum()
    cum = t.pnl_rupees.cumsum()
    mo = t.groupby(t.entry_time.dt.strftime("%Y-%m")).pnl_rupees.sum()
    first, second = t[t.entry_time.dt.date <= mid].pnl_rupees.sum(), t[t.entry_time.dt.date > mid].pnl_rupees.sum()
    return {"trades": len(t), "trades_per_month": round(len(t) / n_months, 1),
            "win_rate": round(len(w) / len(t) * 100, 1),
            "profit_factor": round(w.pnl_rupees.sum() / gl, 2) if gl > 0 else np.inf,
            "nifty_points_total": round(t.nifty_points.sum(), 0),
            "nifty_points_per_trade": round(t.nifty_points.mean(), 2),
            "net_rupees_per_lot": round(t.pnl_rupees.sum(), 0),
            "rupees_per_month": round(t.pnl_rupees.sum() / n_months, 0),
            "avg_minutes_held": round(t.minutes.mean()),
            "losing_months": f"{int((mo < 0).sum())} of {len(mo)}",
            "worst_month": round(mo.min(), 0),
            "max_drawdown": round((cum - cum.cummax()).min(), 0),
            "first_half": round(first, 0), "second_half": round(second, 0),
            "passes": "yes" if first > 0 and second > 0 else "no",
            "FUTURES_net_rupees": round(t.pnl_futures.sum(), 0),
            "FUTURES_per_month": round(t.pnl_futures.sum() / n_months, 0),
            "FUTURES_first_half": round(t[t.entry_time.dt.date <= mid].pnl_futures.sum(), 0),
            "FUTURES_second_half": round(t[t.entry_time.dt.date > mid].pnl_futures.sum(), 0),
            "FUTURES_max_drawdown": round((t.pnl_futures.cumsum() - t.pnl_futures.cumsum().cummax()).min(), 0)}


def main():
    frames = []
    if os.path.exists(CACHE_OLDER) or "--no-older" not in sys.argv:
        frames.append(load_older(OLDER_MONTHS))
    frames.append(load_1min())
    m1 = pd.concat(frames)
    m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
    days = sorted(set(m1.index.date))
    n_months = len(days) / 21.0
    mid = days[len(days) // 2 - 1]
    print(f"\n{days[0]} to {days[-1]}: {len(days)} trading days (~{n_months:.0f} months). "
          f"First half ends {mid}.\n")

    all_trades = (setup_noise(m1, False) + setup_noise(m1, True)
                  + setup_orb(m1, ORB_MIN_RANGE) + setup_orb(m1, 0)
                  + setup_first_last(m1, False) + setup_first_last(m1, True))
    if not all_trades:
        print("No trades. Check data."); sys.exit(1)
    T = pd.DataFrame(all_trades)
    T["entry_time"] = pd.to_datetime(T["entry_time"])
    T.to_csv("research_trades.csv", index=False)

    rows = [{"setup": name, **summarise(t.sort_values("entry_time"), n_months, mid)}
            for name, t in T.groupby("setup")]
    res = pd.DataFrame(rows)
    res.to_csv("research_results.csv", index=False)
    yearly = T.pivot_table(index=T.entry_time.dt.year, columns="setup", values="pnl_rupees",
                           aggfunc="sum").round(0)
    yearly.index.name = "year"
    yearly.to_csv("research_yearly.csv")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print("=" * 120)
    print("  PUBLISHED SETUPS ON NIFTY — net of costs. Nothing here was tuned.")
    print("  Upper rows: buying a ~Rs 95 option. FUTURES rows: same trades with one futures lot.")
    print("=" * 120)
    print(res.T.to_string(header=False))
    print("\n  BY YEAR, option buying (rupees per lot)")
    print(yearly.T.to_string())
    print("\n'passes' = profitable in both halves of the data.")
    print("Saved: research_results.csv, research_yearly.csv, research_trades.csv")


if __name__ == "__main__":
    main()
