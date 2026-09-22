"""
Nifty 50 Options Strategy Backtester
=====================================
Systematically tests all combinations of key strategy parameters
over 6 months of historical Nifty 5-min data fetched from Upstox.

Usage:
  cd ~/nifty-bot
  source ~/.nifty_bot_env
  python3 backtest.py

Outputs:
  backtest_all_results.csv   — every combination with full metrics
  backtest_best_configs.csv  — top 20 configurations ranked by profit factor
  Console summary of best config

Runtime: ~5-10 minutes for all combinations on the VM.
"""

import os
import sys
import time
import itertools
import warnings
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
IST = ZoneInfo("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GRID — edit here to add/remove values to test
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "min_angle_deg":      [5, 7, 10, 15, 20],
    "angle_confirm_bars": [1, 2, 3],
    "risk_reward_ratio":  [1.0, 1.5, 2.0, 2.5, 3.0],
    "max_wait_bars":      [8, 12, 15, 20],
    "retest_buffer_pct":  [0.0, 0.05],    # % of fast EMA
}

# Fixed parameters (not being optimised)
FAST_LEN          = 9
SLOW_LEN          = 15
ATR_LEN           = 14
USE_WICK          = True
DELTA             = 0.45    # approximate ATM option delta for P&L scaling
FORCE_CLOSE_HOUR  = 15
FORCE_CLOSE_MIN   = 15
MAX_CONSEC_LOSSES = 2       # pause after N consecutive losses (mirrors live bot)

LOOKBACK_MONTHS   = 6       # how many months of history to fetch
ENCODED_KEY       = "NSE_INDEX%7CNifty%2050"
BASE_V2           = "https://api.upstox.com/v2"


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError("UPSTOX_ANALYTICS_TOKEN not set. Run: source ~/.nifty_bot_env")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _parse_candles_1min(response_json):
    candles = response_json.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(candles,
                      columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def _resample_to_5min(df_1min):
    if df_1min.empty:
        return df_1min
    df_5min = df_1min.resample("5min").agg(
        open=("open",  "first"),
        high=("high",  "max"),
        low= ("low",   "min"),
        close=("close","last"),
    ).dropna()
    return df_5min.between_time("09:15", "15:30")


def _last_trading_day():
    d = date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def fetch_historical_data(months=6):
    """
    Fetch N months of 1-min Nifty candles from Upstox and resample to 5-min.
    Uses 7-day chunks per API call — confirmed working at any depth up to 120+ days.
    """
    print(f"Fetching {months} months of 5-min Nifty history from Upstox...")

    last_trade  = _last_trading_day()
    total_days  = months * 30
    num_chunks  = (total_days // 7) + 1
    all_frames  = []

    chunk_end = last_trade
    for chunk in range(num_chunks):
        chunk_start = chunk_end - timedelta(days=7)
        # Skip if chunk is entirely in the future or too old
        if chunk_start > last_trade:
            chunk_end = chunk_start - timedelta(days=1)
            continue

        url = (f"{BASE_V2}/historical-candle/{ENCODED_KEY}/1minute"
               f"/{chunk_end.strftime('%Y-%m-%d')}"
               f"/{chunk_start.strftime('%Y-%m-%d')}")
        try:
            resp = requests.get(url, headers=_headers(), timeout=30)
            if resp.status_code == 200:
                df_chunk = _parse_candles_1min(resp.json())
                if not df_chunk.empty:
                    all_frames.append(df_chunk)
                    if (chunk + 1) % 4 == 0:  # print every ~4 weeks
                        print(f"  Week {chunk+1}/{num_chunks}: "
                              f"{chunk_start} → {chunk_end} "
                              f"({len(df_chunk)} bars)")
            else:
                print(f"  Week {chunk+1}/{num_chunks}: HTTP {resp.status_code} — "
                      f"{resp.text[:80]}")
        except Exception as e:
            print(f"  Week {chunk+1}/{num_chunks}: Error — {e}")

        chunk_end = chunk_start - timedelta(days=1)
        while chunk_end.weekday() >= 5:
            chunk_end -= timedelta(days=1)
        time.sleep(0.3)  # gentle rate limiting

    if not all_frames:
        raise RuntimeError("No historical data fetched. Check token and connectivity.")

    df_1min = pd.concat(all_frames)
    df_1min = df_1min[~df_1min.index.duplicated(keep="last")].sort_index()
    df_5min = _resample_to_5min(df_1min)
    print(f"\n✅ Total 5-min candles: {len(df_5min)} "
          f"({df_5min.index[0].date()} to {df_5min.index[-1].date()})\n")
    return df_5min


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df, fast_len=9, slow_len=15, atr_len=14):
    df = df.copy()
    df["fast_ema"] = df["close"].ewm(span=fast_len, adjust=False).mean()
    df["slow_ema"] = df["close"].ewm(span=slow_len, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_len, adjust=False).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE BACKTEST RUN
# ─────────────────────────────────────────────────────────────────────────────

def is_gap_bar(df, i):
    if i == 0:
        return False
    if df.index[i].date() != df.index[i - 1].date():
        return df["open"].iloc[i] != df["close"].iloc[i - 1]
    return False


def is_force_close(ts):
    return ts.hour > FORCE_CLOSE_HOUR or (
        ts.hour == FORCE_CLOSE_HOUR and ts.minute >= FORCE_CLOSE_MIN
    )


def run_backtest(df, params):
    """
    Run a full backtest on df using the given parameter dict.
    Returns a dict of performance metrics.
    Mirrors the live bot's logic exactly (same state machine).
    """
    min_angle   = params["min_angle_deg"]
    conf_bars   = params["angle_confirm_bars"]
    rr          = params["risk_reward_ratio"]
    max_wait    = params["max_wait_bars"]
    buf_pct     = params["retest_buffer_pct"]

    fast = df["fast_ema"]
    slow = df["slow_ema"]
    atr  = df["atr"]

    # State
    pending_dir      = 0
    pending_cross_i  = None
    armed            = False
    armed_dir        = 0
    wait_start_i     = None

    # Position
    in_trade      = False
    trade_dir     = 0
    entry_price   = 0.0
    sl_price      = 0.0
    target_price  = 0.0

    # Results
    trades = []
    consec_losses = 0
    paused_date   = None
    last_day      = None

    n = len(df)

    for i in range(max(conf_bars + 2, 25), n):
        ts    = df.index[i]
        day   = ts.date()

        # Daily reset
        if day != last_day:
            pending_dir     = 0
            pending_cross_i = None
            armed           = False
            armed_dir       = 0
            wait_start_i    = None
            last_day        = day
            # Reset consecutive-loss pause on new day
            if paused_date is not None and day > paused_date:
                paused_date   = None
                consec_losses = 0

        # Skip if paused
        if paused_date is not None and day <= paused_date:
            continue

        # ── If in trade: check SL / target / force close ──────────────
        if in_trade:
            hi = df["high"].iloc[i]
            lo = df["low"].iloc[i]
            closed = False
            result = None

            force = is_force_close(ts)
            if force:
                exit_px = df["close"].iloc[i]
                result  = "EOD"
                closed  = True
            elif trade_dir == 1:   # long (CE)
                if lo <= sl_price:
                    exit_px = sl_price; result = "SL"; closed = True
                elif hi >= target_price:
                    exit_px = target_price; result = "Target"; closed = True
            else:                  # short (PE)
                if hi >= sl_price:
                    exit_px = sl_price; result = "SL"; closed = True
                elif lo <= target_price:
                    exit_px = target_price; result = "Target"; closed = True

            if closed:
                # P&L on underlying (DELTA-scaled to approximate option premium move)
                raw_pnl = (exit_px - entry_price) * trade_dir
                opt_pnl = raw_pnl * DELTA
                trades.append({
                    "entry_time":  df.index[i - 1],   # approximate — retest bar
                    "exit_time":   ts,
                    "direction":   trade_dir,
                    "entry_price": entry_price,
                    "exit_price":  exit_px,
                    "sl_price":    sl_price,
                    "target_price":target_price,
                    "exit_reason": result,
                    "underlying_pnl": round(raw_pnl, 2),
                    "option_pnl_est": round(opt_pnl, 2),
                    "win":         1 if opt_pnl > 0 else 0,
                })
                in_trade = False
                if opt_pnl <= 0:
                    consec_losses += 1
                    if consec_losses >= MAX_CONSEC_LOSSES:
                        paused_date = day
                else:
                    consec_losses = 0
            continue  # don't look for new signals while in trade

        # ── Signal engine ──────────────────────────────────────────────

        # Step 1: crossover
        cross_up   = fast.iloc[i-1] < slow.iloc[i-1] and fast.iloc[i] > slow.iloc[i]
        cross_down = fast.iloc[i-1] > slow.iloc[i-1] and fast.iloc[i] < slow.iloc[i]
        gap        = is_gap_bar(df, i)

        if (cross_up or cross_down) and not gap and not armed and pending_dir == 0:
            pending_dir     = 1 if cross_up else -1
            pending_cross_i = i

        # Step 2: angle confirmation
        if pending_dir != 0 and pending_cross_i is not None:
            bars_since = i - pending_cross_i
            if bars_since == conf_bars:
                denom = float(atr.iloc[i])
                if denom != 0:
                    slope    = (float(fast.iloc[i]) - float(fast.iloc[pending_cross_i])) / conf_bars
                    angle    = float(np.degrees(np.arctan(slope / denom)))
                    passed   = (angle >= min_angle if pending_dir == 1
                                else angle <= -min_angle)
                    if passed:
                        armed        = True
                        armed_dir    = pending_dir
                        wait_start_i = i
                pending_dir     = 0
                pending_cross_i = None
            elif bars_since > conf_bars:
                pending_dir     = 0
                pending_cross_i = None

        # Step 3: invalidation
        if armed and wait_start_i is not None:
            bars_waited  = i - wait_start_i
            timed_out    = bars_waited > max_wait
            trend_broken = (df["close"].iloc[i] < slow.iloc[i] if armed_dir == 1
                            else df["close"].iloc[i] > slow.iloc[i])
            if timed_out or trend_broken or is_force_close(ts):
                armed        = False
                armed_dir    = 0
                wait_start_i = None

        # Step 4: retest → entry (minimum 1 bar after arming)
        if armed and not is_force_close(ts) and wait_start_i is not None:
            bars_since_arm = i - wait_start_i
            if bars_since_arm >= 1:
                fa  = float(fast.iloc[i])
                buf = fa * buf_pct / 100
                if armed_dir == 1:
                    retest = df["low"].iloc[i] <= fa + buf
                else:
                    retest = df["high"].iloc[i] >= fa - buf

                if retest:
                    ep       = float(df["close"].iloc[i])
                    sl       = float(slow.iloc[i])
                    sl_dist  = abs(ep - sl)
                    if sl_dist < 1.0:
                        sl_dist = 1.0
                    tp = ep + armed_dir * rr * sl_dist

                    in_trade     = True
                    trade_dir    = armed_dir
                    entry_price  = ep
                    sl_price     = sl
                    target_price = tp
                    armed        = False
                    armed_dir    = 0
                    wait_start_i = None

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades, params):
    if not trades:
        return None

    df_t = pd.DataFrame(trades)
    total  = len(df_t)
    wins   = int(df_t["win"].sum())
    losses = total - wins

    if total < 3:   # skip configs with too few trades — not statistically meaningful
        return None

    win_rate    = wins / total * 100
    avg_win     = df_t.loc[df_t["win"] == 1, "option_pnl_est"].mean() if wins   > 0 else 0
    avg_loss    = df_t.loc[df_t["win"] == 0, "option_pnl_est"].mean() if losses > 0 else 0
    total_pnl   = df_t["option_pnl_est"].sum()
    gross_win   = df_t.loc[df_t["win"] == 1, "option_pnl_est"].sum()
    gross_loss  = abs(df_t.loc[df_t["win"] == 0, "option_pnl_est"].sum())
    profit_factor = round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf")

    # Max drawdown (on cumulative P&L curve)
    cum_pnl = df_t["option_pnl_est"].cumsum()
    roll_max = cum_pnl.cummax()
    drawdown = (cum_pnl - roll_max)
    max_dd   = round(drawdown.min(), 2)

    # Expectancy per trade
    expectancy = round(total_pnl / total, 2)

    by_reason  = df_t["exit_reason"].value_counts().to_dict()

    return {
        # Parameters
        **params,
        # Trade counts
        "total_trades":   total,
        "wins":           wins,
        "losses":         losses,
        "win_rate_pct":   round(win_rate, 1),
        # P&L (delta-scaled option estimate, per-lot)
        "total_pnl":      round(total_pnl, 2),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "expectancy":     expectancy,
        "profit_factor":  profit_factor,
        "max_drawdown":   max_dd,
        # Exit breakdown
        "sl_exits":       by_reason.get("SL", 0),
        "target_exits":   by_reason.get("Target", 0),
        "eod_exits":      by_reason.get("EOD", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Nifty 50 Strategy Backtester")
    print("=" * 60)

    # Fetch data once — reused across all param combinations
    df_raw = fetch_historical_data(months=LOOKBACK_MONTHS)
    df     = compute_indicators(df_raw, FAST_LEN, SLOW_LEN, ATR_LEN)

    # Build all combinations
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total_combos = len(combos)
    print(f"Testing {total_combos} parameter combinations...\n")

    results = []
    start_t = time.time()

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        trades = run_backtest(df, params)
        metrics = compute_metrics(trades, params)
        if metrics:
            results.append(metrics)

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == total_combos:
            elapsed = time.time() - start_t
            pct     = (idx + 1) / total_combos * 100
            eta     = (elapsed / (idx + 1)) * (total_combos - idx - 1)
            print(f"  [{idx+1:>4}/{total_combos}] {pct:.0f}% done | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s | "
                  f"valid configs so far: {len(results)}")

    print(f"\n✅ Done. {len(results)} valid configurations (≥3 trades).\n")

    if not results:
        print("⚠️  No valid configurations found. Try reducing min_angle_deg "
              "or extending the lookback period.")
        return

    df_results = pd.DataFrame(results)

    # ── Rank by profit factor (primary), then win rate (secondary) ────
    df_results = df_results.sort_values(
        ["profit_factor", "win_rate_pct", "total_pnl"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    df_results.index += 1  # rank starts at 1

    # Save full results
    all_path = "backtest_all_results.csv"
    df_results.to_csv(all_path)
    print(f"📄 All results saved → {all_path}")

    # Save top 20
    top20_path = "backtest_best_configs.csv"
    df_results.head(20).to_csv(top20_path)
    print(f"🏆 Top 20 saved       → {top20_path}\n")

    # Print top 10 to console
    print("=" * 60)
    print("  TOP 10 CONFIGURATIONS")
    print("=" * 60)
    display_cols = [
        "min_angle_deg", "angle_confirm_bars", "risk_reward_ratio",
        "max_wait_bars", "retest_buffer_pct",
        "total_trades", "win_rate_pct", "profit_factor",
        "total_pnl", "max_drawdown", "expectancy"
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df_results[display_cols].head(10).to_string())

    # ── Best config summary ────────────────────────────────────────────
    best = df_results.iloc[0]
    print("\n" + "=" * 60)
    print("  🥇 BEST CONFIGURATION")
    print("=" * 60)
    print(f"  Angle threshold:    {best['min_angle_deg']}°")
    print(f"  Confirm bars:       {best['angle_confirm_bars']}")
    print(f"  Risk:Reward:        1:{best['risk_reward_ratio']}")
    print(f"  Max wait bars:      {best['max_wait_bars']}")
    print(f"  Retest buffer:      {best['retest_buffer_pct']}%")
    print(f"  ─────────────────────────────────────")
    print(f"  Total trades:       {int(best['total_trades'])}")
    print(f"  Win rate:           {best['win_rate_pct']}%")
    print(f"  Profit factor:      {best['profit_factor']}")
    print(f"  Total P&L (est):    ₹{best['total_pnl']:.2f}  (delta-scaled, per-lot)")
    print(f"  Max drawdown (est): ₹{best['max_drawdown']:.2f}")
    print(f"  Expectancy:         ₹{best['expectancy']:.2f} per trade")
    print()
    print("  To apply these settings, update config.py:")
    print(f"    MIN_ANGLE_DEG        = {best['min_angle_deg']}")
    print(f"    ANGLE_CONFIRM_BARS   = {int(best['angle_confirm_bars'])}")
    print(f"    RISK_REWARD_RATIO    = {best['risk_reward_ratio']}")
    print(f"    MAX_WAIT_BARS        = {int(best['max_wait_bars'])}")
    print(f"    RETEST_BUFFER_PCT    = {best['retest_buffer_pct']}")
    print("=" * 60)

    # Warn if top config has very few trades
    if best["total_trades"] < 10:
        print("\n⚠️  Warning: best config has few trades — may be overfitting.")
        print("   Consider the top config with ≥15 trades instead:")
        robust = df_results[df_results["total_trades"] >= 15]
        if not robust.empty:
            r = robust.iloc[0]
            print(f"   Angle={r['min_angle_deg']}°  "
                  f"Confirm={int(r['angle_confirm_bars'])}  "
                  f"RR=1:{r['risk_reward_ratio']}  "
                  f"Wait={int(r['max_wait_bars'])}  "
                  f"PF={r['profit_factor']:.2f}  "
                  f"Win={r['win_rate_pct']}%  "
                  f"Trades={int(r['total_trades'])}")


if __name__ == "__main__":
    main()