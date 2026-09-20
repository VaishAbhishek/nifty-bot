"""
Backtester for the Nifty EMA Crossover + Angle + Retest strategy.

Reuses SignalEngine exactly as the live bot does, but walks through
historical candles bar-by-bar instead of polling live data.

Usage:
    source ~/.nifty_bot_env
    python3 backtest.py                          # default: last 30 trading days
    python3 backtest.py --days 60                # last 60 trading days
    python3 backtest.py --from 2026-08-01 --to 2026-09-19

Output: backtest_trades.csv  (same columns as paper_trades.csv)
"""

import argparse
import csv
import os
import sys
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config
from signal_engine import SignalEngine, compute_indicators

IST = ZoneInfo("Asia/Kolkata")
BASE_V2 = "https://api.upstox.com/v2"
ENCODED_KEY = "NSE_INDEX%7CNifty%2050"
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
FORCE_CLOSE = dtime(config.FORCE_CLOSE_HOUR, config.FORCE_CLOSE_MINUTE)

OUTPUT_CSV = "backtest_trades.csv"

# ── Upstox historical data ────────────────────────────────────────────────


def _headers():
    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "")
    if not token:
        raise RuntimeError(
            "UPSTOX_ANALYTICS_TOKEN not set. Run: source ~/.nifty_bot_env"
        )
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _fetch_1min_chunk(from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch 1-min candles for a date range from Upstox v2 historical API."""
    url = f"{BASE_V2}/historical-candle/{ENCODED_KEY}/1minute/{to_date}/{from_date}"
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()

    candles = resp.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume", "oi"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def fetch_historical_5min(from_date: date, to_date: date) -> pd.DataFrame:
    """
    Fetch historical 1-min data in chunks (Upstox limits per-request range)
    and resample to 5-min candles.

    Upstox v2 historical API may limit to ~30 days per call, so we chunk
    into 25-day windows to be safe.
    """
    all_chunks = []
    chunk_start = from_date
    chunk_size = timedelta(days=25)

    print(f"Fetching historical data: {from_date} → {to_date}")
    while chunk_start <= to_date:
        chunk_end = min(chunk_start + chunk_size, to_date)
        print(f"  Chunk: {chunk_start} → {chunk_end} ...", end=" ", flush=True)

        try:
            df = _fetch_1min_chunk(
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            if not df.empty:
                all_chunks.append(df)
                print(f"{len(df)} bars")
            else:
                print("empty")
        except Exception as e:
            print(f"error: {e}")

        chunk_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        raise RuntimeError("No historical data fetched from Upstox.")

    df_1min = pd.concat(all_chunks).sort_index()
    df_1min = df_1min[~df_1min.index.duplicated(keep="last")]

    # Resample to 5-min
    df_5min = df_1min.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()

    df_5min = df_5min.between_time("09:15", "15:30")
    print(f"Total 5-min candles: {len(df_5min)}")
    return df_5min


# ── Backtest engine ────────────────────────────────────────────────────────


def default_params():
    """Build params dict from config.py values — used by standalone CLI."""
    return {
        "fast_ema_len": config.FAST_EMA_LEN,
        "slow_ema_len": config.SLOW_EMA_LEN,
        "atr_len": config.ATR_LEN,
        "min_angle_deg": config.MIN_ANGLE_DEG,
        "angle_confirm_bars": config.ANGLE_CONFIRM_BARS,
        "retest_buffer_pct": config.RETEST_BUFFER_PCT,
        "use_wick_for_retest": config.USE_WICK_FOR_RETEST,
        "max_wait_bars": config.MAX_WAIT_BARS,
        "risk_reward_ratio": config.RISK_REWARD_RATIO,
        "target_premium": config.TARGET_PREMIUM,
        "lot_size": config.LOT_SIZE,
        "max_consecutive_losses": config.MAX_CONSECUTIVE_LOSSES,
    }


class BacktestPosition:
    """Tracks a single simulated position during the backtest."""

    def __init__(self, direction, entry_bar_ts, entry_close, fast_ema, slow_ema,
                 rr_ratio, target_premium):
        self.direction = direction
        self.entry_time = entry_bar_ts

        # Entry at Fast EMA (retest price)
        self.entry_price = fast_ema

        # SL just beyond Slow EMA
        self.sl_price = slow_ema

        # Convert underlying levels to option premium terms
        sl_distance = abs(self.entry_price - self.sl_price)
        delta = 0.45  # approximate near-ATM delta

        self.entry_premium = target_premium
        self.sl_premium = max(self.entry_premium - (sl_distance * delta), 0.5)
        self.target_premium = self.entry_premium + (
            sl_distance * delta * rr_ratio
        )

        self.option_type = "CE" if direction == 1 else "PE"
        # Approximate strike (round to nearest 50)
        self.strike = int(round(entry_close / 50) * 50)


def run_backtest(df_5min: pd.DataFrame, params: dict = None,
                 verbose: bool = True) -> list[dict]:
    """
    Walk through candles bar-by-bar, running the SignalEngine on each,
    and simulate trades matching the live bot's logic.

    params: dict with strategy parameters (see default_params()).
            If None, uses config.py defaults.
    verbose: if False, suppresses per-trade print output (for optimizer).
    """
    if params is None:
        params = default_params()

    engine = SignalEngine(
        min_angle_deg=params["min_angle_deg"],
        angle_confirm_bars=params["angle_confirm_bars"],
        retest_buffer_pct=params["retest_buffer_pct"],
        use_wick_for_retest=params["use_wick_for_retest"],
        max_wait_bars=params["max_wait_bars"],
    )

    rr_ratio = params["risk_reward_ratio"]
    target_premium = params["target_premium"]
    lot_size = params["lot_size"]
    max_consec_losses = params["max_consecutive_losses"]
    fast_len = params["fast_ema_len"]
    slow_len = params["slow_ema_len"]
    atr_len = params["atr_len"]

    trades = []
    position = None  # BacktestPosition or None
    consecutive_losses = 0
    paused_date = None
    last_day = None

    for bar_idx in range(1, len(df_5min)):
        # Work with the slice [0..bar_idx] — only bars up to "now"
        df_slice = df_5min.iloc[: bar_idx + 1].copy()
        ts = df_slice.index[-1]
        current_day = ts.date()
        current_time = ts.time()

        # ── Daily reset ───────────────────────────────────────────────
        if last_day is not None and current_day != last_day:
            engine.reset_day()
            if paused_date is not None and current_day > paused_date:
                paused_date = None
                consecutive_losses = 0
        last_day = current_day

        # ── Skip if paused ────────────────────────────────────────────
        if paused_date is not None:
            continue

        # ── Compute indicators on the slice ───────────────────────────
        df_slice = compute_indicators(df_slice, fast_len, slow_len, atr_len)

        is_flat = position is None

        # ── Force close at 3:15 PM ────────────────────────────────────
        if position is not None and current_time >= FORCE_CLOSE:
            exit_premium = position.entry_premium  # approximate: exit at entry (flat)
            pnl_per_lot = 0.0
            result = "LOSS"
            trades.append(_make_trade_row(position, ts, exit_premium, "EOD", pnl_per_lot, result))
            if result == "LOSS":
                consecutive_losses += 1
                if consecutive_losses >= max_consec_losses:
                    paused_date = current_day
            position = None
            continue

        # ── Monitor open position (SL/target on underlying) ───────────
        if position is not None:
            close_now = float(df_slice["close"].iloc[-1])
            high_now = float(df_slice["high"].iloc[-1])
            low_now = float(df_slice["low"].iloc[-1])
            fast_ema_now = float(df_slice["fast_ema"].iloc[-1])
            slow_ema_now = float(df_slice["slow_ema"].iloc[-1])

            exit_reason = None
            if position.direction == 1:  # CE
                # SL: price drops below slow EMA → option premium collapses
                if low_now <= position.sl_price:
                    exit_reason = "SL"
                # Target: price rises enough → option premium hits target
                elif high_now >= position.entry_price + (
                    abs(position.entry_price - position.sl_price) * rr_ratio
                ):
                    exit_reason = "Target"
            else:  # PE
                # SL: price rises above slow EMA
                if high_now >= position.sl_price:
                    exit_reason = "SL"
                # Target: price drops enough
                elif low_now <= position.entry_price - (
                    abs(position.entry_price - position.sl_price) * rr_ratio
                ):
                    exit_reason = "Target"

            if exit_reason:
                sl_dist = abs(position.entry_price - position.sl_price)
                delta = 0.45

                if exit_reason == "SL":
                    exit_premium = position.sl_premium
                elif exit_reason == "Target":
                    exit_premium = position.target_premium
                else:
                    exit_premium = position.entry_premium

                pnl_per_lot = (exit_premium - position.entry_premium) * lot_size
                result = "WIN" if pnl_per_lot > 0 else "LOSS"
                trades.append(_make_trade_row(position, ts, exit_premium, exit_reason, pnl_per_lot, result))

                if result == "LOSS":
                    consecutive_losses += 1
                    if consecutive_losses >= max_consec_losses:
                        paused_date = current_day
                else:
                    consecutive_losses = 0
                position = None

        # ── Run signal engine ─────────────────────────────────────────
        events = engine.process(df_slice, flat=(position is None))

        for event in events:
            kind = event[0]

            if kind == "angle_check":
                _, direction, angle_deg, passed = event
                if verbose:
                    dir_str = "Bullish" if direction == 1 else "Bearish"
                    status = "PASSED ✅" if passed else "rejected"
                    print(f"  [{ts:%Y-%m-%d %H:%M}] {dir_str} angle={angle_deg:.1f}° {status}")

            elif kind == "armed":
                if verbose:
                    _, direction = event
                    dir_str = "CE" if direction == 1 else "PE"
                    print(f"  [{ts:%Y-%m-%d %H:%M}] Armed: {dir_str}")

            elif kind == "invalidated":
                if verbose:
                    _, reason = event
                    print(f"  [{ts:%Y-%m-%d %H:%M}] Invalidated: {reason}")

            elif kind == "entry" and position is None:
                _, direction, entry_close, slow_ema_val = event
                fast_ema_val = float(df_slice["fast_ema"].iloc[-1])

                position = BacktestPosition(
                    direction=direction,
                    entry_bar_ts=ts,
                    entry_close=entry_close,
                    fast_ema=fast_ema_val,
                    slow_ema=slow_ema_val,
                    rr_ratio=rr_ratio,
                    target_premium=target_premium,
                )
                if verbose:
                    dir_str = "CE" if direction == 1 else "PE"
                    print(
                        f"  [{ts:%Y-%m-%d %H:%M}] ENTRY {dir_str} @ fast_ema={fast_ema_val:.2f} "
                        f"SL={slow_ema_val:.2f} "
                        f"premium: ₹{position.entry_premium:.2f} "
                        f"SL_prem: ₹{position.sl_premium:.2f} "
                        f"TGT_prem: ₹{position.target_premium:.2f}"
                    )

    # ── Force close any open position at end of data ──────────────────
    if position is not None:
        ts = df_5min.index[-1]
        trades.append(_make_trade_row(position, ts, position.entry_premium, "EOD", 0.0, "LOSS"))

    return trades


def _make_trade_row(pos, exit_ts, exit_premium, exit_reason, pnl_per_lot, result):
    return {
        "entry_time": pos.entry_time,
        "exit_time": exit_ts,
        "direction": pos.direction,
        "strike": pos.strike,
        "option_type": pos.option_type,
        "instrument_key": "",
        "entry_premium": round(pos.entry_premium, 2),
        "exit_premium": round(exit_premium, 2),
        "sl_premium": round(pos.sl_premium, 2),
        "target_premium": round(pos.target_premium, 2),
        "exit_reason": exit_reason,
        "pnl_per_lot": round(pnl_per_lot, 2),
        "result": result,
    }


def write_csv(trades: list[dict], output_path: str):
    """Write trades to CSV matching paper_trades.csv format."""
    if not trades:
        print("No trades to write.")
        return

    fieldnames = [
        "entry_time", "exit_time", "direction", "strike",
        "option_type", "instrument_key",
        "entry_premium", "exit_premium", "sl_premium",
        "target_premium", "exit_reason", "pnl_per_lot", "result",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)

    print(f"\n✅ {len(trades)} trades written to {output_path}")


def print_summary(trades: list[dict]):
    """Print a quick summary of backtest results."""
    if not trades:
        print("\nNo trades generated during backtest period.")
        return

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pnl = sum(t["pnl_per_lot"] for t in trades)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Total trades : {len(trades)}")
    print(f"  Wins         : {len(wins)}")
    print(f"  Losses       : {len(losses)}")
    print(f"  Win rate     : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Total P&L    : ₹{total_pnl:,.2f} (per lot)")
    print(f"  Avg P&L/trade: ₹{total_pnl/len(trades):,.2f}")

    if wins:
        avg_win = sum(t["pnl_per_lot"] for t in wins) / len(wins)
        print(f"  Avg Win      : ₹{avg_win:,.2f}")
    if losses:
        avg_loss = sum(t["pnl_per_lot"] for t in losses) / len(losses)
        print(f"  Avg Loss     : ₹{avg_loss:,.2f}")

    # Breakdown by exit reason
    reasons = {}
    for t in trades:
        r = t["exit_reason"]
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\n  Exit reasons:")
    for reason, count in sorted(reasons.items()):
        print(f"    {reason}: {count}")

    # Breakdown by direction
    ce_trades = [t for t in trades if t["direction"] == 1]
    pe_trades = [t for t in trades if t["direction"] == -1]
    if ce_trades:
        ce_pnl = sum(t["pnl_per_lot"] for t in ce_trades)
        ce_wins = len([t for t in ce_trades if t["result"] == "WIN"])
        print(f"\n  CE trades: {len(ce_trades)} ({ce_wins}W) → ₹{ce_pnl:,.2f}")
    if pe_trades:
        pe_pnl = sum(t["pnl_per_lot"] for t in pe_trades)
        pe_wins = len([t for t in pe_trades if t["result"] == "WIN"])
        print(f"  PE trades: {len(pe_trades)} ({pe_wins}W) → ₹{pe_pnl:,.2f}")

    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest Nifty EMA strategy")
    parser.add_argument("--days", type=int, default=30, help="Number of calendar days to backtest (default 30)")
    parser.add_argument("--from", dest="from_date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", type=str, help="End date YYYY-MM-DD (default today)")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, help=f"Output CSV path (default {OUTPUT_CSV})")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.from_date:
        from_dt = date.fromisoformat(args.from_date)
    else:
        from_dt = date.today() - timedelta(days=args.days)

    to_dt = date.fromisoformat(args.to_date) if args.to_date else date.today()

    print(f"Strategy config:")
    print(f"  EMA: {config.FAST_EMA_LEN}/{config.SLOW_EMA_LEN}, ATR: {config.ATR_LEN}")
    print(f"  Angle: ≥{config.MIN_ANGLE_DEG}° after {config.ANGLE_CONFIRM_BARS} bars")
    print(f"  RR: 1:{config.RISK_REWARD_RATIO}, Max wait: {config.MAX_WAIT_BARS} bars")
    print(f"  Wick retest: {config.USE_WICK_FOR_RETEST}")
    print()

    # Fetch data
    df = fetch_historical_5min(from_dt, to_dt)

    if df.empty:
        print("No data available for the specified range.")
        sys.exit(1)

    print(f"Date range: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
    trading_days = df.index.normalize().nunique()
    print(f"Trading days: {trading_days}")
    print()

    # Run backtest
    trades = run_backtest(df)

    # Output
    write_csv(trades, args.output)
    print_summary(trades)


if __name__ == "__main__":
    main()
