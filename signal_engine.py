"""
Direct port of the Pine Script strategy's signal logic:
crossover -> angle confirmation (ATR-normalized) -> retest -> entry,
with gap-candle skip and Slow-EMA / timeout invalidation.

IMPORTANT: Before trusting this for paper trading, run it against the same
historical period as your TradingView backtest and compare the entries -
they should match closely. Small EMA-seeding differences in the first few
bars of a dataset are normal and not a bug.
"""

import numpy as np
import pandas as pd


def compute_indicators(df, fast_len, slow_len, atr_len):
    df = df.copy()
    df["fast_ema"] = df["close"].ewm(span=fast_len, adjust=False).mean()
    df["slow_ema"] = df["close"].ewm(span=slow_len, adjust=False).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_len, adjust=False).mean()

    return df


def is_gap_bar(df, i):
    if i == 0:
        return False
    cur_day = df.index[i].date()
    prev_day = df.index[i - 1].date()
    if cur_day != prev_day:
        return df["open"].iloc[i] != df["close"].iloc[i - 1]
    return False


class SignalEngine:
    """
    Stateful engine mirroring the Pine script's `var` variables.
    Call process(df, flat) once per new closed candle, where `flat` tells
    the engine whether you currently have no open position (mirrors Pine's
    strategy.position_size == 0 gating).
    """

    def __init__(self, min_angle_deg, angle_confirm_bars, retest_buffer_pct,
                 use_wick_for_retest, max_wait_bars):
        self.min_angle_deg = min_angle_deg
        self.angle_confirm_bars = angle_confirm_bars
        self.retest_buffer_pct = retest_buffer_pct
        self.use_wick_for_retest = use_wick_for_retest
        self.max_wait_bars = max_wait_bars

        self.pending_dir = 0
        self.pending_cross_bar = None
        self.armed = False
        self.armed_dir = 0
        self.wait_start_bar = None

    def process(self, df, flat):
        """
        df must already have fast_ema/slow_ema/atr columns from compute_indicators().
        Returns a list of event tuples for the latest bar only.
        """
        events = []
        i = len(df) - 1
        min_bars = max(self.angle_confirm_bars, 20)
        if i < min_bars:
            return events

        fast, slow, atr = df["fast_ema"], df["slow_ema"], df["atr"]
        close, high, low = df["close"], df["high"], df["low"]

        cross_up = fast.iloc[i - 1] <= slow.iloc[i - 1] and fast.iloc[i] > slow.iloc[i]
        cross_down = fast.iloc[i - 1] >= slow.iloc[i - 1] and fast.iloc[i] < slow.iloc[i]
        gap_bar = is_gap_bar(df, i)

        # Step 1: new crossover -> start pending confirmation
        if (cross_up or cross_down) and not gap_bar and not self.armed and flat and self.pending_dir == 0:
            self.pending_dir = 1 if cross_up else -1
            self.pending_cross_bar = i
        elif (cross_up or cross_down) and gap_bar:
            events.append(("skipped_gap", 1 if cross_up else -1))

        # Step 2: angle confirmation
        if self.pending_dir != 0:
            bars_since = i - self.pending_cross_bar
            if bars_since == self.angle_confirm_bars:
                denom = atr.iloc[i] if atr.iloc[i] != 0 else np.nan
                avg_slope = (fast.iloc[i] - fast.iloc[i - self.angle_confirm_bars]) / self.angle_confirm_bars
                angle_deg = float(np.degrees(np.arctan(avg_slope / denom))) if not np.isnan(denom) else 0.0
                passed = angle_deg >= self.min_angle_deg if self.pending_dir == 1 else angle_deg <= -self.min_angle_deg
                events.append(("angle_check", self.pending_dir, angle_deg, passed))
                if passed:
                    self.armed = True
                    self.armed_dir = self.pending_dir
                    self.wait_start_bar = i
                    events.append(("armed", self.armed_dir))
                self.pending_dir = 0
                self.pending_cross_bar = None

        # Step 3: invalidation while armed
        if self.armed:
            timed_out = (i - self.wait_start_bar) > self.max_wait_bars
            trend_broken = (close.iloc[i] < slow.iloc[i]) if self.armed_dir == 1 else (close.iloc[i] > slow.iloc[i])
            if timed_out or trend_broken:
                events.append(("invalidated", "timeout" if timed_out else "trend_break"))
                self.armed = False
                self.armed_dir = 0

        # Step 4: retest -> entry signal (actual order placement handled by caller)
        if self.armed and flat:
            buffer = fast.iloc[i] * self.retest_buffer_pct / 100
            if self.armed_dir == 1:
                retest = (low.iloc[i] <= fast.iloc[i] + buffer) if self.use_wick_for_retest else (close.iloc[i] <= fast.iloc[i] + buffer)
            else:
                retest = (high.iloc[i] >= fast.iloc[i] - buffer) if self.use_wick_for_retest else (close.iloc[i] >= fast.iloc[i] - buffer)
            if retest:
                sl_price = float(slow.iloc[i])
                entry_price = float(close.iloc[i])
                events.append(("entry", self.armed_dir, entry_price, sl_price))
                self.armed = False
                self.armed_dir = 0

        return events
