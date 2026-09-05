"""
Direct port of the Pine Script strategy's signal logic.

Key fixes vs original:
1. Timestamp-based crossover tracking (not integer index) — fixes index drift
   when yfinance returns different-length dataframes across polls.
2. Partial candle filtering moved to data_feed.py — engine only sees closed candles.
3. Gap filter matches Pine exactly: only the first candle of a new day with a gap
   is skipped (not 3 candles).
4. wait_start_ts is timestamp-based — fixes timeout drift.
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
    """
    Matches Pine exactly: only the FIRST candle of a new day is a gap bar,
    and only when open != prior day's close.
    Pine: isNewDay = ta.change(time("D")) != 0
          isGapBar = isNewDay and (open != close[1])
    """
    if i == 0:
        return False
    cur_day = df.index[i].date()
    prev_day = df.index[i - 1].date()
    if cur_day != prev_day:
        return df["open"].iloc[i] != df["close"].iloc[i - 1]
    return False


class SignalEngine:
    """
    Stateful engine mirroring the Pine script's var variables.
    Uses timestamps for crossover tracking to avoid integer index drift.
    """

    def __init__(self, min_angle_deg, angle_confirm_bars, retest_buffer_pct,
                 use_wick_for_retest, max_wait_bars):
        self.min_angle_deg = min_angle_deg
        self.angle_confirm_bars = angle_confirm_bars
        self.retest_buffer_pct = retest_buffer_pct
        self.use_wick_for_retest = use_wick_for_retest
        self.max_wait_bars = max_wait_bars

        # All state uses timestamps, not integer indices
        self.pending_dir = 0
        self.pending_cross_ts = None      # timestamp of crossover candle
        self.armed = False
        self.armed_dir = 0
        self.wait_start_ts = None         # timestamp when setup was armed

    def reset_day(self):
        """Call at start of each new trading day to match Pine's bar-history reset."""
        self.pending_dir = 0
        self.pending_cross_ts = None
        self.armed = False
        self.armed_dir = 0
        self.wait_start_ts = None

    def _get_loc(self, df, ts):
        """Return integer position of timestamp in df, or None if not found."""
        try:
            return df.index.get_loc(ts)
        except KeyError:
            return None

    def process(self, df, flat):
        """
        df must have fast_ema/slow_ema/atr columns from compute_indicators().
        Only call with a fully-closed candle as the last row (data_feed handles this).
        Returns list of event tuples for the latest bar only.
        """
        events = []
        i = len(df) - 1

        # Need enough bars for EMA seeding
        if i < max(self.angle_confirm_bars, 20):
            return events

        fast = df["fast_ema"]
        slow = df["slow_ema"]
        atr  = df["atr"]
        close, high, low = df["close"], df["high"], df["low"]
        ts = df.index[i]

        # Crossover detection (matches Pine: strict < on prev bar)
        cross_up   = fast.iloc[i-1] < slow.iloc[i-1] and fast.iloc[i] > slow.iloc[i]
        cross_down = fast.iloc[i-1] > slow.iloc[i-1] and fast.iloc[i] < slow.iloc[i]
        gap_bar    = is_gap_bar(df, i)

        # ===== STEP 1: CROSSOVER → PENDING =====
        # Matches Pine exactly: not armed, flat, no existing pending
        if (cross_up or cross_down) and not gap_bar and not self.armed and flat and self.pending_dir == 0:
            self.pending_dir = 1 if cross_up else -1
            self.pending_cross_ts = ts
        elif (cross_up or cross_down) and gap_bar:
            events.append(("skipped_gap", 1 if cross_up else -1))

        # ===== STEP 2: ANGLE CONFIRMATION =====
        if self.pending_dir != 0 and self.pending_cross_ts is not None:
            cross_pos = self._get_loc(df, self.pending_cross_ts)

            if cross_pos is None:
                # Crossover candle dropped from yfinance window — abandon
                self.pending_dir = 0
                self.pending_cross_ts = None
            else:
                bars_since = i - cross_pos

                if bars_since == self.angle_confirm_bars:
                    denom = float(atr.iloc[i])
                    avg_slope = (float(fast.iloc[i]) - float(fast.iloc[cross_pos])) / self.angle_confirm_bars
                    angle_deg = float(np.degrees(np.arctan(avg_slope / denom))) if denom != 0 else 0.0

                    passed = (angle_deg >= self.min_angle_deg if self.pending_dir == 1
                              else angle_deg <= -self.min_angle_deg)
                    events.append(("angle_check", self.pending_dir, angle_deg, passed))

                    if passed:
                        self.armed = True
                        self.armed_dir = self.pending_dir
                        self.wait_start_ts = ts
                        events.append(("armed", self.armed_dir))

                    self.pending_dir = 0
                    self.pending_cross_ts = None

                elif bars_since > self.angle_confirm_bars:
                    # Missed the confirmation window — abandon
                    self.pending_dir = 0
                    self.pending_cross_ts = None

        # ===== STEP 3: INVALIDATION WHILE ARMED =====
        if self.armed and self.wait_start_ts is not None:
            wait_pos = self._get_loc(df, self.wait_start_ts)
            bars_waited = (i - wait_pos) if wait_pos is not None else self.max_wait_bars + 1

            timed_out    = bars_waited > self.max_wait_bars
            trend_broken = (close.iloc[i] < slow.iloc[i]) if self.armed_dir == 1 else (close.iloc[i] > slow.iloc[i])

            if timed_out or trend_broken:
                reason = "timeout" if timed_out else "trend_break"
                events.append(("invalidated", reason))
                self.armed = False
                self.armed_dir = 0
                self.wait_start_ts = None

        # ===== STEP 4: RETEST → ENTRY =====
        if self.armed and flat:
            buffer = float(fast.iloc[i]) * self.retest_buffer_pct / 100

            if self.armed_dir == 1:
                retest = (low.iloc[i] <= fast.iloc[i] + buffer) if self.use_wick_for_retest \
                         else (close.iloc[i] <= fast.iloc[i] + buffer)
            else:
                retest = (high.iloc[i] >= fast.iloc[i] - buffer) if self.use_wick_for_retest \
                         else (close.iloc[i] >= fast.iloc[i] - buffer)

            if retest:
                events.append(("entry", self.armed_dir, float(close.iloc[i]), float(slow.iloc[i])))
                self.armed = False
                self.armed_dir = 0
                self.wait_start_ts = None

        return events