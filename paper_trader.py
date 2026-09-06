"""
Simulates trade execution for paper trading: tracks an open position,
checks SL/target/EOD-close each candle, logs every trade to CSV, and
enforces the consecutive-loss pause (auto-resumes next trading day).
"""

import csv
import os
from datetime import datetime
import config


class PaperTrader:
    def __init__(self):
        self.position = None  # dict: direction, strike, option_type, entry_premium, sl, target, entry_time
        self.consecutive_losses = 0
        self.paused_for_day = None  # date when pause was triggered; resumes automatically the next day
        self._init_log()

    def _init_log(self):
        if not os.path.exists(config.TRADE_LOG_PATH):
            with open(config.TRADE_LOG_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "entry_time", "exit_time", "direction", "strike",
                    "option_type", "instrument_key",
                    "entry_premium", "exit_premium", "sl_premium",
                    "target_premium", "exit_reason", "pnl_per_lot", "result"
                ])

    def is_flat(self):
        return self.position is None

    def is_paused(self, today):
        if self.paused_for_day is None:
            return False
        if today > self.paused_for_day:
            # New trading day - auto-resume
            self.paused_for_day = None
            self.consecutive_losses = 0
            return False
        return True

    def open_position(self, direction, strike_info, underlying_entry, underlying_sl, rr_ratio, now):
        entry_premium = strike_info["premium"]
        sl_distance_underlying = abs(underlying_entry - underlying_sl)

        # Scale by approximate delta for near-ATM options
        delta = 0.45
        sl_premium     = max(entry_premium - (sl_distance_underlying * delta), 0.5)
        target_premium = entry_premium + (sl_distance_underlying * delta * rr_ratio)

        self.position = {
            "direction":      direction,
            "strike":         strike_info["strike"],
            "option_type":    strike_info["option_type"],
            "entry_premium":  entry_premium,
            "sl_premium":     sl_premium,
            "target_premium": target_premium,
            "entry_time":     now,
            "instrument_key": strike_info.get("instrument_key", ""),  # Upstox key for live LTP
        }
        return self.position

        self.position = {
            "direction": direction,
            "strike": strike_info["strike"],
            "option_type": strike_info["option_type"],
            "entry_premium": entry_premium,
            "sl_premium": sl_premium,
            "target_premium": target_premium,
            "entry_time": now,
        }
        return self.position

    def check_exit(self, current_premium, now, force_close=False):
        """Call every candle while a position is open. Returns exit info dict or None."""
        if self.position is None:
            return None

        pos = self.position
        exit_reason = None

        if force_close:
            exit_reason = "EOD"
        elif current_premium <= pos["sl_premium"]:
            exit_reason = "SL"
        elif current_premium >= pos["target_premium"]:
            exit_reason = "Target"

        if exit_reason is None:
            return None

        pnl_per_lot = (current_premium - pos["entry_premium"]) * config.LOT_SIZE
        result = "WIN" if pnl_per_lot > 0 else "LOSS"

        self._log_trade(pos, current_premium, now, exit_reason, pnl_per_lot, result)

        if result == "LOSS":
            self.consecutive_losses += 1
            if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
                self.paused_for_day = now.date()
        else:
            self.consecutive_losses = 0

        exit_info = {**pos, "exit_premium": current_premium, "exit_time": now,
                     "exit_reason": exit_reason, "pnl_per_lot": pnl_per_lot, "result": result}
        self.position = None
        return exit_info

    def _log_trade(self, pos, exit_premium, exit_time, exit_reason, pnl_per_lot, result):
        with open(config.TRADE_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                pos["entry_time"], exit_time, pos["direction"],
                pos["strike"], pos["option_type"], pos.get("instrument_key", ""),
                pos["entry_premium"], exit_premium, pos["sl_premium"],
                pos["target_premium"], exit_reason, round(pnl_per_lot, 2), result
            ])
