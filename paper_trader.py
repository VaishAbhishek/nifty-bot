"""
Simulates trade execution for paper trading: tracks open positions (supports
multiple simultaneous positions), checks SL/target/EOD-close each candle,
logs every trade to CSV, and enforces the consecutive-loss pause.
"""

import csv
import os
from datetime import datetime
import config


class PaperTrader:
    def __init__(self):
        self.positions = []   # list of open position dicts — supports multiple simultaneous trades
        self.consecutive_losses = 0
        self.paused_for_day = None
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
        """True only when there are zero open positions."""
        return len(self.positions) == 0

    @property
    def position(self):
        """Backwards-compat shim: return first open position or None."""
        return self.positions[0] if self.positions else None

    def is_paused(self, today):
        if self.paused_for_day is None:
            return False
        if today > self.paused_for_day:
            self.paused_for_day = None
            self.consecutive_losses = 0
            return False
        return True

    def open_position(self, direction, strike_info, underlying_entry, underlying_sl, rr_ratio, now):
        entry_premium = strike_info["premium"]
        sl_distance_underlying = abs(underlying_entry - underlying_sl)

        delta = 0.45
        sl_premium     = max(entry_premium - (sl_distance_underlying * delta), 0.5)
        target_premium = entry_premium + (sl_distance_underlying * delta * rr_ratio)

        pos = {
            "direction":      direction,
            "strike":         strike_info["strike"],
            "option_type":    strike_info["option_type"],
            "entry_premium":  entry_premium,
            "sl_premium":     sl_premium,
            "target_premium": target_premium,
            "entry_time":     now,
            "instrument_key": strike_info.get("instrument_key", ""),
        }
        self.positions.append(pos)
        return pos

    def check_exit(self, current_premium, now, force_close=False, instrument_key=None):
        """
        Check exit for a specific position (matched by instrument_key) or
        the first position if instrument_key is None.
        Returns exit info dict on close, None if no exit triggered.
        """
        if not self.positions:
            return None

        pos = None
        if instrument_key:
            for p in self.positions:
                if p.get("instrument_key") == instrument_key:
                    pos = p
                    break
        if pos is None:
            pos = self.positions[0]

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
        self.positions.remove(pos)
        return exit_info

    def check_all_exits(self, get_premium_fn, now, force_close=False):
        """
        Monitor all open positions and close any that hit SL / Target / EOD.
        get_premium_fn(instrument_key) -> current premium float or None.
        Returns list of exit_info dicts for every position closed this call.
        """
        closed = []
        for pos in list(self.positions):
            try:
                current_premium = get_premium_fn(pos.get("instrument_key", ""))
                if current_premium is None:
                    continue
            except Exception:
                continue
            exit_info = self.check_exit(
                current_premium, now,
                force_close=force_close,
                instrument_key=pos.get("instrument_key")
            )
            if exit_info:
                closed.append(exit_info)
        return closed

    def _log_trade(self, pos, exit_premium, exit_time, exit_reason, pnl_per_lot, result):
        with open(config.TRADE_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                pos["entry_time"], exit_time, pos["direction"],
                pos["strike"], pos["option_type"], pos.get("instrument_key", ""),
                pos["entry_premium"], exit_premium, pos["sl_premium"],
                pos["target_premium"], exit_reason, round(pnl_per_lot, 2), result
            ])
