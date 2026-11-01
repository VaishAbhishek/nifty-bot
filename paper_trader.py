"""
Paper trade execution with realistic fills and the risk rules:

- Entry fills at the ASK, exits at the BID (LTP only if no depth), minus charges.
- Stop loss: beyond the recent swing low (CE) / swing high (PE) of Nifty and the
  slow EMA, converted to premium with the option's delta, clamped to
  SL_MIN_POINTS..SL_MAX_POINTS premium points.
- Trailing: TRAIL_STEPS moves the stop up as the trade moves in your favour
  (breakeven + costs at +1R, +0.5R locked at +1.5R by default).
- One position at a time; no new trades after MAX_LOSSES_PER_DAY losses or
  DAILY_MAX_LOSS_RUPEES net loss today.
- Open positions and today's counters are saved to STATE_PATH, so a crash or
  restart never loses a trade.
"""

import json
import os
from datetime import datetime

import config
import costs
import journal


def structure_stop(direction, df, slow_ema):
    """Underlying (Nifty) stop level from recent swing + slow EMA + buffer."""
    recent = df.iloc[-config.SWING_LOOKBACK_BARS:]
    if direction == 1:
        return min(float(recent["low"].min()), float(slow_ema)) - config.SWING_BUFFER_POINTS
    return max(float(recent["high"].max()), float(slow_ema)) + config.SWING_BUFFER_POINTS


def premium_risk_points(direction, underlying_entry, underlying_stop, delta):
    """Premium stop distance in points, clamped to the configured band."""
    delta = abs(delta) if delta else config.DEFAULT_DELTA
    raw = abs(underlying_entry - underlying_stop) * delta
    return round(min(max(raw, config.SL_MIN_POINTS), config.SL_MAX_POINTS), 2), round(raw, 2)


def _iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else dt


class PaperTrader:
    def __init__(self, state_path=None):
        self.state_path = state_path or config.STATE_PATH
        self.positions = []
        self.day = None
        self.losses_today = 0
        self.trades_today = 0
        self.pnl_today = 0.0
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                s = json.load(f)
            self.positions = s.get("positions", [])
            self.day = s.get("day")
            self.losses_today = s.get("losses_today", 0)
            self.trades_today = s.get("trades_today", 0)
            self.pnl_today = s.get("pnl_today", 0.0)
            if self.positions:
                print(f"[PaperTrader] Restored {len(self.positions)} open position(s) from state")
        except Exception as e:
            print(f"[PaperTrader] Could not read state ({e}); starting fresh")

    def _save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"positions": self.positions, "day": self.day,
                       "losses_today": self.losses_today, "trades_today": self.trades_today,
                       "pnl_today": self.pnl_today}, f, indent=1, default=str)
        os.replace(tmp, self.state_path)

    # ── daily rules ──────────────────────────────────────────────────────
    def new_day(self, today):
        iso = today.isoformat()
        if self.day != iso:
            self.day = iso
            self.losses_today = 0
            self.trades_today = 0
            self.pnl_today = 0.0
            self._save()

    def is_flat(self):
        return len(self.positions) == 0

    def can_open(self):
        """(allowed, reason) for opening a new position now."""
        if len(self.positions) >= config.MAX_OPEN_POSITIONS:
            return False, "position_open"
        if self.losses_today >= config.MAX_LOSSES_PER_DAY:
            return False, f"daily_loss_count ({self.losses_today} losses)"
        if self.pnl_today <= -config.DAILY_MAX_LOSS_RUPEES:
            return False, f"daily_loss_cap (₹{self.pnl_today:,.0f})"
        return True, ""

    def is_paused(self, today=None):
        """True when the daily loss rules block new trades (positions still monitored)."""
        ok, reason = self.can_open()
        return not ok and reason != "position_open"

    # ── entries ──────────────────────────────────────────────────────────
    def open_position(self, direction, strike_info, quote, underlying_entry,
                      underlying_stop, now, context=None, signal_id=""):
        """
        strike_info: from option_chain.find_strike_near_premium()
        quote: fresh {"bid","ask","ltp"} taken right before entry
        """
        ltp = float(quote.get("ltp") or strike_info["premium"])
        fill = float(quote.get("ask") or 0) or ltp
        risk, raw_risk = premium_risk_points(direction, underlying_entry, underlying_stop,
                                             strike_info.get("delta"))
        bid, ask = quote.get("bid") or 0, quote.get("ask") or 0
        spread = round((ask - bid) / ((ask + bid) / 2) * 100, 2) if bid and ask else None

        pos = {
            "trade_id": f"T{now:%Y%m%d-%H%M%S}",
            "signal_id": signal_id,
            "direction": direction,
            "strike": strike_info["strike"],
            "option_type": strike_info["option_type"],
            "expiry": strike_info.get("expiry"),
            "instrument_key": strike_info.get("instrument_key", ""),
            "entry_time": _iso(now),
            "entry_ltp": round(ltp, 2),
            "entry_fill": round(fill, 2),
            "entry_spread_pct": spread,
            "delta": strike_info.get("delta"),
            "iv": strike_info.get("iv"),
            "oi": strike_info.get("oi"),
            "underlying_entry": round(float(underlying_entry), 2),
            "underlying_stop": round(float(underlying_stop), 2),
            "structure_risk_points": raw_risk,
            "risk_points": risk,
            "initial_sl": round(max(fill - risk, 0.5), 2),
            "sl_premium": round(max(fill - risk, 0.5), 2),
            "target_premium": round(fill + risk * config.RISK_REWARD_RATIO, 2),
            "trail_stage": "initial",
            "peak": fill,
            "trough": fill,
            "last_price": fill,
            "missed_quotes": 0,
            "context": context or {},
        }
        self.positions.append(pos)
        self._save()
        return pos

    # ── exits ────────────────────────────────────────────────────────────
    def _apply_trailing(self, pos):
        risk, entry = pos["risk_points"], pos["entry_fill"]
        gain = pos["peak"] - entry
        for at_r, lock_r in config.TRAIL_STEPS:
            if gain >= at_r * risk:
                new_sl = entry + lock_r * risk
                if lock_r == 0:
                    new_sl += costs.cost_per_unit(entry)   # true breakeven after charges
                if new_sl > pos["sl_premium"]:
                    pos["sl_premium"] = round(new_sl, 2)
                    pos["trail_stage"] = f"+{at_r}R→{'BE' if lock_r == 0 else f'+{lock_r}R'}"

    def check_position(self, pos, quote, now, force_close=False):
        """Update one position with a quote; close it if a rule triggers."""
        if quote:
            price = float(quote.get("bid") or 0) or float(quote.get("ltp") or 0)
        else:
            price = 0.0

        if price <= 0:
            pos["missed_quotes"] += 1
            if force_close and pos["missed_quotes"] >= 3:
                return self._close(pos, pos["last_price"], now, "EOD_NOQUOTE")
            self._save()
            return None

        pos["missed_quotes"] = 0
        pos["last_price"] = price
        pos["peak"] = max(pos["peak"], price)
        pos["trough"] = min(pos["trough"], price)
        self._apply_trailing(pos)

        if force_close:
            return self._close(pos, price, now, "EOD")
        if price <= pos["sl_premium"]:
            reason = "TRAIL_SL" if pos["sl_premium"] > pos["initial_sl"] else "SL"
            return self._close(pos, price, now, reason)
        if price >= pos["target_premium"]:
            return self._close(pos, price, now, "Target")
        self._save()
        return None

    def check_all_exits(self, get_quote_fn, now, force_close=False):
        """get_quote_fn(instrument_key) -> {"bid","ask","ltp"} or None."""
        closed = []
        for pos in list(self.positions):
            try:
                quote = get_quote_fn(pos.get("instrument_key", ""))
            except Exception as e:
                print(f"[PaperTrader] quote error: {e}")
                quote = None
            info = self.check_position(pos, quote, now, force_close)
            if info:
                closed.append(info)
        return closed

    def _close(self, pos, exit_price, now, reason):
        entry = pos["entry_fill"]
        gross = (exit_price - entry) * config.LOT_SIZE
        charges = costs.round_trip_charges(entry, exit_price)
        net = gross - charges
        result = "SCRATCH" if abs(net) <= config.SCRATCH_RUPEES else ("WIN" if net > 0 else "LOSS")
        entry_dt = datetime.fromisoformat(pos["entry_time"])
        risk = pos["risk_points"]

        row = {k: pos.get(k) for k in journal.TRADE_FIELDS if k in pos}
        row.update(pos.get("context") or {})
        row.update({
            "date": entry_dt.date().isoformat(),
            "entry_time": entry_dt.strftime("%H:%M:%S"),
            "exit_time": now.strftime("%H:%M:%S"),
            "minutes_held": round((now - entry_dt).total_seconds() / 60, 1),
            "final_sl": pos["sl_premium"],
            "exit_premium": round(exit_price, 2),
            "exit_reason": reason,
            "mfe_points": round(pos["peak"] - entry, 2),
            "mae_points": round(entry - pos["trough"], 2),
            "mfe_r": round((pos["peak"] - entry) / risk, 2) if risk else None,
            "gross_pnl": round(gross, 2),
            "charges": charges,
            "net_pnl": round(net, 2),
            "result": result,
        })
        journal.record_trade(row)

        self.trades_today += 1
        self.pnl_today += net
        if result == "LOSS":
            self.losses_today += 1
        self.positions.remove(pos)
        self._save()
        return row
