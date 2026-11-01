"""
Configuration for the Nifty EMA Crossover + Angle + Retest paper trading bot.

Edit the values below as needed. Credentials (Upstox, Telegram) are loaded from
environment variables (~/.nifty_bot_env via systemd) — never hardcode them here.
"""

import os
from datetime import date

# ===== STRATEGY PARAMETERS (mirrors the Pine Script logic) =====
FAST_EMA_LEN = 9
SLOW_EMA_LEN = 15
ATR_LEN = 14
MIN_ANGLE_DEG = 7              # Entry threshold
ANGLE_CONFIRM_BARS = 2         # Candles after crossover to confirm angle
RETEST_BUFFER_PCT = 0.02       # % of fast EMA
USE_WICK_FOR_RETEST = True
MAX_WAIT_BARS = 15
RISK_REWARD_RATIO = 2.0        # Target = RR x initial risk (in premium points)
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 15

# ===== DATA SOURCE (Upstox API — Analytics Token) =====
UNDERLYING_TICKER = "NSE_INDEX|Nifty 50"   # Upstox instrument key
VIX_INSTRUMENT_KEY = "NSE_INDEX|India VIX"
CANDLE_INTERVAL   = "5m"
POLL_SECONDS      = 30
LTP_POLL_SECONDS      = 5    # Faster poll when armed and watching for EMA touch
POSITION_POLL_SECONDS = 10   # Poll when holding a position (exit detection)

# ===== OPTIONS (Upstox real option chain) =====
TARGET_PREMIUM = 95.0
LOT_SIZE       = 65    # Nifty lot size since Jan 2026 contracts (was 75)

# Liquidity filter for strike selection — skip strikes that would fill badly
MAX_SPREAD_PCT = 3.0   # (ask - bid) / mid, in %. Above this the strike is skipped.
MIN_STRIKE_OI  = 0     # Minimum open interest on the chosen strike (0 = off). The journal
                       # logs OI at every entry — set a floor once you've seen typical values.

# ===== STOP LOSS & TRAILING (premium points, per unit) =====
# Stop is placed beyond the recent swing low (CE) / swing high (PE) of Nifty,
# converted to premium with the option's real delta, then clamped so the
# premium stop is never tighter than SL_MIN_POINTS or wider than SL_MAX_POINTS.
SWING_LOOKBACK_BARS  = 6      # 5-min bars used to find the recent swing high/low
SWING_BUFFER_POINTS  = 5.0    # Extra Nifty points beyond the swing level
SL_MIN_POINTS        = 10.0   # Premium points (₹ per unit) — min risk
SL_MAX_POINTS        = 15.0   # Premium points (₹ per unit) — max risk
DEFAULT_DELTA        = 0.45   # Used only if the option chain has no delta

# Trailing stop: when the open profit reaches `at_r` x initial risk, the stop moves
# to `lock_r` x initial risk above entry (0.0 = breakeven + costs).
TRAIL_STEPS = [
    (1.0, 0.0),   # +1R reached  -> stop to breakeven (+ costs)
    (1.5, 0.5),   # +1.5R reached -> lock in +0.5R
]

# ===== POSITION & DAILY RISK RULES =====
MAX_OPEN_POSITIONS   = 1      # One trade at a time
MAX_LOSSES_PER_DAY   = 2      # Stop taking new trades after this many losing trades today
DAILY_MAX_LOSS_RUPEES = 2100  # Hard ₹ cap per day (≈ 2 losses at 15 pts x 65 + costs)
SCRATCH_RUPEES       = 100    # |net P&L| at or below this counts as a scratch, not a loss

# ===== TRANSACTION COSTS (verify against your Upstox contract note) =====
BROKERAGE_PER_ORDER  = 20.0       # ₹ per executed order
STT_SELL_PCT         = 0.15       # % of sell premium (options, from 1 Apr 2026)
EXCHANGE_TXN_PCT     = 0.03553    # % of premium, both sides (NSE options)
SEBI_FEE_PER_CRORE   = 10.0       # ₹ per crore of turnover, both sides
STAMP_DUTY_BUY_PCT   = 0.003      # % of buy premium
GST_PCT              = 18.0       # % on brokerage + exchange + SEBI fees

# ===== PAPER TRADING =====
PAPER_TRADING  = True           # Keep True until results are validated
TRADE_LOG_PATH = "paper_trades.csv"
STATE_PATH     = "bot_state.json"   # Open positions + today's risk counters (survives restarts)

# ===== JOURNAL =====
JOURNAL_DIR              = "journal"
EVENTS_CALENDAR_PATH     = "events.csv"   # Scheduled market events (RBI, Fed, Budget...)
SNAPSHOT_EVERY_MINUTES   = 15             # Market context snapshot cadence
JOURNAL_HOUR, JOURNAL_MINUTE = 15, 40     # When the daily journal is written

# ===== MARKET HOLIDAYS (NSE trading holidays on weekdays) =====
# Used for the expiry fallback and to skip holiday sessions. Add new years here.
NSE_HOLIDAYS = {
    date(2026, 11, 10),
    date(2026, 11, 24),
    date(2026, 12, 25),
}

# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
APPROVAL_TIMEOUT_SECONDS = 60  # Auto-executes if no response
