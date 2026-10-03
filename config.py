"""
Configuration for the Nifty EMA Crossover + Angle + Retest paper trading bot.

Edit the values below as needed. Telegram credentials should be set as
environment variables (see README.md) rather than hardcoded here.
"""

import os

# ===== STRATEGY PARAMETERS (mirrors the Pine Script logic exactly) =====
FAST_EMA_LEN = 9
SLOW_EMA_LEN = 15
ATR_LEN = 14
MIN_ANGLE_DEG = 7              # Entry threshold (lowered from 30 per latest decision)
ANGLE_CONFIRM_BARS = 2         # Candles after crossover to confirm angle (test 1/2/3)
RETEST_BUFFER_PCT = 0.02     
USE_WICK_FOR_RETEST = True
MAX_WAIT_BARS = 15             
RISK_REWARD_RATIO = 2.0        # test 1.0, then 2.0
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 15

# ===== DATA SOURCE (Upstox API — Analytics Token) =====
# Token is loaded from ~/.nifty_bot_env — never hardcode here
UNDERLYING_TICKER = "NSE_INDEX|Nifty 50"   # Upstox instrument key
CANDLE_INTERVAL   = "5m"
POLL_SECONDS      = 30

# ===== OPTIONS (Upstox real option chain) =====
TARGET_PREMIUM = 95.0
LOT_SIZE       = 75    # Verify current NSE Nifty lot size before going live

# ===== PAPER TRADING =====
PAPER_TRADING = True           # Keep True until you've validated results and are ready to risk real money
TRADE_LOG_PATH = "paper_trades.csv"

# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
APPROVAL_TIMEOUT_SECONDS = 60  # 1 minute, auto-executes if no response

# ===== RISK CONTROL =====
MAX_CONSECUTIVE_LOSSES = 10     # Pause new entries after this many losses in a row
                                # Auto-resumes the next trading day
