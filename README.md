# Nifty EMA Crossover + Angle + Retest — Paper Trading Bot

A Python port of the TradingView Pine Script strategy, running fully free using
yfinance (underlying data) and NSE's public option chain (premiums), with
Telegram approval notifications. PAPER TRADING ONLY by default.

## Important limitations to know going in

- **yfinance data has no SLA** — it's free and generally reliable for Indian
  indices, but can lag or occasionally drop a candle. Don't treat it as
  exchange-grade for real-money decisions later.
- **NSE's option chain endpoint is unofficial** — it can rate-limit, block, or
  change format without notice. If `option_chain.py` starts failing, that's
  the most likely cause — check NSE hasn't changed their site.
- **This has NOT been run against live data yet** — I built it from the same
  logic as your validated Pine Script, but I can't execute it myself (no
  network access to these services from where I work). Please run a few days
  of paper trading and compare the trade log against what you'd expect from
  the Pine backtest before trusting it.
- **Run this on a machine that stays on during market hours** — your laptop
  sleeping or closing will stop the bot. A cheap always-on VPS is the
  eventual answer; for the 2-week paper test, anything that stays awake works.

## Setup

1. Install Python 3.10+ if you don't have it.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up Telegram notifications (optional but recommended):
   - Open Telegram, search for **@BotFather**, send `/newbot`, follow the
     prompts. It gives you a **bot token** — copy it.
   - Send any message to your new bot (search for it by the username you gave it).
   - Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
     — find `"chat":{"id":...}` in the response, that number is your **chat ID**.
   - Set these as environment variables before running:
     ```
     export TELEGRAM_BOT_TOKEN="your_token_here"
     export TELEGRAM_CHAT_ID="your_chat_id_here"
     ```
     (On Windows, use `set` instead of `export`, or add them in PowerShell with `$env:`.)
   - If you skip this, the bot still runs and auto-approves every trade — it
     just won't notify you.

## Running

```
python main.py
```

It will run continuously during market hours (9:15 AM–3:30 PM IST,
weekdays), checking for new candles every 30 seconds. Leave it running.

## Output

Every paper trade (win, loss, or EOD close) is logged to `paper_trades.csv`
in this folder — review this after your 2-week test to see win rate, average
P&L, and how often the strategy actually fired.

## Known simplifications in this v1 (worth knowing)

- **SL/target premium distance is approximated** using the same point
  distance as the underlying's SL/target, not scaled by option delta. Real
  option premium moves aren't 1:1 with the underlying, so simulated P&L will
  be directionally right but not exact. We can refine this with proper
  delta-based estimation once the core logic is validated.
- **No historical parity test included yet** — recommend running this and
  comparing entries against the same dates in your TradingView Strategy
  Tester to catch any logic drift between the Pine and Python versions.
