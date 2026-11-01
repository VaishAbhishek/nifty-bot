# Nifty EMA Crossover + Angle + Retest — Paper Trading Bot

Python port of the TradingView Pine Script strategy. Uses Upstox (analytics
token) for Nifty candles, India VIX and the real option chain, with Telegram
notifications/approvals. **PAPER TRADING ONLY.**

## How a trade works

1. EMA 9/15 crossover on 5-min Nifty → angle check 2 bars later (≥ 7°, ATR-normalised).
2. Armed: wait up to 15 bars for price to touch the fast EMA (live LTP).
3. Strike: the **liquid** CE/PE (spread ≤ `MAX_SPREAD_PCT`) with premium nearest ₹95, nearest expiry
   (holiday shifts handled).
4. Fill at the **ask** using a fresh quote taken after Telegram approval (auto-approves after 60s).
5. Stop: beyond the recent swing low/high and slow EMA, converted with the option's real delta,
   clamped to **10–15 premium points**. Target = 2R.
6. Trailing: at +1R the stop moves to breakeven + costs; at +1.5R it locks +0.5R.
7. Exits are taken at the **bid**, minus charges (`costs.py`). Force-close at 15:15.

## Risk rules
- One open position at a time.
- No new trades after **2 losing trades** or **₹2,100 net loss** in a day (positions still monitored).
- Open positions and daily counters are saved to `bot_state.json`, so restarts lose nothing.

## Daily journal
Written at 15:40 into `journal/` and summarised on Telegram:
- `journal/YYYY-MM-DD.md` — market summary (gap, day type, VIX, PCR, OI walls, events), every trade
  with a rule-based "Why" list, the full signal timeline, and a **My notes** section that is kept
  when the page is regenerated.
- `journal/signals.csv` — every signal (passed, rejected, skipped, entered) with market context.
- `journal/market_snapshots.csv` — context every 15 minutes.
- `paper_trades.csv` — every closed trade with entry context, MFE/MAE, gross, costs, net.
- `journal/weekly-YYYY-MM-DD.md` — after the last trading day of each week: results grouped by
  entry time, weekday, VIX band, trend alignment, expiry distance and more.
  Run by hand: `python3 weekly_review.py --all`

Add scheduled events (RBI, Fed, Budget, CPI) to `events.csv`; trades on those days are tagged.
Add exchange holidays to `NSE_HOLIDAYS` in `config.py`.

## Setup
1. Python 3.10+, then `pip install -r requirements.txt`
2. `~/.nifty_bot_env` (plain `KEY=VALUE`, no `export`, for systemd):
   ```
   UPSTOX_ANALYTICS_TOKEN=...
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```
3. `python3 main.py` (or the `nifty_bot` systemd service)

## Known limits
- Backtests still approximate option P&L with a fixed delta (no theta/IV); treat them as optimistic.
  Real option-candle backtesting needs Upstox's expired-instrument data (Plus plan).
- Charges in `config.py` are estimates — check them against a real contract note.
- News and FII/DII flows aren't captured automatically (NSE blocks cloud IPs); use the
  **My notes** section of each day's journal.
- `data_feed.py`, `option_chain.py`, `option_chain_synth.py` are legacy (yfinance/NSE/synthetic)
  and not used by `main.py`.
