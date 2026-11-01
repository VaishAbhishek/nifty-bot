"""
Weekly review: groups closed trades by the context they happened in, so
patterns (which hours, VIX levels, trend alignment... work or don't) show up.

Runs automatically after the last trading day of each week, or by hand:
    python3 weekly_review.py              # last 7 days
    python3 weekly_review.py --days 30    # last 30 days
    python3 weekly_review.py --all        # everything logged
"""

import argparse
import csv
import os
from datetime import date, timedelta

import config
import journal


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _vix_band(v):
    v = _f(v)
    if v is None:
        return "unknown"
    return "<12" if v < 12 else "12-16" if v < 16 else "16-20" if v < 20 else "20+"


def _aligned(t, key):
    v = t.get(key)
    if not v:
        return "unknown"
    want = "up" if t.get("option_type") == "CE" else "down"
    return "with trade" if v == want else "against trade"


GROUPINGS = [
    ("Entry time", lambda t: t.get("time_bucket") or "unknown"),
    ("Weekday", lambda t: t.get("weekday") or "unknown"),
    ("Days to expiry", lambda t: t.get("days_to_expiry") or "unknown"),
    ("India VIX", lambda t: _vix_band(t.get("vix"))),
    ("15-min trend", lambda t: _aligned(t, "trend_15m")),
    ("First 30 min direction", lambda t: _aligned(t, "first30_dir")),
    ("Price vs prior-day range", lambda t: t.get("vs_prev_range") or "unknown"),
    ("Direction", lambda t: t.get("option_type") or "unknown"),
    ("Exit reason", lambda t: t.get("exit_reason") or "unknown"),
    ("Event day", lambda t: "yes" if t.get("events") else "no"),
]


def _table(trades, keyfn):
    groups = {}
    for t in trades:
        groups.setdefault(keyfn(t), []).append(t)
    rows = ["| Group | Trades | Win % | Net ₹ | Avg net ₹ |", "| --- | --- | --- | --- | --- |"]
    for k in sorted(groups, key=str):
        g = groups[k]
        net = sum(_f(x.get("net_pnl")) or 0 for x in g)
        wins = sum(x.get("result") == "WIN" for x in g)
        rows.append(f"| {k} | {len(g)} | {wins / len(g) * 100:.0f}% | {net:,.0f} | {net / len(g):,.0f} |")
    return rows


def write_weekly(end: date, days=7, all_time=False):
    start = date.min if all_time else end - timedelta(days=days - 1)
    trades = []
    if os.path.exists(config.TRADE_LOG_PATH):
        with open(config.TRADE_LOG_PATH, newline="") as f:
            for t in csv.DictReader(f):
                try:
                    d = date.fromisoformat(t.get("date", ""))
                except ValueError:
                    continue
                if start <= d <= end:
                    trades.append(t)

    sig_path = os.path.join(config.JOURNAL_DIR, "signals.csv")
    funnel = {}
    if os.path.exists(sig_path):
        with open(sig_path, newline="") as f:
            for s in csv.DictReader(f):
                try:
                    d = date.fromisoformat(s["date"])
                except ValueError:
                    continue
                if start <= d <= end:
                    key = s["kind"] + (f" ({s['detail'].split(' (')[0]})"
                                       if s["kind"] in ("entry_skipped", "invalidated") else "")
                    funnel[key] = funnel.get(key, 0) + 1

    period = "all time" if all_time else f"{start} to {end}"
    L = [f"# Weekly review ({period})", ""]
    n = len(trades)
    net = sum(_f(t.get("net_pnl")) or 0 for t in trades)
    gross = sum(_f(t.get("gross_pnl")) or 0 for t in trades)
    charges = sum(_f(t.get("charges")) or 0 for t in trades)
    wins = [t for t in trades if t.get("result") == "WIN"]
    losses = [t for t in trades if t.get("result") == "LOSS"]
    L.append("## Summary")
    L.append(f"- Trades {n}: wins {len(wins)}, losses {len(losses)}, scratches {n - len(wins) - len(losses)}")
    if n:
        avg_w = sum(_f(t["net_pnl"]) for t in wins) / len(wins) if wins else 0
        avg_l = sum(_f(t["net_pnl"]) for t in losses) / len(losses) if losses else 0
        pf = (sum(_f(t["net_pnl"]) for t in wins) / abs(sum(_f(t["net_pnl"]) for t in losses))
              if losses else float("inf"))
        L.append(f"- Net ₹{net:,.0f} (gross ₹{gross:,.0f}, costs ₹{charges:,.0f}); "
                 f"avg win ₹{avg_w:,.0f}, avg loss ₹{avg_l:,.0f}; profit factor {pf:.2f}")
        L.append(f"- Win rate {len(wins) / n * 100:.0f}% — with {n} trades this can be off by "
                 f"±{(1.96 * ((len(wins) / n) * (1 - len(wins) / n) / n) ** 0.5) * 100:.0f} points")
    L.append("")
    L.append("## Signal funnel")
    for k, v in sorted(funnel.items()):
        L.append(f"- {k}: {v}")
    L.append("")
    if n:
        for title, fn in GROUPINGS:
            L.append(f"## By {title.lower()}")
            L.extend(_table(trades, fn))
            L.append("")
    L.append("Small groups (1–3 trades) are noise. Look for patterns that hold over several weeks.")

    os.makedirs(config.JOURNAL_DIR, exist_ok=True)
    path = os.path.join(config.JOURNAL_DIR, f"weekly-{end.isoformat()}.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    print(write_weekly(date.today(), a.days, a.all))
