"""
Trading-day, expiry and scheduled-event helpers.

Holidays come from config.NSE_HOLIDAYS; scheduled events (RBI policy, US Fed,
Budget, holidays that shift expiry...) come from events.csv, which you can
edit by hand. The journal tags every signal and trade with today's events.
"""

import csv
import os
from datetime import date, timedelta

import config


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in config.NSE_HOLIDAYS


def prev_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def weekly_expiry_for(d: date) -> date:
    """
    Nifty weekly expiry for the week containing (or after) `d`: normally Tuesday;
    if that Tuesday is a holiday, the previous trading day.
    """
    days_to_tuesday = (1 - d.weekday()) % 7
    tuesday = d + timedelta(days=days_to_tuesday)
    expiry = tuesday if is_trading_day(tuesday) else prev_trading_day(tuesday)
    if expiry < d:   # holiday shift moved it before today -> next week's expiry
        return weekly_expiry_for(tuesday + timedelta(days=1))
    return expiry


def days_to_expiry(today: date, expiry: date) -> int:
    """Trading days from today until expiry (0 = expiry day)."""
    n, d = 0, today
    while d < expiry:
        d += timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n


def load_events(path=None):
    path = path or config.EVENTS_CALENDAR_PATH
    events = []
    if not os.path.exists(path):
        return events
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("date") or row["date"].startswith("#"):
                continue
            events.append(row)
    return events


def events_for(d: date, path=None):
    """List of 'event (time)' strings for date d."""
    iso = d.isoformat()
    out = []
    for e in load_events(path):
        if e["date"].strip() == iso:
            t = (e.get("time") or "").strip()
            out.append(f"{e['event'].strip()}" + (f" ({t})" if t else ""))
    return out
