"""
Minimal Telegram integration using raw HTTP calls to the Bot API - no extra
dependencies beyond `requests`.

Setup (see README.md for full steps):
1. Message @BotFather on Telegram, run /newbot, copy the token it gives you
2. Message your new bot once (anything), then visit
   https://api.telegram.org/bot<TOKEN>/getUpdates to find your chat_id
3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables
"""

import requests
import time
import config

API_BASE = "https://api.telegram.org/bot{token}"


def _url(method):
    return f"{API_BASE.format(token=config.TELEGRAM_BOT_TOKEN)}/{method}"


def send_message(text):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print(f"[Telegram not configured] {text}")
        return
    try:
        requests.post(_url("sendMessage"), data={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        print(f"[Telegram send failed: {e}] {text}")


def request_approval(text, timeout_seconds=None):
    """
    Sends a message with Yes/No inline buttons, then polls for a response.
    Returns True if approved (button tap OR timeout elapses - per your
    "auto-execute if no response" choice), False only if explicitly rejected.
    """
    timeout_seconds = timeout_seconds or config.APPROVAL_TIMEOUT_SECONDS

    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print(f"[Telegram not configured - auto-approving] {text}")
        return True

    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": "approve"},
            {"text": "❌ Reject", "callback_data": "reject"},
        ]]
    }

    try:
        resp = requests.post(_url("sendMessage"), data={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "reply_markup": __import__("json").dumps(keyboard),
        }, timeout=10)
        sent_message_id = resp.json()["result"]["message_id"]
    except Exception as e:
        print(f"[Telegram approval request failed: {e} - auto-approving]")
        return True

    deadline = time.time() + timeout_seconds
    last_update_id = None

    while time.time() < deadline:
        try:
            params = {"timeout": 5}
            if last_update_id is not None:
                params["offset"] = last_update_id + 1
            resp = requests.get(_url("getUpdates"), params=params, timeout=10)
            updates = resp.json().get("result", [])
            for update in updates:
                last_update_id = update["update_id"]
                cb = update.get("callback_query")
                if cb and cb["message"]["message_id"] == sent_message_id:
                    choice = cb["data"]
                    send_message("Got it - " + ("approved ✅" if choice == "approve" else "rejected ❌"))
                    return choice == "approve"
        except Exception as e:
            print(f"[Telegram poll error: {e}]")
        time.sleep(2)

    send_message(f"No response within {timeout_seconds}s - auto-executing as configured.")
    return True
