import httpx
from config.settings import settings


def send_telegram(message: str) -> None:
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    httpx.post(url, json={
        "chat_id": settings.telegram_chat_id,
        "text": f"🔔 *vdetec*\n{message}",
        "parse_mode": "Markdown",
    }, timeout=10)
