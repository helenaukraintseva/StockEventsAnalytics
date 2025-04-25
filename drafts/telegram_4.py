from telethon.sync import TelegramClient
from datetime import datetime, timedelta, timezone
from config import api_id, api_hash, phone
import asyncio

def parse_telegram_news(channel_title, days_back, api_id, api_hash, phone):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    start_date = datetime.now(timezone.utc)
    end_date = start_date - timedelta(days=days_back)

    client = TelegramClient('session_name', api_id, api_hash)
    client.start(phone=phone)

    all_messages = []
    offset_id = 0
    limit = 100

    with client:
        while True:
            messages = client.get_messages(f"@{channel_title}", limit=limit, offset_id=offset_id)
            if not messages:
                break

            for msg in messages:
                if msg.date < end_date:
                    break
                if end_date <= msg.date <= start_date:
                    message_url = f"https://t.me/{channel_title}/{msg.id}"
                    all_messages.append({
                        "text": msg.message or "",
                        "date": msg.date.strftime("%Y-%m-%d"),
                        "time": msg.date.strftime("%H:%M"),
                        "url": message_url
                    })

            offset_id = messages[-1].id
            if messages[-1].date < end_date:
                break

    return sorted(all_messages, key=lambda x: (x['date'], x['time']))


if __name__ == "__main__":

    news = parse_telegram_news("if_market_news", days_back=3, api_id=api_id, api_hash=api_hash, phone=phone)

    for item in news:
        print(f"[{item['date']} {item['time']}] {item['text']}")
