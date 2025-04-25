import os
import time
import csv
import logging
from datetime import datetime, timedelta, timezone
from telethon.sync import TelegramClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
phone = os.getenv("PHONE_NUMBER")

start_date = datetime.now(timezone.utc)
end_date = start_date - timedelta(days=1)

channels = os.getenv("TELEGRAM_CHANNELS", "if_market_news").split(",")

client = TelegramClient("session_name", api_id, api_hash)
client.start(phone=phone)
logging.info("Успешная авторизация")


def fetch_messages(channel: str, output_dir: str = "channels_content/"):
    """
    Получает сообщения из канала Telegram в заданном диапазоне дат и сохраняет в CSV.

    :param channel: username канала без @
    :param output_dir: папка для CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{channel}.csv")

    all_messages = []
    offset_id = 0
    limit = 100
    round_counter = 0

    while True:
        logging.info("Запрос #%d для канала %s", round_counter, channel)
        round_counter += 1
        time.sleep(2)

        messages = client.get_messages(f"@{channel}", limit=limit, offset_id=offset_id)
        if not messages:
            break

        for msg in messages:
            if msg.date < start_date:
                break
            if start_date >= msg.date >= end_date:
                all_messages.append(msg)

        offset_id = messages[-1].id
        if messages[-1].date < start_date:
            break

    all_messages.sort(key=lambda m: m.date)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["message_id", "date", "text"])
        for msg in all_messages:
            writer.writerow([msg.id, msg.date, msg.message or ""])

    logging.info("Сохранено %d сообщений в %s", len(all_messages), output_file)


if __name__ == "__main__":
    with client:
        for channel in channels:
            try:
                fetch_messages(channel.strip())
            except Exception as ex:
                logging.error("Ошибка в канале %s: %s", channel, ex)
