import os
import sys
import csv
import logging
from telethon.sync import TelegramClient
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Переменные окружения
api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
phone = os.getenv("PHONE_NUMBER")

client = TelegramClient("session_name", api_id, api_hash)


def main():
    """
    Авторизуется в Telegram, запрашивает канал, скачивает и сохраняет сообщения.
    """
    client.start(phone=phone)
    logging.info("Успешная авторизация")

    channel_username = input("Введите @username канала (например, @testchannel): ").strip()
    if not channel_username.startswith("@"):
        logging.error("Неверный формат канала. Должно начинаться с @")
        sys.exit(1)

    messages = client.get_messages(channel_username, limit=1000)
    logging.info("Всего сообщений получено: %d", len(messages))

    with open("channel_posts.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["message_id", "date", "text"])
        for msg in messages:
            writer.writerow([msg.id, msg.date, msg.message or ""])

    logging.info("Данные сохранены в 'channel_posts.csv'")


if __name__ == "__main__":
    with client:
        main()
