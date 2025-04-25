import csv
import asyncio
import os
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import Channel, Chat

class TelegramParser:
    """
    Класс для парсинга (получения) постов из публичных (или доступных) Telegram-каналов.
    """

    def __init__(self, api_id, api_hash, phone=None, session_name="session", proxy=None):
        """
        :param api_id: Ваш API_ID с my.telegram.org
        :param api_hash: Ваш API_HASH
        :param phone: номер телефона (при необходимости, если нет готовой сессии)
        :param session_name: имя файла/сессии, например 'anon'
        :param proxy: при необходимости, прокси
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.session_name = session_name
        self.client = None
        self.proxy = proxy  # например, (socks.SOCKS5, '127.0.0.1', 9050) если нужно

    async def connect(self):
        """Инициализирует и подключается к Telegram."""
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash, proxy=self.proxy)
        await self.client.start(self.phone)

        # Если включена двухфакторная аутентификация - запрашивается пароль:
        if not await self.client.is_user_authorized():
            try:
                await self.client.sign_in(self.phone)
            except SessionPasswordNeededError:
                pw = input("Введите пароль для 2FA: ")
                await self.client.sign_in(password=pw)

        print("[INFO] Успешное подключение к Telegram.")

    async def parse_channel(self, channel_username, limit=50, csv_filename="telegram_posts.csv"):
        """
        Парсит заданный канал (username или ссылка t.me/...) и сохраняет последние N сообщений в CSV.
        :param channel_username: строка формата '@channel_name' или 'https://t.me/channel_name'
        :param limit: сколько сообщений получить (по умолч. 50)
        :param csv_filename: куда сохранить результат
        """
        if not self.client:
            print("[ERROR] Сначала вызовите connect()")
            return

        # Попытаемся получить entity канала
        try:
            entity = await self.client.get_entity(channel_username)
        except Exception as e:
            print(f"[ERROR] Не удалось получить информацию о канале '{channel_username}': {e}")
            return

        # Проверим, действительно ли это канал (Channel) или супергруппа
        if not isinstance(entity, (Channel, Chat)):
            print(f"[WARN] '{channel_username}' не похоже на канал / группу.")
            return

        # Получаем сообщения (телеграм objects)
        messages = []
        async for msg in self.client.iter_messages(entity, limit=limit):
            # msg.date, msg.sender_id, msg.text, msg.media ...
            messages.append(msg)

        # Сохраняем в CSV
        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["id", "date", "sender_id", "message", "views", "forwards", "reply_count"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for m in messages:
                row = {
                    "id": m.id,
                    "date": m.date,
                    "sender_id": m.sender_id,
                    "message": m.message if m.message else "",
                    "views": m.views if m.views else "",
                    "forwards": m.forwards if m.forwards else "",
                    "reply_count": m.replies.replies if m.replies else ""
                }
                writer.writerow(row)

        print(f"[INFO] Сохранено {len(messages)} сообщений в {csv_filename} (из канала {channel_username}).")

    async def disconnect(self):
        """Отключаемся от сервера Telegram."""
        if self.client:
            await self.client.disconnect()
            print("[INFO] Отключено от Telegram.")

# -------------------- Пример использования --------------------
async def main():
    # Вставьте свои API_ID, API_HASH
    api_id = 123456
    api_hash = "YOUR_API_HASH"
    phone_number = "+10000000000"  # при необходимости

    parser = TelegramParser(api_id, api_hash, phone=phone_number, session_name="my_session")
    await parser.connect()

    # Допустим, парсим публичный канал "CryptoNews" (пример)
    await parser.parse_channel("@cryptonews", limit=30, csv_filename="crypto_news.csv")

    # Отключаемся
    await parser.disconnect()

if __name__ == "__main__":
    # Запуск асинхронной функции
    asyncio.run(main())