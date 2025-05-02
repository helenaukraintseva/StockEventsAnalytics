api_id = 25729614
api_hash = "702ddd928d7d562441a441f0f4df6280"
title = "parserbot"
phone = "+7 928 851 4993"

from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty
import csv
import asyncio

# ✅ Вставь свои данные
# api_id = 123456  # Твой API ID
# api_hash = 'abcdef1234567890abcdef1234567890'  # Твой API Hash
# phone = '+1234567890'  # Твой номер телефона

client = TelegramClient('session_name', api_id, api_hash)


async def main():
    await client.start(phone=phone)

    # Получаем список чатов
    chats = []
    last_date = None
    chunk_size = 200
    result = await client(GetDialogsRequest(
        offset_date=last_date,
        offset_id=0,
        offset_peer=InputPeerEmpty(),
        limit=chunk_size,
        hash=0
    ))
    chats.extend(result.chats)

    # Выбираем нужный чат по имени
    for i, chat in enumerate(chats):
        print(f"{i}: {chat.title}")

    chat_index = int(input("Введите номер чата для парсинга: "))
    target_chat = chats[chat_index]

    participants = await client.get_participants(target_chat)

    print(f"Найдено {len(participants)} участников.")

    # Сохраняем участников в CSV
    with open("members.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["username", "user id", "access hash", "name"])
        for user in participants:
            username = user.username if user.username else ""
            user_id = user.id
            access_hash = user.access_hash
            name = f"{user.first_name or ''} {user.last_name or ''}".strip()
            writer.writerow([username, user_id, access_hash, name])

    print("Готово! Данные сохранены в members.csv")


# Запуск
with client:
    client.loop.run_until_complete(main())
