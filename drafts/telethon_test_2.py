from telethon.sync import TelegramClient
import csv
import sys

# Вставьте свои данные
api_id = 1234567
api_hash = 'abcdef1234567890abcdef1234567890'
phone = '+10001234567'  # Ваш номер телефона в формате +1234567

# Создаем клиент. 'session_name' — имя файла сессии, где хранится авторизация
client = TelegramClient('session_name', api_id, api_hash)


def main():
    # Запуск клиента и авторизация
    client.start(phone=phone)

    # Если код запустили без проблем:
    print("Успешная авторизация!")

    # Запрашиваем у пользователя юзернейм канала (например, @testchannel)
    channel_username = input("Введите @username канала (например, @testchannel): ").strip()
    if not channel_username.startswith('@'):
        print("Неверный формат канала. Должно начинаться с @")
        sys.exit(1)

    # Считываем сообщения. limit=None — значит все доступные сообщения
    # Можно поставить limit=1000, если нужно ограничить парсинг тысячу последних сообщений.
    messages = client.get_messages(channel_username, limit=1000)

    print(f"Всего сообщений получено: {len(messages)}")

    # Сохраняем в CSV
    with open("channel_posts.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["message_id", "date", "text"])

        for msg in messages:
            msg_id = msg.id
            msg_date = msg.date
            # Некоторые сообщения могут быть пустыми или содержать только медиа
            msg_text = msg.message if msg.message else ""

            writer.writerow([msg_id, msg_date, msg_text])

    print("Данные успешно сохранены в 'channel_posts.csv'.")


if __name__ == "__main__":
    with client:
        main()
