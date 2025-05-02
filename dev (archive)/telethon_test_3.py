from telethon.sync import TelegramClient
from datetime import datetime, timedelta, timezone
import time
import os
import csv

# Вставьте свои данные
api_id = 1234567
api_hash = 'abcdef1234567890abcdef1234567890'
phone = '+10001234567'  # Ваш номер телефона в формате +1234567

# Задаем желаемые даты начала и конца (пример)
# start_date = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
start_date = datetime.now(timezone.utc)
delta_date = timedelta(days=1)
# end_date = datetime.datetime(2025, 5, 30, tzinfo=datetime.timezone.utc)
end_date = start_date - delta_date
print(start_date)
print(end_date)

# Юзернейм канала (например, @durov)
# channel_username = '@defi_cryptonews'
channels = ["https://t.me/if_market_news",
            "https://t.me/ru_holder",
            "https://t.me/WhattoNews",
            "https://t.me/web3news",
            "https://t.me/crypnews247",
            "https://t.me/slezisatoshi",
            "https://t.me/cryptooru",
            "https://t.me/skamshot",
            "https://t.me/cryptodaily",
            "https://t.me/chetamcrypto",
            "https://t.me/holder_pump_alert",
            "https://t.me/topslivs",
            "https://t.me/monkeytraderclub",
            "https://t.me/Cr_ideas",
            "https://t.me/cryptodurkaofficial",
            "https://t.me/kriptosanya",
            "https://t.me/CryptoAmbitions",
            "https://t.me/Mamkin_Treyd",
            "https://t.me/bitsmedia",
            "https://t.me/proscalping_trade",
            "https://t.me/blockchaingerman",
            "https://t.me/ico_btc",
            "https://t.me/HareCrypta",
            "https://t.me/ICOmuzhik",
            "https://t.me/coingoing",
            "https://t.me/crypto_hike",
            "https://t.me/sttrade88",
            "https://t.me/investkingyru",
            "https://t.me/saverscrypto",
            "https://t.me/ICOmuzhik",
            "https://t.me/web3easy",
            "https://t.me/CryptoLamer",
            "https://t.me/idoresearch",
            "https://t.me/cryptolevan",
            "https://t.me/criptopatolog",
            "https://t.me/EthereumIC",
            "https://t.me/proton_miners",
            "https://t.me/miningcluster",
            "https://t.me/ArnoldCrypto",
]
channels = [
    "https://t.me/if_market_news"
]
channels = [elem.split("/")[-1] for elem in channels]
# Имя файла для сохранения

client = TelegramClient('session_name', api_id, api_hash)
client.start(phone=phone)
print("Успешная авторизация!")


def main(channel, output_csv='channels_content_2024_1_1/'):
    output_csv = output_csv + channel + ".csv"
    # if os.path.exists(output_csv):
    #     print(f"File {output_csv} is ALREADY done.")
    #     return
    start_time = time.time()
    all_messages = []
    offset_id = 0  # с какого ID сообщения начинаем (0 = с последнего свежего)
    limit = 100  # размер одной порции. Если нужно быстрее (но дольше ждать) - можно увеличить.
    counter = 0
    while True:
        print(counter)
        counter += 1
        # Получаем очередную порцию сообщений (от самых свежих к более старым)
        time.sleep(5)
        messages = client.get_messages(f"@{channel}", limit=limit, offset_id=offset_id)
        if not messages:
            # Если сообщений не осталось
            break

        # Перебираем сообщения и фильтруем по дате
        for msg in messages:
            # Если дата сообщения < start_date, значит мы ушли слишком «глубоко» (слишком старые сообщения)
            if msg.date < start_date:
                # Можем прервать цикл целиком
                break

            # Если сообщение <= end_date и >= start_date, добавляем в список
            if start_date <= msg.date <= end_date:
                all_messages.append(msg)

        # Обновляем offset_id для следующей порции (переходим к более старым сообщениям)
        offset_id = messages[-1].id

        # Если последнее сообщение в порции уже старше start_date, нет смысла продолжать
        if messages[-1].date < start_date:
            break

    # Сортируем найденные сообщения по возрастанию даты (не обязательно, но часто удобнее)
    all_messages.sort(key=lambda m: m.date)

    # Сохраняем результаты в CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['message_id', 'date', 'text'])
        for msg in all_messages:
            writer.writerow([msg.id, msg.date, msg.message or ''])

    print(f"Собрано сообщений в диапазоне {start_date} - {end_date}: {len(all_messages)}")
    print(f"Данные сохранены в {output_csv}")
    print(f"Время на заугрузку: {round(time.time() - start_time, 2)} секунд.")


if __name__ == "__main__":
    with client:
        for channel in channels:
            try:
                main(channel=channel)
            except Exception as ex:
                print(f"Some error: {ex}")
