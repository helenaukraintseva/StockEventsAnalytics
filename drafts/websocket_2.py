import asyncio
import websockets
import json
import sqlite3
from datetime import datetime
from collections import defaultdict

# Список символов
symbols = ['btcusdt', 'ethusdt', 'bnbusdt']
channels = [f"{sym}@trade" for sym in symbols]

# WebSocket URL с несколькими потоками
url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(channels)}"

# Подключение к SQLite
conn = sqlite3.connect("crypto_aggregates.db")
cursor = conn.cursor()

# Создание таблицы
cursor.execute("""
CREATE TABLE IF NOT EXISTS aggregated_data (
    timestamp TEXT,
    symbol TEXT,
    avg_price REAL,
    total_volume REAL
)
""")
conn.commit()

# Агрегационное хранилище: {symbol -> {timestamp -> {...}}}
aggregated_data = defaultdict(lambda: defaultdict(lambda: {'price_sum': 0, 'volume_sum': 0, 'count': 0, 'volume_sum': 0}))

# Обработчик входящих сообщений
async def handle_trades():
    async with websockets.connect(url) as ws:
        async for message in ws:
            data = json.loads(message)
            trade = data['data']
            symbol = trade['s'].lower()
            price = float(trade['p'])
            volume = float(trade['q'])
            trade_time = datetime.fromtimestamp(trade['T'] / 1000).replace(microsecond=0).isoformat()

            entry = aggregated_data[symbol][trade_time]
            entry['price_sum'] += price
            entry['volume_sum'] += volume
            entry['count'] += 1

# Сохраняет агрегацию в базу данных каждую секунду
async def save_aggregated_data():
    while True:
        await asyncio.sleep(1)
        current_time = datetime.utcnow().replace(microsecond=0).isoformat()

        for symbol, data_per_symbol in aggregated_data.items():
            if current_time in data_per_symbol:
                entry = data_per_symbol.pop(current_time)
                avg_price = entry['price_sum'] / entry['count'] if entry['count'] else 0
                total_volume = entry['volume_sum']
                cursor.execute(
                    "INSERT INTO aggregated_data (timestamp, symbol, avg_price, total_volume) VALUES (?, ?, ?, ?)",
                    (current_time, symbol, avg_price, total_volume)
                )
                conn.commit()
                print(f"[{current_time}] {symbol.upper()} → Avg Price: {avg_price:.2f}, Volume: {total_volume:.4f}")
            else:
                print(f"[{current_time}] {symbol.upper()} → No data")

# Основная функция
async def main():
    await asyncio.gather(
        handle_trades(),
        save_aggregated_data()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Остановлено пользователем")
        conn.close()
