import asyncio
import websockets
import json
import sqlite3
from datetime import datetime
from collections import defaultdict

# Список символов
symbols = ['btcusdt', 'ethusdt', 'bnbusdt']
channels = [f"{sym}@trade" for sym in symbols]
url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(channels)}"

# Подключение к SQLite
conn = sqlite3.connect("crypto_ticks.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS ticks (
    timestamp TEXT,
    symbol TEXT,
    avg_price REAL,
    total_volume REAL
)
""")
conn.commit()

# Временное хранилище данных
aggregated_data = defaultdict(lambda: defaultdict(lambda: {'price_sum': 0, 'volume_sum': 0, 'count': 0}))

# Обработка WebSocket
async def handle_ws():
    async with websockets.connect(url) as ws:
        async for msg in ws:
            try:
                data = json.loads(msg)["data"]
                symbol = data["s"].lower()
                price = float(data["p"])
                volume = float(data["q"])
                timestamp = datetime.fromtimestamp(data["T"] / 1000).replace(microsecond=0).isoformat()

                agg = aggregated_data[symbol][timestamp]
                agg["price_sum"] += price
                agg["volume_sum"] = agg.get("volume_sum", 0) + volume
                agg["count"] += 1
            except Exception as e:
                print("Ошибка при разборе сообщения:", e)

# Сохранение в БД каждую секунду
async def save_loop():
    while True:
        await asyncio.sleep(1)
        now = datetime.utcnow().replace(microsecond=0).isoformat()
        for symbol, time_dict in aggregated_data.items():
            if now in time_dict:
                entry = time_dict.pop(now)
                avg_price = entry["price_sum"] / entry["count"]
                total_volume = entry["volume_sum"]
                cursor.execute("INSERT INTO ticks (timestamp, symbol, avg_price, total_volume) VALUES (?, ?, ?, ?)",
                               (now, symbol, avg_price, total_volume))
                conn.commit()
                print(f"[{now}] {symbol.upper()} → Avg Price: {avg_price:.2f}, Volume: {total_volume:.4f}")

# Запуск
async def main():
    await asyncio.gather(
        handle_ws(),
        save_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Завершение работы...")
        conn.close()
