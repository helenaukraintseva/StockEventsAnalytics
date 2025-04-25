import asyncio
import websockets
import json
from datetime import datetime
from collections import defaultdict

# Хранилище агрегации
aggregated_data = defaultdict(lambda: {'price_sum': 0, 'volume_sum': 0, 'count': 0})

# Символ и URL Binance WebSocket
symbol = "btcusdt"
url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"

# Обработчик входящих сообщений
async def handle_trades():
    async with websockets.connect(url) as ws:
        async for message in ws:
            data = json.loads(message)
            trade_time = datetime.fromtimestamp(data['T'] / 1000)
            price = float(data['p'])
            volume = float(data['q'])

            # Округление до секунды
            timestamp = trade_time.replace(microsecond=0).isoformat()

            # Агрегируем цену и объем
            entry = aggregated_data[timestamp]
            entry['price_sum'] += price
            entry['volume_sum'] += volume
            entry['count'] += 1

# Периодический вывод агрегации
async def print_aggregated_data():
    while True:
        await asyncio.sleep(1)
        now = datetime.utcnow().replace(microsecond=0).isoformat()
        if now in aggregated_data:
            entry = aggregated_data[now]
            avg_price = entry['price_sum'] / entry['count'] if entry['count'] else 0
            print(f"[{now}] Avg Price: {avg_price:.2f}, Total Volume: {entry['volume_sum']:.4f}")
        else:
            print(f"[{now}] No data")

# Запуск двух задач параллельно
async def main():
    await asyncio.gather(
        handle_trades(),
        print_aggregated_data()
    )

if __name__ == "__main__":
    asyncio.run(main())
