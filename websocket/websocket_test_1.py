import os
import json
import asyncio
import datetime as dt
import websockets
import logging
from dotenv import load_dotenv
from databases.pg_db import PostgresClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BinanceParser:
    def __init__(self, aggregation_level="1s"):
        """
        Инициализация класса парсера.

        :param aggregation_level: уровень агрегации ("1s", "1m", "5m")
        """
        self.db = None
        self.aggregation_level = aggregation_level
        self.aggregated_data = {}
        self.base_url = "wss://stream.binance.com:443/stream?streams="
        self.symbols = os.getenv("BINANCE_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")

    async def get_data_websocket(self, save_db: bool):
        """
        Основной метод запуска всех подключений.

        :param save_db: сохранять ли данные в БД
        """
        connections = [self.connect(symbol.lower(), save_db=save_db) for symbol in self.symbols]
        connections.append(self.print_aggregated_data(interval_sec=1))
        await asyncio.gather(*connections)

    def run_websocket(self, db_client=None, save_db: bool = False):
        """
        Запуск WebSocket клиента.

        :param db_client: клиент PostgreSQL
        :param save_db: сохранять ли данные в БД
        """
        if save_db:
            self.db = db_client
        asyncio.run(self.get_data_websocket(save_db=save_db))

    def get_aggregation_timestamp(self, timestamp):
        """
        Вычисляет округлённое время по уровню агрегации.

        :param timestamp: UNIX timestamp
        :return: datetime объект округленного времени
        """
        dt_obj = dt.datetime.fromtimestamp(timestamp)
        if self.aggregation_level == "1s":
            return dt_obj.replace(microsecond=0)
        elif self.aggregation_level == "1m":
            return dt_obj.replace(second=0, microsecond=0)
        elif self.aggregation_level == "5m":
            minute = (dt_obj.minute // 5) * 5
            return dt_obj.replace(minute=minute, second=0, microsecond=0)
        else:
            raise ValueError("Unsupported aggregation level")

    def aggregate_trade(self, symbol, price, volume, timestamp):
        """
        Агрегирует полученные трейды.

        :param symbol: торговый инструмент
        :param price: цена
        :param volume: объём
        :param timestamp: время сделки (в секундах)
        """
        agg_time = self.get_aggregation_timestamp(timestamp).timestamp()
        key = (symbol, agg_time)
        if key not in self.aggregated_data:
            self.aggregated_data[key] = {"price_sum": 0, "volume_sum": 0, "count": 0}
        agg = self.aggregated_data[key]
        agg["price_sum"] += price * volume
        agg["volume_sum"] += volume
        agg["count"] += 1

    async def connect(self, symbol, save_db: bool):
        """
        Подключение к WebSocket и обработка трейдов.

        :param symbol: торговая пара
        :param save_db: сохранять ли в БД
        """
        url = self.base_url + f"{symbol}@trade"
        async with websockets.connect(url) as ws:
            while True:
                result = json.loads(await ws.recv())["data"]
                if result.get("e") == "trade":
                    symbol = result["s"]
                    price = float(result["p"])
                    volume = float(result["q"])
                    timestamp = result["T"] / 1000
                    self.aggregate_trade(symbol, price, volume, timestamp)

    async def print_aggregated_data(self, interval_sec=1):
        """
        Печатает агрегированные данные каждую секунду.

        :param interval_sec: интервал печати (в секундах)
        """
        while True:
            await asyncio.sleep(interval_sec)
            now = dt.datetime.utcnow().timestamp()
            for key in list(self.aggregated_data.keys()):
                symbol, ts = key
                if ts < now - interval_sec:
                    agg = self.aggregated_data.pop(key)
                    avg_price = agg["price_sum"] / agg["volume_sum"] if agg["volume_sum"] else 0
                    logging.info(f"[{dt.datetime.fromtimestamp(ts)}] {symbol} | AvgPrice: {avg_price:.2f} | Volume: {agg['volume_sum']:.2f} | Trades: {agg['count']}")


if __name__ == "__main__":
    db_client = PostgresClient(
        db_name=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "your_password"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    parser = BinanceParser(aggregation_level="1s")
    parser.run_websocket(db_client=db_client, save_db=True)
