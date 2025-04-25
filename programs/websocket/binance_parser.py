import os
import json
import asyncio
import logging
import websockets
from dotenv import load_dotenv
from databases.pg_db import PostgresClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BinanceParser:
    def __init__(self):
        """
        Парсер Binance WebSocket.
        """
        self.db = None
        self.base_url = "wss://stream.binance.com:443/stream?streams="
        self.symbols = os.getenv("BINANCE_SYMBOLS", "ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,BNBUSDT,LINKUSDT,BTCUSDT").split(",")

    async def get_data_websocket(self, save_db: bool):
        """
        Запускает параллельные подключения по всем символам.

        :param save_db: Флаг, сохранять ли в базу данных
        :return: None
        """
        connections = [self.connect(symbol.lower(), save_db=save_db) for symbol in self.symbols]
        await asyncio.gather(*connections)

    async def connect(self, symbol: str, save_db: bool):
        """
        Подключение к одному торговому потоку Binance.

        :param symbol: Символ, например btcusdt
        :param save_db: Флаг записи в БД
        :return: None
        """
        url = self.base_url + f"{symbol}@trade"
        timing = 1  # интервальность (в секундах)

        async with websockets.connect(url) as ws:
            logging.info("📡 Подключено к: %s", url)
            while True:
                result = json.loads(await ws.recv())["data"]

                if result.get("e") == "trade":
                    trade_id = result["t"]
                    price = float(result["p"])
                    volume = float(result["q"])
                    timestamp = result["T"] // (1000 * timing)

                    logging.info(f"📈 {symbol.upper()} | Цена: {price} | Объём: {volume} | Время: {timestamp}")

                    if save_db and self.db:
                        query = (
                            f"INSERT INTO websocket_data (symbol, price, volume, timestamp) "
                            f"VALUES ('{symbol.upper()}', {price}, {volume}, {timestamp});"
                        )
                        self.db.set_data(query)

    def run_websocket(self, db_client=None, save_db: bool = False):
        """
        Запускает WebSocket-парсинг.

        :param db_client: Объект БД (PostgresClient)
        :param save_db: Сохранять ли данные в БД
        :return: None
        """
        if save_db:
            self.db = db_client
        asyncio.run(self.get_data_websocket(save_db=save_db))


if __name__ == "__main__":
    db_client = PostgresClient(
        db_name=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "your_password"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )

    parser = BinanceParser()
    parser.run_websocket(db_client=db_client, save_db=True)
