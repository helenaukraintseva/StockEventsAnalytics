import os
import json
import asyncio
import logging
import datetime as dt
import pandas as pd
import websockets
import psycopg2
from dotenv import load_dotenv
from databases.pg_db import PostgresClient
from algorithms import AlgorithmMACD, AlgorithmRSI, AlgorithmIchimoku

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BinanceSignalProcessor:
    def __init__(self, aggregation_level: str = "1s"):
        """
        :param aggregation_level: Уровень агрегации времени ("1s", "1m", "5m")
        """
        self.base_url = "wss://stream.binance.com:443/stream?streams="
        self.symbols = os.getenv("BINANCE_SYMBOLS", "ETHUSDT,BTCUSDT").split(",")
        self.aggregation_level = aggregation_level
        self.db = None
        self.data = {}
        self.models = {
            "MACD": AlgorithmMACD(),
            "RSI": AlgorithmRSI(window=14),
            "ICHIMOKU": AlgorithmIchimoku()
        }
        self.signals = {model: [] for model in self.models}

    def get_agg_timestamp(self, ts: float) -> int:
        """
        :param ts: UNIX timestamp в секундах
        :return: округлённый timestamp по уровню агрегации
        """
        dt_obj = dt.datetime.fromtimestamp(ts)
        if self.aggregation_level == "1s":
            return int(dt_obj.replace(microsecond=0).timestamp())
        elif self.aggregation_level == "1m":
            return int(dt_obj.replace(second=0, microsecond=0).timestamp())
        elif self.aggregation_level == "5m":
            minute = (dt_obj.minute // 5) * 5
            return int(dt_obj.replace(minute=minute, second=0, microsecond=0).timestamp())

    def aggregate_trade(self, symbol: str, price: float, volume: float, timestamp: float):
        """
        :param symbol: Символ (например, BTCUSDT)
        :param price: Цена сделки
        :param volume: Объём сделки
        :param timestamp: Время сделки
        """
        agg_ts = self.get_agg_timestamp(timestamp)
        if symbol not in self.data:
            self.data[symbol] = {
                "open": [], "high": [], "low": [], "close": [],
                "volume": [], "timestamp": [],
                "open_mem": price, "high_mem": price, "low_mem": price,
                "total_volume": volume
            }

        if agg_ts not in self.data[symbol]["timestamp"]:
            self.data[symbol]["open"].append(self.data[symbol]["open_mem"])
            self.data[symbol]["high"].append(self.data[symbol]["high_mem"])
            self.data[symbol]["low"].append(self.data[symbol]["low_mem"])
            self.data[symbol]["close"].append(price)
            self.data[symbol]["volume"].append(self.data[symbol]["total_volume"])
            self.data[symbol]["timestamp"].append(agg_ts)
            self.data[symbol]["total_volume"] = volume
            self.data[symbol]["open_mem"] = price
            self.data[symbol]["high_mem"] = price
            self.data[symbol]["low_mem"] = price
        else:
            self.data[symbol]["total_volume"] += volume
            self.data[symbol]["high_mem"] = max(self.data[symbol]["high_mem"], price)
            self.data[symbol]["low_mem"] = min(self.data[symbol]["low_mem"], price)

        self.model_signal(symbol)

    def model_signal(self, symbol: str):
        """
        Вызывает сигнальные модели для символа
        :param symbol: Символ актива (например, ETHUSDT)
        """
        if symbol not in self.data or len(self.data[symbol]["close"]) < 50:
            return

        df = pd.DataFrame(self.data[symbol])
        price = df["close"].iloc[-1]
        timestamp = df["timestamp"].iloc[-1]

        for model_name, model in self.models.items():
            result = model.run(df)
            recent_signals = result['Signal'].iloc[-5:]
            if recent_signals.nunique() == 1:
                signal = recent_signals.iloc[-1]
                self.create_signal(model_name, signal, price, timestamp)

    def create_signal(self, model: str, signal: int, price: float, timestamp: int):
        """
        Сохраняет сигнал и при необходимости логирует его.

        :param model: Название модели
        :param signal: Тип сигнала (1 = buy, -1 = sell, 0 = hold)
        :param price: Цена сигнала
        :param timestamp: Время сигнала
        """
        history = self.signals[model]
        if not history:
            self.signals[model].append({"signal": signal, "price": price, "timestamp": timestamp})
            return

        previous = history[0]
        time_delta = timestamp - previous["timestamp"] > 10
        if previous["signal"] != signal and time_delta:
            pnl = price - previous["price"] if signal < 0 else previous["price"] - price
            logging.info(f"Model: {model} | Prev: {previous['signal']} @ {previous['price']} → {signal} @ {price} | Δ: {pnl:.4f}")
            del history[0]

    async def connect(self, symbol: str):
        """
        Подключается к Binance WebSocket и читает поток данных
        :param symbol: Символ для подключения
        """
        url = self.base_url + f"{symbol}@trade"
        async with websockets.connect(url) as ws:
            while True:
                result = json.loads(await ws.recv())["data"]
                if result.get("e") == "trade":
                    price = float(result["p"])
                    volume = float(result["q"])
                    timestamp = result["T"] / 1000
                    self.aggregate_trade(result["s"], price, volume, timestamp)

    async def get_data_websocket(self):
        """
        Запускает WebSocket подключения ко всем символам.
        """
        connections = [self.connect(symbol.lower()) for symbol in self.symbols]
        await asyncio.gather(*connections)

    def run(self):
        """
        Запуск основного event loop.
        """
        asyncio.run(self.get_data_websocket())


if __name__ == "__main__":
    parser = BinanceSignalProcessor(aggregation_level="1m")
    parser.run()