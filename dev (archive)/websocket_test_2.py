import pandas as pd
import datetime as dt
import websockets
import asyncio
import json
from databases.pg_db import PostgresClient
from algorithms import AlgorithmMACD, AlgorithmRSI, AlgorithmIchimoku
import psycopg2


class BinanceParser:
    def __init__(self, aggregation_level="1s"):
        self.data = dict()
        self.db = None
        self.base_url = "wss://stream.binance.com:443/stream?streams="
        # self.base_url = f"wss://stream.binance.com:9443/ws/"
        self.symbols = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "BNBUSDT", "LINKUSDT", "BTCUSDT"]
        self.aggregation_level = aggregation_level
        self.data = {}
        self.models = {
            "MACD": AlgorithmMACD(),
            "RSI": AlgorithmRSI(window=14),
            "ICHIMOKU": AlgorithmIchimoku(),
        }
        self.signals = {
            "MACD": list(),
            "RSI": list(),
            "ICHIMOKU": list(),
        }

    def get_agg_timestamp(self, ts):
        dt_obj = dt.datetime.fromtimestamp(ts)
        if self.aggregation_level == "1s":
            return int(dt_obj.replace(microsecond=0).timestamp())
        elif self.aggregation_level == "1m":
            return int(dt_obj.replace(second=0, microsecond=0).timestamp())
        elif self.aggregation_level == "5m":
            minute = (dt_obj.minute // 5) * 5
            return int(dt_obj.replace(minute=minute, second=0, microsecond=0).timestamp())

    def aggregate_trade(self, symbol, price, volume, timestamp):
        dt_obj = dt.datetime.fromtimestamp(timestamp)
        agg_ts = int(dt_obj.replace(microsecond=0).timestamp())
        if symbol not in self.data:
            self.data[symbol] = {"open": list(), "high": list(), "low": list(), "close": list(),
                                 "open_mem": price, "high_mem": price, "low_mem": price,
                                 "volume": list(), "timestamp": list(), "total_volume": volume}
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
            # if agg_ts - 1 in self.data[symbol]["timestamp"]:
            #     print(f"{symbol} --- {self.data[symbol]}")
        else:
            self.data[symbol]["total_volume"] += volume
            if price > self.data[symbol]["high_mem"]:
                self.data[symbol]["high_mem"] = price
            if price < self.data[symbol]["low_mem"]:
                self.data[symbol]["low_mem"] = price
        self.model_signal()

    def model_signal(self):
        new_data = {}
        new_data["ETHUSDT"] = {}
        if "ETHUSDT" in self.data:
            new_data["ETHUSDT"]["open"] = self.data["ETHUSDT"]["open"].copy()
            new_data["ETHUSDT"]["high"] = self.data["ETHUSDT"]["high"].copy()
            new_data["ETHUSDT"]["low"] = self.data["ETHUSDT"]["low"].copy()
            new_data["ETHUSDT"]["close"] = self.data["ETHUSDT"]["close"].copy()
            new_data["ETHUSDT"]["volume"] = self.data["ETHUSDT"]["volume"].copy()
            new_data["ETHUSDT"]["timestamp"] = self.data["ETHUSDT"]["timestamp"].copy()
            df = pd.DataFrame(new_data["ETHUSDT"])
            timestamp = df["timestamp"].iloc[-1]
            price = df["close"].iloc[-1]
            if len(df) > 50:
                for model in self.models:
                    result = self.models[model].run(df)
                    if result['Signal'].iloc[-5:].nunique() == 1:
                        signal = result['Signal'].iloc[-1]
                        self.create_signal(model, signal, price, timestamp)
                    if result['Signal'].iloc[-5:].nunique() == -1:
                        signal = result['Signal'].iloc[-1]
                        self.create_signal(model, signal, price, timestamp)
                    else:
                        signal = 0

    def create_signal(self, model, signal, price, timestamp):
        if len(self.signals[model]) < 1:
            self.signals[model].append({"signal": signal, "price": price, "timestamp": timestamp})
        else:
            time_delta = (self.signals[model][0]["timestamp"] - timestamp) > 10
            if self.signals[model][0]["signal"] != signal and time_delta:
                if signal < 0:
                    delta = price - self.signals[model][0]["price"]
                else:
                    delta = self.signals[model][0]["price"] - price
                print(f"Model: {model}, Signal: {self.signals[model][0]['signal']},"
                      f" start_time: {self.signals[model][0]['timestamp']}, end time: {timestamp}\nDelta: {delta}, time delta: {time_delta}")
                del self.signals[model][0]

    def insert_signal(symbol, model, start_price, end_price, start_time, end_time):
        conn = psycopg2.connect(
            dbname="crypto_signals",
            user="crypto_user",
            password="crypto_pass",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        query = """
            INSERT INTO signals (symbol, model, start_price, end_price, start_time, end_time)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        cursor.execute(query, (symbol, model, start_price, end_price, start_time, end_time))
        conn.commit()
        cursor.close()
        conn.close()

    def fetch_recent_signals(symbol, model):
        conn = psycopg2.connect(
            dbname="crypto_signals",
            user="crypto_user",
            password="strong_password_here",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        query = """
            SELECT symbol, model, start_price, end_price, start_time, end_time
            FROM signals
            WHERE symbol = %s
              AND model = %s
              AND end_time IS NOT NULL
              AND end_time >= NOW() - INTERVAL '6 hours';
        """
        cursor.execute(query, (symbol, model))
        results = cursor.fetchall()

        cursor.close()
        conn.close()
        return results

    async def print_aggregated(self, interval_sec=1):
        while True:
            await asyncio.sleep(interval_sec)
            now = dt.datetime.utcnow().timestamp()
            cutoff = self.get_agg_timestamp(now - interval_sec)
            to_remove = []
            for (symbol, agg_ts), data in self.data.items():
                if agg_ts <= cutoff:
                    print(
                        f"[{dt.datetime.fromtimestamp(agg_ts)}] {symbol} | Price: {data['last_price']} | Volume: {data['total_volume']:.4f}")
                    to_remove.append((symbol, agg_ts))
            for key in to_remove:
                del self.data[key]

    async def get_data_websocket(self, save_db: bool):
        connections = [self.connect(symbol.lower(), save_db=save_db) for symbol in self.symbols]
        # connections.append(self.print_aggregated(interval_sec=1))  # каждую секунду печатает
        await asyncio.gather(*connections)

    async def connect(self, symbol, save_db: bool):
        url = self.base_url + f"{symbol}@trade"
        async with websockets.connect(url) as ws:
            while True:
                result = json.loads(await ws.recv())["data"]
                if result.get("e") == "trade":
                    symbol = result["s"]
                    price = float(result["p"])
                    volume = float(result["q"])
                    timestamp = result["T"] / 1000  # в секундах
                    self.aggregate_trade(symbol, price, volume, timestamp)

    def run_websocket(self, db_client=None, save_db: bool = False):
        if save_db:
            self.db = db_client
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.get_data_websocket(save_db=save_db))



if __name__ == "__main__":
    parser = BinanceParser()

    db_name = 'postgres'
    user = 'postgres'
    password = '13zx2002xz'
    host = 'localhost'
    port = '5432'
    db_client = PostgresClient(db_name=db_name, user=user, password=password, host=host, port=port)

    parser.run_websocket(save_db=True, db_client=db_client)
