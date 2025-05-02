import pandas as pd
import datetime as dt
import websockets
import asyncio
import json
from databases.pg_db import PostgresClient


class BinanceParser:
    def __init__(self, aggregation_level="1s"):
        self.data = dict()
        self.db = None
        self.aggregation_level = aggregation_level
        self.aggregated_data = {}
        self.base_url = "wss://stream.binance.com:443/stream?streams="
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT",
            "MATICUSDT", "LTCUSDT", "TRXUSDT", "LINKUSDT", "UNIUSDT", "BCHUSDT", "ETCUSDT", "APTUSDT", "ARBUSDT",
            "OPUSDT", "ATOMUSDT", "SANDUSDT", "APEUSDT", "FILUSDT", "NEARUSDT", "XLMUSDT", "INJUSDT", "AAVEUSDT",
        ]

    async def get_data_websocket(self, save_db: bool):
        connections = [self.connect(symbol.lower(), save_db=save_db) for symbol in self.symbols]
        connections.append(self.print_aggregated_data(interval_sec=1))
        await asyncio.gather(*connections)

    def run_websocket(self, db_client=None, save_db: bool = False):
        if save_db:
            self.db = db_client
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.get_data_websocket(save_db=save_db))

    def get_aggregation_timestamp(self, timestamp):
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
        agg_time = self.get_aggregation_timestamp(timestamp).timestamp()
        key = (symbol, agg_time)
        if key not in self.aggregated_data:
            self.aggregated_data[key] = {"price_sum": 0, "volume_sum": 0, "count": 0}
        agg = self.aggregated_data[key]
        agg["price_sum"] += price * volume
        agg["volume_sum"] += volume
        agg["count"] += 1

    async def connect(self, symbol, save_db: bool):
        url = self.base_url + f"{symbol}@trade"
        async with websockets.connect(url) as ws:
            while True:
                result = json.loads(await ws.recv())["data"]
                if "e" in result and result["e"] == "trade":
                    symbol = result["s"]
                    price = float(result["p"])
                    volume = float(result["q"])
                    timestamp = result["T"] / 1000  # to seconds

                    self.aggregate_trade(symbol, price, volume, timestamp)

    async def print_aggregated_data(self, interval_sec=1):
        while True:
            await asyncio.sleep(interval_sec)
            now = dt.datetime.utcnow().timestamp()
            for key in list(self.aggregated_data.keys()):
                symbol, ts = key
                if ts < now - interval_sec:
                    agg = self.aggregated_data.pop(key)
                    avg_price = agg["price_sum"] / agg["volume_sum"] if agg["volume_sum"] else 0
                    print(
                        f"[{dt.datetime.fromtimestamp(ts)}] {symbol} | AvgPrice: {avg_price:.2f} | Volume: {agg['volume_sum']:.2f} | Trades: {agg['count']}")


if __name__ == "__main__":

    parser = BinanceParser(aggregation_level="1s")

    db_name = 'postgres'
    user = 'postgres'
    password = '13zx2002xz'
    host = 'localhost'
    port = '5432'
    db_client = PostgresClient(db_name=db_name, user=user, password=password, host=host, port=port)

    parser.run_websocket(save_db=True, db_client=db_client)