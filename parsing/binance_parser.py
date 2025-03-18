from config import BIN_APIKEY
import pandas as pd
import datetime as dt
from binance.client import Client
import websockets
import asyncio
from settings import futures_symbols, base_url
from config import BIN_APIKEY, BIN_SECRETKEY, DB_PAR_1
import json
from databases.pg_db import PostgresClient
from settings import bin_tokens_corr
import os


class BinanceParser:
    def __init__(self, secret_key: str, api_key: str):
        self.client = Client(api_key, secret_key)
        self.data = dict()
        self.db = None

    def check_file_exists(self, directory, filename):
        # Полный путь к файлу
        file_path = os.path.join(directory, filename)

        # Проверяем, существует ли файл
        if os.path.isfile(file_path):
            return True
        else:
            return False

    def get_data(self, symbols: list,
                 start_time: str,
                 end_time: str,
                 interval: str
                 ):
        data = list()
        for symbol in symbols:
            line = self.client.futures_historical_klines(symbol=symbol,
                                                         interval=interval,
                                                         start_str=start_time,
                                                         end_str=end_time)

            for elem in line:
                elem.extend([symbol])
            data.extend(line)
        return data

    def get_dataset_1(self,
                      symbols: list,
                      start_time: str,
                      interval: str,
                      end_time: str = None,
                      delta_time: str = None,
                      ):
        if end_time is None:
            end_time = start_time + delta_time
        counter = 0
        for symbol in symbols:
            filename = f"crypto_data/{symbol}_{start_time.split()[0]}_{interval}.csv"
            counter += 1
            print(f"{counter}/{len(symbols)}")
            if self.check_file_exists(directory="crypto_data", filename=filename.split("/")[1]):
                continue
            else:
                line = self.client.futures_historical_klines(symbol=symbol,
                                                             interval=interval,
                                                             start_str=start_time,
                                                             end_str=end_time)
                self.to_csv_file(filename=filename, data=line)

    def to_csv_file(self, filename: str, data: list):
        new_data = {
            "symbol": list(),
            'open_time': list(),
            'open': list(),
            'high': list(),
            'low': list(),
            'close': list(),
            'volume': list(),
            'close_time': list(),
            'qav': list(),
            'num_trades': list(),
            'taker_base_vol': list(),
            'taker_quote_vol': list(),
            'ignore': list()
        }

        for elem in data:
            new_data["symbol"].append(elem[-1])
            new_data["open_time"].append(elem[0])
            new_data["open"].append(elem[1])
            new_data["high"].append(elem[2])
            new_data["low"].append(elem[3])
            new_data["close"].append(elem[4])
            new_data["volume"].append(elem[5])
            new_data["close_time"].append(elem[6])
            new_data["qav"].append(elem[7])
            new_data["num_trades"].append(elem[8])
            new_data["taker_base_vol"].append(elem[9])
            new_data["taker_quote_vol"].append(elem[10])
            new_data["ignore"].append(elem[11])

        df = pd.DataFrame(new_data)
        df.to_csv(filename, index=False)
        print(f"File {filename} is done!")

    def get_list_token_1(self):
        result = self.client.futures_exchange_info().get("symbols", [])
        tokens = [token["symbol"] for token in result if token["status"] == "TRADING"]
        return tokens

    def get_data_model(self, symbol: str, interval: str, keys: list, count: int = 40):
        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(minutes=count*10)
        data = self.get_data(symbols=[symbol],
                             start_time=str(start_time),
                             end_time=str(end_time),
                             interval=interval)
        new_data = list()
        for ii in range(-count-1, -1):
            new_data.append(float(data[ii][4]))
        return [new_data]

    def get_data_graphic(self, symbol: str, interval: str):
        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=1)
        data = self.get_data(symbols=[symbol],
                             start_time=str(start_time),
                             end_time=str(end_time),
                             interval=interval)
        new_data = list()
        for ii in range(len(data)):
            new_data.append(float(data[ii][4]))
        return new_data

    def get_csv(self,
                filename: str,
                symbols: list,
                start_time: str,
                end_time: str,
                interval: str = "1m",
                ):

        data = self.get_data(symbols=symbols,
                             start_time=start_time,
                             end_time=end_time,
                             interval=interval)
        new_data = {
            "symbol": list(),
            'open_time': list(),
            'open': list(),
            'high': list(),
            'low': list(),
            'close': list(),
            'volume': list(),
            'close_time': list(),
            'qav': list(),
            'num_trades': list(),
            'taker_base_vol': list(),
            'taker_quote_vol': list(),
            'ignore': list()
        }

        for elem in data:
            new_data["symbol"].append(elem[-1])
            new_data["open_time"].append(elem[0])
            new_data["open"].append(elem[1])
            new_data["high"].append(elem[2])
            new_data["low"].append(elem[3])
            new_data["close"].append(elem[4])
            new_data["volume"].append(elem[5])
            new_data["close_time"].append(elem[6])
            new_data["qav"].append(elem[7])
            new_data["num_trades"].append(elem[8])
            new_data["taker_base_vol"].append(elem[9])
            new_data["taker_quote_vol"].append(elem[10])
            new_data["ignore"].append(elem[11])

        df = pd.DataFrame(new_data)
        df.to_csv(filename, index=False)
        print("File is done!")

    async def get_data_websocket(self, save_db: bool):
        connections = [self.connect(symbol.lower(), save_db=save_db) for symbol in futures_symbols]
        await asyncio.gather(*connections)

    async def connect(self, symbol, save_db: bool):
        url = base_url + f"{symbol}@trade"
        # Тайминг, по которому будут передаваться данные в систему, timing = 1 - каждую секунду
        timing = 1
        async with websockets.connect(url) as ws:
            index_trade = 0
            # actual_time = -1
            # actual_volume = 0
            while True:
                # try:
                # Получение данных с Вебсокета
                result = json.loads(await ws.recv())["data"]
                if "e" in result and result["e"] == "trade":
                    symbol = result["s"]
                    trade_id = result["t"]
                    price = float(result["p"])
                    volume = float(result["q"])
                    timestamp = result["T"] // (1000 * timing)
                    # Accumulate data for filters
                    print(f"Symbol: {symbol}\nPrice: {price}, volume: {volume}, timestamp: {timestamp}")
                # if save_db:
                #     db_client.set_data(request=f"INSERT INTO websocket_data (symbol, price, volume, timestamp) "
                #                                f"VALUES ('{symbol}', '{price}', '{volume}', '{timestamp}');")

    def run_websocket(self, db_client=None, save_db: bool = False):
        if save_db:
            self.db = db_client
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.get_data_websocket(save_db=save_db))


# db_client = PostgresClient(db_name=DB_PAR_1["db_name"],
#                            host=DB_PAR_1["host"],
#                            port=DB_PAR_1["port"],
#                            user=DB_PAR_1["user"],
#                            password=DB_PAR_1["password"]
#                            )

def cycle_parsing(bin_client, interval: list, delta_time: list):
    for ii in range(len(interval)):
        start_time = str(dt.datetime.now() - delta_time[ii])
        end_time = str(dt.datetime.now())
        tokens = bin_tokens_corr[:]
        bin_client.get_dataset_1(
            symbols=tokens,
            start_time=start_time,
            end_time=end_time,
            interval=interval[ii]
        )


if __name__ == "__main__":

    parser = BinanceParser(secret_key=BIN_SECRETKEY, api_key=BIN_APIKEY)

    interval = ["1m", "5m", "30m", "1h"]
    delta_time = [dt.timedelta(days=20),
                  dt.timedelta(days=100),
                  dt.timedelta(days=500),
                  dt.timedelta(days=1000)]
    cycle_parsing(bin_client=parser, interval=interval, delta_time=delta_time)
    #

    # print(parser.get_list_token_1())
    # parser.get_csv(filename="dataset_1.csv",
    #                symbols=["BTCUSDT"],
    #                start_time="2022-01-01 00:00:00",
    #                end_time="2022-02-01 00:00:00",
    #                interval='1m')

    # print(parser.get_data_model(symbol="BTCUSDT", interval="1m", keys=["close"]))
    # print(parser.get_data_graphic(symbol="BTCUSDT", interval="1m"))

    # print(end_time)

    # parser.run_websocket(save_db=True, db_client=db_client)

    # print(db_client.get_data("SELECT * FROM websocket_data;"))

