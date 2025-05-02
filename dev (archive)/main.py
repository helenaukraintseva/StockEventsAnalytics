import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
import datetime as dt
from parsing.binance_parser import BinanceParser
from config import TELEG_API, BIN_APIKEY, BIN_SECRETKEY

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Вставьте сюда ваш токен
API_TOKEN = TELEG_API

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

@dp.message_handler(commands=["start"])
async def send_msg(msg: types.Message):
    await msg.answer("Напиши /sendcsv, чтобы запустить парсер статистики")


# Функция для обработки команды /sendcsv
@dp.message_handler(commands=['sendcsv'])
async def send_csv(message: types.Message):
    start_time = str(dt.datetime.now() - dt.timedelta(hours=24))
    end_time = str(dt.datetime.now())
    parser = BinanceParser(secret_key=BIN_SECRETKEY, api_key=BIN_APIKEY)
    symbols = ["BTCUSDT", "ETHUSDT", "XTZUSDT", "XRPUSDT", "XMRUSDT", "XLMUSDT"]

    file_path = 'data.csv'  # Укажите путь к вашему .csv файлу
    parser.get_csv("data.csv", symbols=symbols, start_time=start_time, end_time=end_time, interval="1h")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            await message.answer(f"Сбор данных по следующим токанам:\n{', '.join(symbols)}")
            await message.answer_document(file)

    else:
        await message.reply("Файл не найден.")


@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def echo_message(message: types.Message):
    start_time = str(dt.datetime.now() - dt.timedelta(hours=24))
    end_time = str(dt.datetime.now())
    parser = BinanceParser(secret_key=BIN_SECRETKEY, api_key=BIN_APIKEY)
    file_path = f'{message.text}.csv'  # Укажите путь к вашему .csv файлу
    symbols = [message.text]
    try:
        parser.get_csv(f"{message.text}.csv", symbols=symbols, start_time=start_time, end_time=end_time, interval="1h")
        await message.answer("Сбор данных завершен.")
        with open(file_path, 'rb') as file:
            await message.answer_document(file)
    except Exception:
        await message.answer("Про такой токен нет информации.")



# Основная функция для запуска бота
if __name__ == '__main__':
    print("ok")
    executor.start_polling(dp, skip_updates=True)

