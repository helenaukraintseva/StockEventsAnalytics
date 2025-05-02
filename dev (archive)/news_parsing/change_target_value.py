import os
import logging
import pandas as pd
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Получение путей из переменных окружения
input_csv = os.getenv("INPUT_CSV", "input.csv")
output_csv = os.getenv("OUTPUT_CSV", "output.csv")

try:
    # Считываем данные
    df = pd.read_csv(input_csv)

    # Меняем значения в столбце target
    df["target"] = df["target"].map({-1: 1, 0: 2, 1: 3})

    # Сохраняем в новый файл
    df.to_csv(output_csv, index=False)

    logging.info("Файл '%s' успешно создан.", output_csv)
except Exception as e:
    logging.error("Ошибка при обработке файла: %s", e)
