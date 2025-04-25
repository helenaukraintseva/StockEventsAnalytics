import os
import time
import logging
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

time_start = time.time()

# Загрузка модели Zero-Shot
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["cryptocurrency", "finance", "politics", "technology"]


def is_crypto_news(text: str) -> bool:
    """
    Определяет, относится ли текст к тематике криптовалют.

    :param text: Заголовок и/или тело статьи
    :return: True, если это крипто-новость
    """
    result = classifier(text, candidate_labels)
    top_label = result['labels'][0]
    return top_label == "cryptocurrency"


def filter_crypto_news(csv_file_path: str) -> pd.DataFrame:
    """
    Фильтрует крипто-новости из CSV-файла на основе zero-shot-классификации.

    :param csv_file_path: Путь к CSV-файлу
    :return: DataFrame с крипто-новостями
    """
    df = pd.read_csv(csv_file_path)

    if 'title' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV должен содержать колонки 'title' и 'text'.")

    df['is_crypto'] = df.apply(
        lambda row: is_crypto_news(f"{row['title']} {row['text']}"),
        axis=1
    )

    return df[df['is_crypto']][['title', 'text']]


if __name__ == "__main__":
    input_path = os.getenv("NEWS_INPUT_CSV", "crypnews247.csv")
    output_path = os.getenv("NEWS_OUTPUT_CSV", "crypto_news_semantic.csv")

    crypto_news_df = filter_crypto_news(input_path)
    logging.info("Найдено крипто-новостей: %d", len(crypto_news_df))

    crypto_news_df.to_csv(output_path, index=False)
    logging.info("Сохранено в файл: %s", output_path)
    logging.info("Total time: %.2f сек.", time.time() - time_start)
