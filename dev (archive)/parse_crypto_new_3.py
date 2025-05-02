import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch

# --- Настройки ---
BATCH_SIZE = 16
TEXT_TRUNCATE_CHARS = 512
KEYWORDS = ["bitcoin", "crypto", "blockchain", "ethereum", "BTC", "altcoin"]

# Более лёгкая и быстрая zero-shot модель
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    tokenizer="facebook/bart-large-mnli",
    use_fast=False,  # ⬅️ ОТКЛЮЧАЕМ Fast Tokenizer
    device=0 if torch.cuda.is_available() else -1
)

candidate_labels = ["cryptocurrency", "finance", "politics", "technology"]


# --- Функции ---

def contains_crypto_keywords(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in KEYWORDS)


def classify_batch(texts):
    # обрезаем и фильтруем пустые
    inputs = [t[:TEXT_TRUNCATE_CHARS] for t in texts]
    results = classifier(inputs, candidate_labels, truncation=True)

    # pipeline возвращает либо dict, либо list[dict] — надо обработать оба варианта
    if isinstance(results, dict):
        results = [results]

    return [res['labels'][0] == 'cryptocurrency' for res in results]


def filter_crypto_news(csv_file_path):
    df = pd.read_csv(csv_file_path)

    if 'title' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV должен содержать колонки 'title' и 'text'.")

    tqdm.pandas(desc="Фильтрация по ключевым словам")
    df['keyword_match'] = df.progress_apply(
        lambda row: contains_crypto_keywords(str(row['title']) + " " + str(row['text'])),
        axis=1
    )

    df = df[df['keyword_match']].reset_index(drop=True)

    all_texts = (df['title'] + " " + df['text']).tolist()
    is_crypto_flags = []

    print("Запуск zero-shot классификации...")

    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Классификация"):
        batch = all_texts[i:i + BATCH_SIZE]
        batch_result = classify_batch(batch)
        is_crypto_flags.extend(batch_result)

    df = df.iloc[:len(is_crypto_flags)]
    df['is_crypto'] = is_crypto_flags

    return df[df['is_crypto']][['title', 'text']]


# --- Основной блок ---

if __name__ == "__main__":
    import torch
    import time

    time_start = time.time()
    path_to_csv = "crypnews247.csv"

    crypto_news_df = filter_crypto_news(path_to_csv)

    print(f"\n⏱️ Выполнено за {round(time.time() - time_start, 2)} сек.")
    print(f"📰 Найдено крипто-новостей: {len(crypto_news_df)}")
    print(crypto_news_df.head())

    crypto_news_df.to_csv("crypto_news_semantic.csv", index=False, encoding="utf-8")
