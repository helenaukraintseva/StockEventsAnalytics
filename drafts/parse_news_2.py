import pandas as pd
from transformers import pipeline
import time

time_start = time.time()

# Загрузка модели zero-shot классификации
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Кандидатные темы
candidate_labels = ["cryptocurrency", "finance", "politics", "technology"]

def is_crypto_news(text):
    result = classifier(text, candidate_labels)
    top_label = result['labels'][0]
    return top_label == "cryptocurrency"

def filter_crypto_news(csv_file_path):
    df = pd.read_csv(csv_file_path)

    if 'title' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV должен содержать колонки 'title' и 'text'.")

    df['is_crypto'] = df.apply(
        lambda row: is_crypto_news(str(row['title']) + ' ' + str(row['text'])),
        axis=1
    )

    crypto_news = df[df['is_crypto']]
    return crypto_news[['title', 'text']]


if __name__ == "__main__":
    path_to_csv = "crypnews247.csv"
    crypto_news_df = filter_crypto_news(path_to_csv)
    print(f"Total time: {round(time.time() - time_start, 2)}")
    print("Найдено крипто-новостей:", len(crypto_news_df))
    print(crypto_news_df.head())
    crypto_news_df.to_csv("crypto_news_semantic.csv", index=False)
