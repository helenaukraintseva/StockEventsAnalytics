import pandas as pd

# Ключевые слова, связанные с криптовалютой
crypto_keywords = [
    "биткоин", "bitcoin", "эфириум", "ethereum", "криптовалюта",
    "crypto", "blockchain", "блокчейн", "bnb", "dogecoin", "solana",
    "trx", "usdt", "stablecoin", "крипторынок", "децентрализация"
]


def is_crypto_related(text, keywords):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def filter_crypto_news(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Убедимся, что есть нужные колонки
    if 'message_id' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV должен содержать колонки 'title' и 'content'.")

    # Добавим колонку-флаг: относится ли новость к крипте
    df['is_crypto'] = df.apply(
        lambda row: is_crypto_related(str(row['message_id']) + ' ' + str(row['text']), crypto_keywords),
        axis=1
    )

    # Отфильтруем только крипто-новости
    crypto_news = df[df['is_crypto']]

    return crypto_news[['message_id', 'text']]


# Пример использования:
if __name__ == "__main__":
    path_to_csv = "channels_content_2024_1_1/if_crypto_ru.csv"  # Замени на путь к своему файлу
    crypto_news_df = filter_crypto_news(path_to_csv)

    print("Найдено криптовалютных новостей:", len(crypto_news_df))
    print(crypto_news_df.head())  # Покажи первые новости

    # Сохраним результат
    crypto_news_df.to_csv("crypto_news.csv", index=False)
