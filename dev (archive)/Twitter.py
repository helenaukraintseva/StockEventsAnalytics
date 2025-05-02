import tweepy
import csv
import time

class TwitterParser:
    """
    Класс для получения (парсинга) твитов при помощи Twitter API v2 через библиотеку tweepy.
    """
    def __init__(self, bearer_token, query="#bitcoin", max_results=10):
        """
        :param bearer_token: str - Ваш Bearer Token от Twitter API
        :param query: str - строка для поиска (например: "#bitcoin" или "crypto lang:en")
        :param max_results: int - максимальное количество твитов за один запрос
        """
        self.bearer_token = bearer_token
        self.query = query
        self.max_results = max_results

        # Инициализация клиента Twitter (tweepy >= 4.0)
        self.client = tweepy.Client(bearer_token=self.bearer_token)

    def search_tweets(self, filename="tweets.csv"):
        """
        Ищем твиты по query, сохраняем в CSV (id, text, created_at, author_id).
        Может быть расширено.
        """
        print(f"[INFO] Поиск твитов по запросу: {self.query}, max_results={self.max_results}")
        try:
            # Параметры поиска
            response = self.client.search_recent_tweets(
                query=self.query,
                max_results=self.max_results,        # максимум 10..100
                tweet_fields=["id","text","created_at","author_id","lang"]
                # можно добавить user_fields, expansions и т.д.
            )
        except tweepy.TweepyException as e:
            print("[ERROR] Ошибка при запросе к Twitter API:", e)
            return

        if not response.data:
            print("[INFO] Нет найденных твитов.")
            return

        tweets_data = response.data   # это список Tweet объектов

        # Запись в CSV
        with open(filename, mode="w", encoding="utf-8", newline="") as f:
            fieldnames = ["id","text","created_at","author_id","lang"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for tweet in tweets_data:
                row = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "author_id": tweet.author_id,
                    "lang": tweet.lang
                }
                writer.writerow(row)

        print(f"[INFO] Сохранено {len(tweets_data)} твитов в {filename}")


if __name__ == "__main__":
    # Вставьте Ваш Bearer Token (cекретный ключ)
    my_bearer_token = "YOUR_BEARER_TOKEN"

    # Пример: хотим искать твиты с #bitcoin
    parser = TwitterParser(bearer_token=my_bearer_token, query="#bitcoin lang:en", max_results=20)
    parser.search_tweets("bitcoin_tweets.csv")
