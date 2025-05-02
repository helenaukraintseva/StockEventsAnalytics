import os
import logging
import praw
from dotenv import load_dotenv

# Загрузка переменных
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)


def get_crypto_news(subreddit_name: str = "cryptocurrency", limit: int = 10):
    """
    Получает популярные посты с Reddit.

    :param subreddit_name: Название сабреддита
    :param limit: Количество постов
    :return: список словарей с заголовками и ссылками
    """
    subreddit = reddit.subreddit(subreddit_name)
    news_items = []

    for submission in subreddit.hot(limit=limit):
        news_items.append({"title": submission.title, "link": submission.url})

    return news_items


if __name__ == "__main__":
    news = get_crypto_news()
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}\n   {item['link']}\n")
