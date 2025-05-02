import os
import logging
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

URL = os.getenv("COINDESK_URL", "https://www.coindesk.com/")
headers = {"User-Agent": UserAgent().random}


def get_crypto_news():
    """
    Получает список новостей с сайта CoinDesk.

    :return: список словарей с заголовками и ссылками
    """
    response = requests.get(URL, headers=headers)
    if response.status_code != 200:
        logging.error("Ошибка при запросе: %s", response.status_code)
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    news_items = []

    articles = soup.find_all("article")

    for article in articles:
        link_tag = article.find("a", href=True)
        title_tag = article.find("h4") or article.find("h3") or article.find("h2")

        if link_tag and title_tag:
            title = title_tag.get_text(strip=True)
            link = link_tag["href"]
            if not link.startswith("http"):
                link = "https://www.coindesk.com" + link
            news_items.append({"title": title, "link": link})

    return news_items


if __name__ == "__main__":
    news = get_crypto_news()
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}\n   {item['link']}\n")
