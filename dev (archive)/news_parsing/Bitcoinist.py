import os
import time
import csv
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Загрузка конфигурации из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BitcoinistParser:
    """
    Класс для парсинга новостей про криптовалюты с сайта Bitcoinist.
    """

    def __init__(self, base_url: str, sleep_time: float = 1):
        """
        :param base_url: URL-адрес сайта для парсинга
        :param sleep_time: Пауза между запросами (секунды)
        """
        self.base_url = base_url
        self.sleep_time = sleep_time

    def parse_main_page(self):
        """
        Получает список статей с главной страницы.

        :return: список словарей с данными статей
        """
        resp = requests.get(self.base_url)
        if resp.status_code != 200:
            logging.error("Ошибка при запросе %s, код = %s", self.base_url, resp.status_code)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        articles_html = soup.select("div.td_module_10")
        articles_data = []

        for article in articles_html:
            title_tag = article.select_one("h3 a")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            link = title_tag.get("href")
            excerpt_tag = article.select_one("div.td-excerpt")
            excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else ""

            articles_data.append({
                "title": title,
                "url": link,
                "excerpt": excerpt
            })

        return articles_data

    def parse_full_article(self, article_url: str) -> str:
        """
        Получает полный текст статьи по URL.

        :param article_url: URL статьи
        :return: Текст статьи или пустая строка
        """
        resp = requests.get(article_url)
        if resp.status_code != 200:
            logging.warning("Не удалось получить статью: %s, код=%s", article_url, resp.status_code)
            return ""

        soup = BeautifulSoup(resp.text, "lxml")
        content_div = soup.select_one("div.td-post-content")
        return content_div.get_text("\n", strip=True) if content_div else ""

    def parse_and_save(self, csv_filename: str):
        """
        Парсит главную страницу и статьи, сохраняет в CSV.

        :param csv_filename: Имя файла для сохранения
        """
        articles = self.parse_main_page()
        logging.info("Найдено статей: %d", len(articles))

        if not articles:
            logging.warning("Список статей пуст. Завершаем.")
            return

        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "url", "excerpt", "content"])
            writer.writeheader()

            for art in articles:
                content = self.parse_full_article(art["url"])
                writer.writerow({
                    "title": art["title"],
                    "url": art["url"],
                    "excerpt": art["excerpt"],
                    "content": content
                })
                time.sleep(self.sleep_time)

        logging.info("Данные сохранены в '%s'.", csv_filename)


if __name__ == "__main__":
    parser = BitcoinistParser(
        base_url=os.getenv("BITCOINIST_URL", "https://bitcoinist.com"),
        sleep_time=float(os.getenv("BITCOINIST_SLEEP", 1))
    )
    parser.parse_and_save("bitcoinist_news.csv")
