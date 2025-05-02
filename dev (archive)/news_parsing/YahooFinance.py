import os
import csv
import time
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class YahooFinanceParser:
    """
    Класс для парсинга новостей с Yahoo Finance.
    """

    def __init__(self, base_url: str, section: str, delay: int):
        """
        :param base_url: Базовый URL сайта
        :param section: Раздел новостей
        :param delay: Задержка между запросами (сек)
        """
        self.base_url = base_url
        self.section = section
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; YahooFinanceParser/1.0)"
        })

    def parse_main_page(self):
        """
        Парсит основную страницу раздела.

        :return: Список словарей с новостями
        """
        url = self.base_url + self.section
        logging.info("Запрос: %s", url)
        resp = self.session.get(url)
        if resp.status_code != 200:
            logging.error("Ошибка загрузки: %s", resp.status_code)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        news_items = soup.select('li[data-test-locator="mega"]')
        if not news_items:
            logging.warning("Не найдено новостных блоков.")
            return []

        articles_list = []
        for item in news_items:
            title_tag = item.select_one('h3 a')
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            link = title_tag.get("href")
            if link.startswith("/"):
                link = self.base_url + link
            excerpt_tag = item.select_one("p")
            excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else ""
            articles_list.append({
                "title": title,
                "url": link,
                "excerpt": excerpt
            })

        return articles_list

    def parse_full_article(self, article_url: str) -> str:
        """
        Загружает полный текст статьи.

        :param article_url: URL статьи
        :return: Контент статьи
        """
        if not article_url.startswith("http"):
            return ""

        resp = self.session.get(article_url)
        if resp.status_code != 200:
            logging.warning("Не удалось загрузить статью: %s", article_url)
            return ""

        soup = BeautifulSoup(resp.text, "lxml")
        body_div = soup.select_one("div.caas-body")
        return body_div.get_text("\n", strip=True) if body_div else ""

    def parse_and_save(self, csv_filename: str, parse_full: bool = False):
        """
        Выполняет полный цикл: парсит статьи, сохраняет в CSV.

        :param csv_filename: Имя выходного CSV
        :param parse_full: Нужно ли парсить весь текст статьи
        """
        articles = self.parse_main_page()
        logging.info("Найдено статей: %d", len(articles))

        if not articles:
            return

        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "url", "excerpt", "content"])
            writer.writeheader()

            for art in articles:
                content = ""
                if parse_full:
                    content = self.parse_full_article(art["url"])
                    time.sleep(self.delay)

                writer.writerow({
                    "title": art["title"],
                    "url": art["url"],
                    "excerpt": art["excerpt"],
                    "content": content
                })

        logging.info("Сохранено %d статей в %s", len(articles), csv_filename)


if __name__ == "__main__":
    parser = YahooFinanceParser(
        base_url=os.getenv("YAHOO_BASE_URL", "https://finance.yahoo.com"),
        section=os.getenv("YAHOO_SECTION", "/topic/crypto"),
        delay=int(os.getenv("YAHOO_DELAY", 1))
    )
    parser.parse_and_save(
        csv_filename=os.getenv("YAHOO_CSV", "yahoo_crypto_news.csv"),
        parse_full=os.getenv("YAHOO_PARSE_FULL", "true").lower() == "true"
    )
