import requests
from bs4 import BeautifulSoup
import csv
import time

class YahooFinanceParser:
    """
    Класс для парсинга новостей о криптовалюте (или другом) с Yahoo Finance.
    По умолчанию смотрим раздел /topic/crypto,
    где обычно публикуются статьи/новости по криптовалютам.
    """

    def __init__(self, base_url="https://finance.yahoo.com", section="/topic/crypto", delay=1):
        """
        :param base_url: базовый URL (не менять, если не нужно)
        :param section: раздел, например '/topic/crypto' или '/cryptocurrencies'
        :param delay: задержка (в секундах) между запросами
        """
        self.base_url = base_url
        self.section = section
        self.delay = delay
        self.session = requests.Session()
        # При желании можно задать User-Agent
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; YahooFinanceParser/1.0)"
        })

    def parse_main_page(self):
        """
        Парсит основную страницу с новостями (по умолчанию /topic/crypto).
        Возвращает список словарей с заголовком, ссылкой, отрывком/описанием.
        """
        url = self.base_url + self.section
        print(f"[INFO] Запрос: {url}")
        resp = self.session.get(url)
        if resp.status_code != 200:
            print(f"[ERROR] Код ответа={resp.status_code}, не удалось загрузить {url}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        # Нужно изучать HTML (F12) – актуально на момент написания:
        # новости могут храниться в блоке div#Fin-Stream,
        # каждая новость - <li> с data-test-locator="mega" или ...
        articles_list = []

        # Пример селектора (может меняться)
        # Возьмём 'div[id="Fin-Stream"] li'
        # Но возможно нужно искать 'li[data-test-locator="mega"]'
        news_items = soup.select('li[data-test-locator="mega"]')
        if not news_items:
            print("[WARN] Не найдено новостных блоков по селектору li[data-test-locator='mega']")
            return []

        for item in news_items:
            # Заголовок
            title_tag = item.select_one('h3 a')
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            link = title_tag.get("href")
            if link and link.startswith("/"):
                link = self.base_url + link

            # Описание (excerpt)
            # Иногда присутствует <p> (содержит краткий текст)
            excerpt_tag = item.select_one("p")
            excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else ""

            articles_list.append({
                "title": title,
                "url": link,
                "excerpt": excerpt
            })

        return articles_list

    def parse_full_article(self, article_url):
        """
        Переход к полной статье, попытка извлечь контент.
        Внимание: структуру часто меняют,
        может быть <div class="caas-body"> ...
        """
        if not article_url.startswith("http"):
            return ""

        resp = self.session.get(article_url)
        if resp.status_code != 200:
            print(f"[WARN] Не удалось загрузить статью {article_url}, код={resp.status_code}")
            return ""

        soup = BeautifulSoup(resp.text, "lxml")
        # Пробуем найти основной текст статьи (напр. <div class="caas-body">)
        body_div = soup.select_one("div.caas-body")
        if not body_div:
            return ""

        content = body_div.get_text("\n", strip=True)
        return content

    def parse_and_save(self, csv_filename="yahoo_crypto_news.csv", parse_full=False):
        """
        Основной метод: парсит раздел (section), получает список новостей,
        при parse_full=True парсит полный текст каждой статьи.
        Сохраняет результат в CSV.
        """
        articles = self.parse_main_page()
        print(f"[INFO] Найдено статей: {len(articles)}")

        if not articles:
            return

        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["title", "url", "excerpt", "content"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for art in articles:
                content = ""
                if parse_full and art["url"]:
                    content = self.parse_full_article(art["url"])
                    time.sleep(self.delay)

                row = {
                    "title": art["title"],
                    "url": art["url"],
                    "excerpt": art["excerpt"],
                    "content": content
                }
                writer.writerow(row)

        print(f"[INFO] Сохранено {len(articles)} статей в {csv_filename}")


if __name__ == "__main__":
    parser = YahooFinanceParser(
        base_url="https://finance.yahoo.com",
        section="/topic/crypto",  # или "/cryptocurrencies"
        delay=1
    )
    # Парсим и сохраняем
    parser.parse_and_save(csv_filename="yahoo_crypto_news.csv", parse_full=True)
