import requests
from bs4 import BeautifulSoup
import time
import csv


class BitcoinistParser:
    """
    Класс для парсинга новостей про криптовалюты с сайта Bitcoinist.
    """

    def __init__(self, base_url="https://bitcoinist.com", sleep_time=1):
        """
        :param base_url: базовый URL сайта,
                         можно менять, например, на 'https://bitcoinist.com/latest' и т.д.
        :param sleep_time: задержка (в секундах) между запросами,
                          чтобы не нагружать сайт.
        """
        self.base_url = base_url
        self.sleep_time = sleep_time

    def parse_main_page(self):
        """
        Парсит главную страницу (или заданный base_url) – извлекает статьи:
        заголовок, ссылка, краткое описание (если есть).

        :return: список словарей вида [
           {
             "title": "...",
             "url": "...",
             "excerpt": "..."
           },
           ...
        ]
        """
        resp = requests.get(self.base_url)
        if resp.status_code != 200:
            print(f"[ERROR] Ошибка при запросе {self.base_url}, код = {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")

        # Пример: ищем блок <div class="td_module_10">
        # (актуальный CSS-селектор может отличаться - уточните вручную через DevTools)
        articles_html = soup.select("div.td_module_10")

        articles_data = []
        for article in articles_html:
            # Находим заголовок
            title_tag = article.select_one("h3 a")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            link = title_tag.get("href")

            # Краткое описание (excerpt), если есть
            excerpt_tag = article.select_one("div.td-excerpt")
            excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else ""

            articles_data.append({
                "title": title,
                "url": link,
                "excerpt": excerpt
            })

        return articles_data

    def parse_full_article(self, article_url):
        """
        Переходит по ссылке статьи, вытягивает полный текст контента.

        :param article_url: полная ссылка на статью
        :return: строка текста (или пустая строка, если не удалось получить)
        """
        resp = requests.get(article_url)
        if resp.status_code != 200:
            print(f"[WARN] Не удалось получить статью: {article_url}, код={resp.status_code}")
            return ""

        soup = BeautifulSoup(resp.text, "lxml")

        # Ищем основной контент статьи, напр. <div class="td-post-content">
        content_div = soup.select_one("div.td-post-content")
        if not content_div:
            return ""

        full_text = content_div.get_text("\n", strip=True)
        return full_text

    def parse_and_save(self, csv_filename="bitcoinist_news.csv"):
        """
        Полный цикл: парсит главную страницу, перебирает ссылки на статьи,
        парсит их контент и сохраняет всё в CSV-файл.

        :param csv_filename: имя выходного CSV
        """
        # 1) Список статей (заголовок, url, excerpt)
        articles = self.parse_main_page()
        print(f"Найдено статей: {len(articles)}")

        if not articles:
            print("Список статей пуст. Завершаем.")
            return

        # 2) Открываем CSV
        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "url", "excerpt", "content"])
            writer.writeheader()

            for art in articles:
                # Парсим полный текст статьи
                content = self.parse_full_article(art["url"])
                row = {
                    "title": art["title"],
                    "url": art["url"],
                    "excerpt": art["excerpt"],
                    "content": content
                }
                writer.writerow(row)

                # Небольшая задержка, чтобы не перегружать сайт запросами
                time.sleep(self.sleep_time)

        print(f"[INFO] Данные сохранены в '{csv_filename}'.")


if __name__ == "__main__":
    # Пример использования
    parser = BitcoinistParser(
        base_url="https://bitcoinist.com",
        sleep_time=1  # задержка 1 сек между статьями
    )
    parser.parse_and_save("bitcoinist_news.csv")
