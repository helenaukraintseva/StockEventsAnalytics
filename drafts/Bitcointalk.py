import requests
from bs4 import BeautifulSoup
import time
import csv

class BitcointalkParser:
    """
    Класс для парсинга форума Bitcointalk (темы + сообщения).
    Внимание: сайт может иметь капчу и анти-спам.
    """

    def __init__(self, board_id=1, base_url="https://bitcointalk.org", user_agent=None, delay=2):
        """
        :param board_id: Числовой ID раздела (напр. 1 - это 'Bitcoin Discussion').
        :param base_url: Базовый URL (не менять, если не нужно).
        :param user_agent: Строка User-Agent (можно подставить реальный браузер).
        :param delay: Задержка (сек) между запросами.
        """
        self.board_id = board_id
        self.base_url = base_url
        self.session = requests.Session()
        self.delay = delay

        # При желании задать User-Agent
        if user_agent:
            self.session.headers.update({"User-Agent": user_agent})
        else:
            self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; BitcointalkParser/1.0)"})

    def parse_board_page(self, page=0):
        """
        Парсит одну страницу раздела (board) - извлекает список тем (название, автор, ссылка).
        Bitcointalk URL-форма:
        https://bitcointalk.org/index.php?board={board_id}.{page}
        - board_id: ID раздела
        - page: кратно 40 (первая страница: .0, вторая: .40)
        :param page: номер "страницы" (0, 40, 80 ...)
        :return: список словарей: [{"title": "...", "url": "...", "author": "...", "replies": ...}, ...]
        """
        full_url = f"{self.base_url}/index.php?board={self.board_id}.{page}"
        print(f"[INFO] Запрос: {full_url}")
        resp = self.session.get(full_url)
        if resp.status_code != 200:
            print(f"[ERROR] Код ответа = {resp.status_code}. Возможно, требуется капча.")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        # Обычно темы лежат в таблице с классом "table_grid"
        # Пример: <div id="bodyarea"> <table class="table_grid" ...> ...
        # Нужно изучить структуру HTML.

        table = soup.select_one("table.table_grid")
        if not table:
            print("[WARN] Не найдена таблица с классом 'table_grid'")
            return []

        # В таблице обычно строки <tr class="windowbg" ...>
        # или <tr class="windowbg2">. В каждой строке - данные о теме.
        rows = table.select("tr.windowbg, tr.windowbg2")
        topics_list = []

        for row in rows:
            # Ищем ячейку с ссылкой на тему
            link_tag = row.select_one("td.subject a")
            if not link_tag:
                # Может быть служебная строка
                continue

            title = link_tag.get_text(strip=True)
            url = link_tag.get("href")  # полный URL: "https://bitcointalk.org/index.php?topic=..."

            # Автор
            author_td = row.select_one("td.lastpost td span a")
            if author_td:
                author = author_td.get_text(strip=True)
            else:
                # fallback
                author = ""

            # Replies (количество ответов) - хранится в <td class="replies">, возможно
            replies_td = row.select_one("td.stats")
            replies = 0
            if replies_td:
                # Пример: "Replies: 123 <br> Views: 4567"
                # Или <a ...> 123 </a>. Нужно смотреть реальную HTML.
                replies_text = replies_td.get_text(strip=True)
                # Можно парсить вручную
                # replies = ...
                # Упростим:
                replies = replies_text

            topics_list.append({
                "title": title,
                "url": url,
                "author": author,
                "replies": replies
            })

        return topics_list

    def parse_topic(self, topic_url):
        """
        Парсит страницу конкретной темы (topic):
         - заголовок, список сообщений (автор, дата, контент)
        :param topic_url: ссылка вида https://bitcointalk.org/index.php?topic=....
        :return: словарь {"title": "...", "posts": [ { "author":"...", "content":"...", ...}, ...]}
        """
        resp = self.session.get(topic_url)
        if resp.status_code != 200:
            print(f"[ERROR] Не удалось загрузить тему: {topic_url}, код={resp.status_code}")
            return {"title": "", "posts": []}

        soup = BeautifulSoup(resp.text, "lxml")
        # Заголовок темы
        title_tag = soup.select_one("div#bodyarea h1")
        topic_title = title_tag.get_text(strip=True) if title_tag else "No Title"

        # Список сообщений: <div id="forumposts"> ... <td class="td_headerandpost">
        # Bitcointalk может иметь разную структуру. Нужно адаптировать.
        post_divs = soup.select("td.td_headerandpost")

        posts_data = []
        for post_div in post_divs:
            # Автор
            author_tag = post_div.select_one("td.poster_info b a")
            author = author_tag.get_text(strip=True) if author_tag else "Unknown"

            # Содержимое поста
            # <div class="post" id="msg12345">
            content_div = post_div.select_one("div.post")
            content = content_div.get_text("\n", strip=True) if content_div else ""

            posts_data.append({
                "author": author,
                "content": content
            })

        return {
            "title": topic_title,
            "posts": posts_data
        }

    def crawl_board(self, pages=1, csv_filename="bitcointalk_threads.csv"):
        """
        Парсит нужное количество страниц (board= self.board_id) и сохраняет темы в CSV.
        :param pages: сколько страниц (каждая страница = +40 тем)
        :param csv_filename: куда сохранить
        """
        all_threads = []
        for i in range(pages):
            # page = i * 40 (формат Bitcointalk)
            page_offset = i * 40
            topics_list = self.parse_board_page(page=page_offset)
            print(f"[PAGE {i}] Найдено тем: {len(topics_list)}")
            all_threads.extend(topics_list)
            time.sleep(self.delay)

        # Сохраняем CSV
        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "url", "author", "replies"])
            writer.writeheader()
            for t in all_threads:
                writer.writerow(t)

        print(f"[INFO] Сохранено {len(all_threads)} тем в {csv_filename}")

    def crawl_topic_details(self, threads_csv="bitcointalk_threads.csv", out_csv="bitcointalk_posts.csv"):
        """
        Считывает список тем из CSV, затем парсит содержимое каждой темы (первую страницу) и сохраняет в новый CSV.
        :param threads_csv: CSV с темами (title, url, author, replies)
        :param out_csv: CSV куда складывать посты
        """
        import pandas as pd
        if not os.path.exists(threads_csv):
            print(f"[ERROR] Файл {threads_csv} не найден.")
            return

        df = pd.read_csv(threads_csv)
        # Для каждой темы парсим
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["topic_title", "post_author", "post_content", "topic_url"])
            writer.writeheader()

            for index, row in df.iterrows():
                topic_url = row["url"]
                if not topic_url.startswith("http"):
                    continue

                topic_data = self.parse_topic(topic_url)
                for post in topic_data["posts"]:
                    writer.writerow({
                        "topic_title": topic_data["title"],
                        "post_author": post["author"],
                        "post_content": post["content"],
                        "topic_url": topic_url
                    })

                time.sleep(self.delay)

        print(f"[INFO] Все посты сохранены в {out_csv}")


if __name__ == "__main__":
    import os

    parser = BitcointalkParser(
        board_id=1,  # Например, Board=1 (Bitcoin Discussion),
        base_url="https://bitcointalk.org",
        user_agent="Mozilla/5.0 (compatible; MyParser/1.0)",
        delay=2
    )
    # 1) Спарсим несколько страниц списка тем
    parser.crawl_board(pages=1, csv_filename="bitcointalk_threads.csv")

    # 2) Спарсим содержимое каждой темы (только 1 страницу topic)
    parser.crawl_topic_details(threads_csv="bitcointalk_threads.csv", out_csv="bitcointalk_posts.csv")
