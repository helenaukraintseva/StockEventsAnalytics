import os
import re
import logging
import wikipedia
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def sanitize_filename(name: str) -> str:
    """
    Удаляет недопустимые символы из имени файла.

    :param name: исходное имя
    :return: безопасное имя файла
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def collect_related_articles(theme: str, max_articles: int = 100):
    """
    Сохраняет контент статей Wikipedia, связанных с заданной темой.

    :param theme: тема основной статьи
    :param max_articles: макс. число связанных статей
    """
    wikipedia.set_lang("ru")

    try:
        main_page = wikipedia.page(theme)
    except wikipedia.exceptions.PageError:
        logging.error("Статья по теме '%s' не найдена.", theme)
        return
    except wikipedia.exceptions.DisambiguationError as e:
        logging.warning("Тема '%s' неоднозначна. Возможные варианты: %s", theme, e.options)
        return

    logging.info("🔎 Основная статья: %s\n%s", main_page.title, main_page.url)

    links = main_page.links[:max_articles]
    logging.info("📄 Найдено связанных статей: %d", len(links))

    os.makedirs("wiki", exist_ok=True)

    for i, title in enumerate(links, 1):
        try:
            page = wikipedia.page(title)
            safe_title = sanitize_filename(page.title)
            filepath = os.path.join("wiki", f"{safe_title}.txt")

            with open(filepath, mode="w", encoding="utf-8") as file:
                file.write(page.content)

            logging.info("%d. Сохранено: %s → %s", i, page.title, filepath)

        except wikipedia.exceptions.PageError:
            logging.warning("%d. %s — страница не найдена", i, title)
        except wikipedia.exceptions.DisambiguationError:
            logging.warning("%d. %s — неоднозначная ссылка", i, title)


if __name__ == "__main__":
    topic = os.getenv("WIKI_TOPIC", "Криптовалюта")
    max_links = int(os.getenv("WIKI_MAX_ARTICLES", 100))
    collect_related_articles(topic, max_articles=max_links)
