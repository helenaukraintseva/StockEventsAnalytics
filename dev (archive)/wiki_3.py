import wikipedia
import os
import csv
import re

def sanitize_filename(name):
    """Удаляет недопустимые символы из имени файла."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def collect_related_articles(theme: str, max_articles: int = 100):
    wikipedia.set_lang("ru")

    try:
        main_page = wikipedia.page(theme)
    except wikipedia.exceptions.PageError:
        print(f"Статья по теме '{theme}' не найдена.")
        return
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Тема '{theme}' неоднозначна. Возможные варианты: {e.options}")
        return

    print(f"🔎 Основная статья: {main_page.title}\n{main_page.url}\n")

    links = main_page.links[:max_articles]
    print(f"📄 Найдено связанных статей: {len(links)}\n")

    # Создаём папку, если не существует
    os.makedirs("wiki", exist_ok=True)

    for i, title in enumerate(links, 1):
        try:
            page = wikipedia.page(title)
            safe_title = sanitize_filename(page.title)
            filepath = os.path.join("wiki", f"{safe_title}.csv")

            with open(filepath, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["title", "content"])
                writer.writerow([page.title, page.content])

            print(f"{i}. Сохранено: {page.title} → {filepath}")

        except wikipedia.exceptions.PageError:
            print(f"{i}. {title} — страница не найдена")
        except wikipedia.exceptions.DisambiguationError:
            print(f"{i}. {title} — неоднозначная ссылка")

# Пример вызова
if __name__ == "__main__":
    тема = "Криптовалюта"
    collect_related_articles(тема, max_articles=100)
