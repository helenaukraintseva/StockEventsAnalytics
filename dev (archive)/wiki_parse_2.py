import wikipediaapi
import os


def get_wikipedia_article(title, lang='ru'):
    wiki = wikipediaapi.Wikipedia(
        language='ru',
        user_agent='VKRProjectBot/1.0 (EugeneDvorcoviy@example.com)'  # заменяй на свои данные
    )
    # wiki = wikipediaapi.Wikipedia(lang)
    page = wiki.page(title)
    if not page.exists():
        print(f"Статья '{title}' не найдена.")
        return None
    return page


def save_article(page, folder="wiki"):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{page.title}.txt")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Заголовок: {page.title}\n\n")
        f.write(f"Резюме:\n{page.summary}\n\n")
        f.write(f"Текст:\n{page.text}\n")
    print(f"[✔] Сохранена: {page.title}")


def collect_related_articles(start_title, max_articles=10, lang='ru'):
    wiki = wikipediaapi.Wikipedia(lang)
    visited = set()
    to_visit = [start_title]

    while to_visit and len(visited) < max_articles:
        current_title = to_visit.pop(0)
        if current_title in visited:
            continue

        page = wiki.page(current_title)
        if not page.exists():
            continue

        save_article(page)
        visited.add(current_title)

        # Добавим внутренние ссылки в очередь
        for linked_title in page.links.keys():
            if linked_title not in visited and linked_title not in to_visit:
                to_visit.append(linked_title)

    print(f"\n[✓] Собрано {len(visited)} статей по теме '{start_title}'.")


if __name__ == "__main__":
    # theme = input("Введите тему (например, 'Криптовалюта'): ")
    theme = "Криптовалюта"
    collect_related_articles(theme, max_articles=100)
