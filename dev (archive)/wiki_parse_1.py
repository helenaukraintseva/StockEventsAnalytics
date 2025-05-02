import wikipediaapi


def get_wikipedia_article(title, lang='ru'):
    wiki_wiki = wikipediaapi.Wikipedia(lang)
    page = wiki_wiki.page(title)

    if not page.exists():
        print(f"Статья '{title}' не найдена.")
        return None

    result = {
        'title': page.title,
        'summary': page.summary,
        'text': page.text
    }

    return result


def save_article_to_file(article, filename='article.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Заголовок: {article['title']}\n\n")
        f.write(f"Резюме:\n{article['summary']}\n\n")
        f.write(f"Текст:\n{article['text']}\n")


if __name__ == "__main__":
    query = input("Введите название статьи: ")
    article = get_wikipedia_article(query)

    if article:
        save_article_to_file(article)
        print("Статья сохранена в файл 'article.txt'")
