import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

URL = "https://decrypt.co/"
headers = {"User-Agent": UserAgent().random}


def get_crypto_news():
    response = requests.get(URL, headers=headers)
    if response.status_code != 200:
        print(f"Ошибка при запросе: {response.status_code}")
        return []
    soup = BeautifulSoup(response.content, "html.parser")
    news_items = []
    for article in soup.find_all("div", class_="post-preview"):
        title_tag = article.find("h2", class_="post-title")
        link_tag = article.find("a", href=True)
        if title_tag and link_tag:
            title = title_tag.get_text(strip=True)
            link = link_tag["href"]
            news_items.append({"title": title, "link": link})
    return news_items


if __name__ == "__main__":
    news = get_crypto_news()
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}\n   {item['link']}\n")
