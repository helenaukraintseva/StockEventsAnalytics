import requests

API_KEY = 'YOUR_CRYPTOPANIC_API_KEY'
URL = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&kind=news"

def get_crypto_news():
    response = requests.get(URL)
    if response.status_code != 200:
        print(f"Ошибка при запросе: {response.status_code}")
        return []
    data = response.json()
    news_items = [{"title": post["title"], "link": post["url"]} for post in data.get("results", [])]
    return news_items

if __name__ == "__main__":
    news = get_crypto_news()
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}\n   {item['link']}\n")
