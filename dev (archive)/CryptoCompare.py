import requests

API_KEY = 'YOUR_CRYPTOCOMPARE_API_KEY'
URL = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={API_KEY}"

def get_crypto_news():
    response = requests.get(URL)
    if response.status_code != 200:
        print(f"Ошибка при запросе: {response.status_code}")
        return []
    data = response.json()
    news_items = [{"title": article["title"], "link": article["url"]} for article in data.get("Data", [])]
    return news_items

if __name__ == "__main__":
    news = get_crypto_news()
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}\n   {item['link']}\n")
