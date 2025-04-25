import praw

reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

def get_crypto_news(subreddit_name='cryptocurrency', limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    news_items = []
    for submission in subreddit.hot(limit=limit):
        news_items.append({"title": submission.title, "link": submission.url})
    return news_items

if __name__ == "__main__":
    news = get_crypto_news()
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}\n   {item['link']}\n")
