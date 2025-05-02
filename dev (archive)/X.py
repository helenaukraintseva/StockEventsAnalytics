import snscrape.modules.twitter as sntwitter
import pandas as pd


def scrape_x_posts(query, max_tweets=100):
    tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        tweets.append({
            'date': tweet.date,
            'username': tweet.user.username,
            'content': tweet.content,
            'url': tweet.url,
            'likes': tweet.likeCount,
            'retweets': tweet.retweetCount,
            'replies': tweet.replyCount,
        })

    return pd.DataFrame(tweets)


# 🔍 Пример использования
if __name__ == "__main__":
    query = "искусственный интеллект lang:ru since:2024-01-01 until:2025-01-01"
    df = scrape_x_posts(query, max_tweets=50)
    print(df.head())
    df.to_csv("x_posts.csv", index=False)
