import tweepy

bearer_token = "YOUR_BEARER_TOKEN"

client = tweepy.Client(bearer_token=bearer_token)

query = "искусственный интеллект lang:ru"
tweets = client.search_recent_tweets(query=query, max_results=10)

for tweet in tweets.data:
    print(tweet.text)
