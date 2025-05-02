import os
import joblib
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

nltk.download("stopwords")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STOPWORDS = stopwords.words("russian")
DATA_PATH = os.getenv("CRYPTO_CLS_DATA", "crypto_news_total.csv")

df = pd.read_csv(DATA_PATH)
df.dropna(subset=["text", "is_crypto"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["is_crypto"], test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words=STOPWORDS, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

logging.info("Accuracy: %.2f", accuracy_score(y_test, y_pred))
logging.info("Precision: %.2f", precision_score(y_test, y_pred))
logging.info("Recall: %.2f", recall_score(y_test, y_pred))
logging.info("F1 Score: %.2f", f1_score(y_test, y_pred))
logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))

joblib.dump(model, "crypto_classifier_model.pkl")

# Примеры
examples = [
    "Binance launches new staking service for Ethereum users",
    """Кремль надеется завершить переговоры по Украине до 9 мая...""",
    """Farcaster — это протокол, обеспечивающий децентрализацию..."""
]

for i, ex in enumerate(examples, 1):
    logging.info("Пример %d: %s", i, model.predict([ex])[0])
