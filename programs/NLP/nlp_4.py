import os
import joblib
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download("stopwords")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STOPWORDS = stopwords.words("russian")
DATA_PATH = os.getenv("FELT_DATA_PATH", "crypto_news_data.csv")

df = pd.read_csv(DATA_PATH)
df.dropna(subset=["text", "felt"], inplace=True)

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["felt"])

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label_encoded"], test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words=STOPWORDS, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial"))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

logging.info("Accuracy: %.2f", accuracy_score(y_test, y_pred))
logging.info("Классификационный отчёт:\n%s", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(model, "felt_classifier_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

example = "Bitcoin price surges after ETF approval"
pred = model.predict([example])[0]
label = label_encoder.inverse_transform([pred])[0]
logging.info("Пример: %s | Предсказание: %s", example, label)
