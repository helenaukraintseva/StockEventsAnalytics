import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
import joblib
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")

russian_stopwords = stopwords.words("russian")

RUSSIAN_STOP_WORDS = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь"
]

# --- 1. Загрузка датасета ---
df = pd.read_csv("crypto_news_total.csv")  # файл с колонками text, is_crypto
df.dropna(subset=["text", "is_crypto"], inplace=True)

# --- 2. Разделение на train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["is_crypto"], test_size=0.2, random_state=42
)

# --- 3. NLP-пайплайн ---
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words=russian_stopwords,  # можно заменить на "russian" для русского
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

# --- 4. Обучение модели ---
model.fit(X_train, y_train)

# --- 5. Оценка качества ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 6. Сохранение модели (опционально) ---
joblib.dump(model, "crypto_classifier_model.pkl")

# --- 7. Пример использования ---
example = ["Binance launches new staking service for Ethereum users"]
print("Пример:", model.predict(example)[0])

ex_1 = ["""азработчик блокчейна Картик Патель представил Anon World, социальную платформу, основанную на протоколе Farcaster, которая гарантирует анонимность и приватность пользователей. Платформа предлагает интерфейс в стиле Reddit, способствующий удобному общению и защите данных.

🔒 Децентрализованный Farcaster в основе безопасности  
Farcaster — это протокол, обеспечивающий:  
- Децентрализацию данных, исключающую контроль одной компании.  
- Защиту от взломов и манипуляций через распределенное хранение.  
- Отсутствие цензуры и внешнего контроля над контентом.  

✨ AnonCast: от подкаста к социальной платформе  
Запуск Anon World стал следующим шагом в развитии проекта AnonCast, известного своей приверженностью анонимности и свободе слова. Платформа станет убежищем для пользователей, ценящих свою приватность.
"""]

ex_2 = ["""Кремль надеется завершить переговоры по Украине до 9 мая, когда будет отмечаться 80-я годовщина победы Советского Союза над нацистской Германией. Цель — устроить двойной праздник, — корреспондент Sky News в России"""]

print("Пример 1:", model.predict(ex_1)[0])
print("Пример 2:", model.predict(ex_2)[0])