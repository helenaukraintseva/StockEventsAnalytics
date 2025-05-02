import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
import joblib
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

russian_stopwords = stopwords.words("russian")

# --- 1. Загружаем данные ---
df = pd.read_csv("crypto_news_data.csv")  # Файл должен содержать колонки: text, felt
df.dropna(subset=["text", "felt"], inplace=True)

# --- 2. Кодируем метки (Positive, Negative, Neutral) ---
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["felt"])  # 0, 1, 2

# --- 3. Разделение на train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_encoded"], test_size=0.2, random_state=42
)

# --- 4. Создаём NLP пайплайн ---
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words=russian_stopwords,  # Использовать "russian" или список для русскоязычных текстов
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial"))
])

# --- 5. Обучение модели ---
model.fit(X_train, y_train)

# --- 6. Оценка модели ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred))
print("\n🧠 Классификационный отчёт:")

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- 7. Сохраняем модель и LabelEncoder ---
joblib.dump(model, "felt_classifier_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# --- 8. Пример использования ---
example = "Bitcoin price surges after ETF approval"
pred = model.predict([example])[0]
label = label_encoder.inverse_transform([pred])[0]
print(f"\n📈 Пример: {example}")
print(f"🔎 Предсказание: {label}")
