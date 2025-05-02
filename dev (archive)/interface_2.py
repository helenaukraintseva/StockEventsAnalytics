import streamlit as st
from datetime import datetime, timedelta
import joblib
from sklearn.pipeline import Pipeline
from parsing_news.telegram_4 import parse_telegram_news
from config import api_id, api_hash, phone
import pandas as pd

# --- Загрузка моделей ---
crypto_model = joblib.load("NLP/crypto_classifier_model.pkl")
sentiment_model = joblib.load("NLP/felt_classifier_model.pkl")
label_encoder = joblib.load("NLP/label_encoder.pkl")

# --- Обёртки ---
crypto_pipe = Pipeline([("clf", crypto_model)])
sentiment_pipe = Pipeline([("clf", sentiment_model)])

# === Получение новостей из Telegram ===
def fetch_news(source, days_back):
    return parse_telegram_news(days_back=days_back, channel_title=source, api_id=api_id, api_hash=api_hash, phone=phone)

# === Обработка и классификация новостей ===
def process_news(news_list):
    results = []
    for news in news_list:
        text = news["text"]
        if crypto_pipe.predict([text])[0]:  # если это крипто-новость
            sentiment = sentiment_pipe.predict([text])[0]
            sentiment_label = label_encoder.inverse_transform([sentiment])[0]
            results.append({
                "Дата": news["date"],
                "Время": news["time"],
                "Новость": text,
                "Настроение": sentiment_label
            })
    return results

# === Интерфейс ===
st.set_page_config(page_title="Классификация крипто-новостей", layout="wide")
st.title("📰 Анализ крипто-новостей по источнику и времени")

# --- Выбор параметров ---
source = st.selectbox("Источник новостей:", ["if_market_news", "web3news", "cryptodaily", "slezisatoshi"])  # можно добавить другие
days_back = st.slider("За сколько дней назад брать новости?", min_value=1, max_value=30, value=7)

# --- Кнопка запуска анализа ---
if st.button("🔍 Анализировать"):
    with st.spinner("Загружаем и анализируем..."):
        raw_news = fetch_news(source, days_back)
        processed_news = process_news(raw_news)

        if processed_news:
            df = pd.DataFrame(processed_news)
            st.success(f"Обнаружено крипто-новостей: {len(df)}")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Крипто-новости не найдены по указанным параметрам.")
