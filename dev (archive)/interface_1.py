import streamlit as st
import joblib
from sklearn.pipeline import Pipeline

# --- Загрузка моделей ---
crypto_model = joblib.load("NLP/crypto_classifier_model.pkl")
sentiment_model = joblib.load("NLP/felt_classifier_model.pkl")
label_encoder = joblib.load("NLP/label_encoder.pkl")

# --- Обёртки ---
crypto_pipe = Pipeline([("clf", crypto_model)])
sentiment_pipe = Pipeline([("clf", sentiment_model)])

# --- Функция обработки новостей ---
def process_news(news_list):
    results = []
    for news in news_list:
        is_crypto = crypto_pipe.predict([news])[0]
        if is_crypto:
            sentiment_pred = sentiment_pipe.predict([news])[0]
            sentiment_label = label_encoder.inverse_transform([sentiment_pred])[0]
            results.append({
                "Новость": news,
                "Тема": "Крипта",
                "Настроение": sentiment_label
            })
    return results


# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Классификация новостей", layout="wide")
st.title("📰 Классификация крипто-новостей по настроению")

with st.form("news_input_form"):
    user_input = st.text_area("Введите новости (по одной на строку):", height=300)
    submitted = st.form_submit_button("🔍 Классифицировать")

if submitted:
    news_list = [line.strip() for line in user_input.split("\n") if line.strip()]
    if not news_list:
        st.warning("❗ Пожалуйста, введите хотя бы одну новость.")
    else:
        with st.spinner("Обрабатываем..."):
            results = process_news(news_list)
            if results:
                st.success(f"Обнаружено {len(results)} крипто-новостей:")
                st.dataframe(results, use_container_width=True)
            else:
                st.info("Крипто-новости не найдены.")