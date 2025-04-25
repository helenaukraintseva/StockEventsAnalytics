import joblib
from sklearn.pipeline import Pipeline

# --- Загрузка моделей ---
crypto_model = joblib.load("crypto_classifier_model.pkl")         # модель "крипта / не крипта"
sentiment_model = joblib.load("felt_classifier_model.pkl")              # модель анализа настроения
label_encoder = joblib.load("label_encoder.pkl")                  # декодер меток настроения

# --- Обёртки для моделей ---
crypto_pipe = Pipeline([("clf", crypto_model)])
sentiment_pipe = Pipeline([("clf", sentiment_model)])

# --- Функция классификации ---
def process_news(news_list):
    results = []
    for news in news_list:
        is_crypto = crypto_pipe.predict([news])[0]
        if is_crypto:  # только если "крипта"
            sentiment_pred = sentiment_pipe.predict([news])[0]
            sentiment_label = label_encoder.inverse_transform([sentiment_pred])[0]
            results.append({
                "текст": news[:120] + "...",
                "тема": "крипта",
                "настроение": sentiment_label
            })
    return results

# --- Пример использования ---
if __name__ == "__main__":
    news_examples = [
        "Ethereum показал рост на 5% после одобрения ETF",
        "Президент выступил с заявлением по Украине",
        "Блокчейн-платформа Anon World интегрировала протокол Farcaster",
        "ЦБ РФ ожидает стабильность инфляции до конца года"
    ]

    output = process_news(news_examples)

    for i, item in enumerate(output, 1):
        print(f"\n🔹 Новость {i}:")
        print(f"Текст: {item['текст']}")
        print(f"Настроение: {item['настроение']}")
