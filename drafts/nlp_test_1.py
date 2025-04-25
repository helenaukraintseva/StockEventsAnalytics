import joblib
from sklearn.pipeline import Pipeline

# Загрузка LabelEncoder и Pipeline
label_encoder = joblib.load('label_encoder.pkl')

pred_model = joblib.load('crypto_classifier_model.pkl')

model = Pipeline([
    ("clf", pred_model),
])
example = "Bitcoin price surges after ETF approval"

pred = model.predict([example])[0]
label = label_encoder.inverse_transform([pred])[0]

print(f"\n📈 Пример: {example}")
print(f"🔎 Предсказание: {label}")