import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np

# --- 1. Загружаем и обрабатываем CSV ---
df = pd.read_csv("crypto_news_total.csv")  # колонки: text, felt
df.dropna(subset=["text", "felt"], inplace=True)

# --- 2. Кодируем метки (felt -> 0,1,2) ---
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["felt"])

# --- 3. Преобразуем в HuggingFace Dataset ---
dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.train_test_split(test_size=0.2)

# --- 4. Подготовка токенизатора ---
model_name = "distilroberta-base"  # лёгкая и точная модель
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

encoded_dataset = dataset.map(tokenize, batched=True)

# --- 5. Загрузка модели ---
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# --- 6. Аргументы обучения ---
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

# --- 7. Метрики ---
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

# --- 8. Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# --- 9. Обучение ---
trainer.train()

# --- 10. Сохранение модели и кодировщика ---
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

# --- 11. Функция для предсказания с вероятностями ---
from transformers import pipeline
classifier = pipeline("text-classification", model="./sentiment_model", tokenizer="./sentiment_model", return_all_scores=True)
label_encoder = joblib.load("label_encoder.pkl")

def predict_sentiment(text):
    result = classifier(text)[0]
    scores = {label_encoder.inverse_transform([i])[0]: round(score["score"], 3) for i, score in enumerate(result)}
    predicted_label = max(scores, key=scores.get)
    return predicted_label, scores

# --- 12. Пример ---
text = "Ethereum is showing strong performance this week and attracting investors"
label, probs = predict_sentiment(text)
print(f"\n📰 Текст: {text}")
print(f"📊 Предсказание: {label}")
print(f"📈 Уверенности: {probs}")
