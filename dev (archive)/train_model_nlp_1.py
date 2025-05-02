import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch


# === 1. Загрузка и подготовка данных ===
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    if 'content' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV должен содержать колонки 'content' и 'sentiment'.")

    # Преобразуем метки в числа
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label_map)

    return train_test_split(df, test_size=0.2, random_state=42), label_map


# === 2. Токенизация ===
def tokenize_data(tokenizer, dataset):
    return dataset.map(lambda x: tokenizer(x['content'], truncation=True, padding='max_length'), batched=True)


# === 3. Обучение ===
def train_sentiment_model(csv_path):
    (train_df, val_df), label_map = load_and_prepare_data(csv_path)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = Dataset.from_pandas(train_df[['content', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['content', 'label']])

    train_dataset = tokenize_data(tokenizer, train_dataset)
    val_dataset = tokenize_data(tokenizer, val_dataset)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="./sentiment_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")

    print("✅ Модель обучена и сохранена в папке ./sentiment_model")


# === 4. Пример использования ===
def predict_sentiment(text):
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs).item()

    label_map_reverse = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map_reverse[label_id]


if __name__ == "__main__":
    # Обучение модели
    train_sentiment_model("news_with_sentiment.csv")

    # Пример предсказания
    test_text = "Биткоин продолжает стремительно расти на фоне интереса институционалов"
    prediction = predict_sentiment(test_text)
    print("Предсказание настроения:", prediction)
