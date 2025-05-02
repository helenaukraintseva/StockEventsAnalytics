import os
import torch
import pandas as pd
import logging
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.model_selection import train_test_split

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_NAME = os.getenv("TRANSFORMER_MODEL", "distilbert-base-uncased")
DATA_PATH = os.getenv("SENTIMENT_TRAIN_CSV", "news_with_sentiment.csv")
OUTPUT_DIR = os.getenv("SENTIMENT_MODEL_DIR", "./sentiment_model")


def load_and_prepare_data(csv_path):
    """
    Загружает данные и преобразует метки в числовой формат.

    :param csv_path: путь к CSV
    :return: (train, test), label_map
    """
    df = pd.read_csv(csv_path)
    if 'content' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV должен содержать колонки 'content' и 'sentiment'.")

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label_map)
    return train_test_split(df, test_size=0.2, random_state=42), label_map


def tokenize_data(tokenizer, dataset):
    """
    Токенизирует датасет.

    :param tokenizer: токенизатор
    :param dataset: HuggingFace Dataset
    :return: токенизированный Dataset
    """
    return dataset.map(lambda x: tokenizer(x['content'], truncation=True, padding='max_length'), batched=True)


def train_sentiment_model(csv_path):
    """
    Обучает модель на датасете по пути.

    :param csv_path: путь к CSV-файлу с данными
    """
    (train_df, val_df), label_map = load_and_prepare_data(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df[['content', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['content', 'label']])

    train_dataset = tokenize_data(tokenizer, train_dataset)
    val_dataset = tokenize_data(tokenizer, val_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("✅ Модель обучена и сохранена в %s", OUTPUT_DIR)


def predict_sentiment(text: str) -> str:
    """
    Выполняет предсказание тональности текста.

    :param text: входной текст
    :return: метка (negative, neutral, positive)
    """
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs).item()

    label_map_reverse = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map_reverse[label_id]


if __name__ == "__main__":
    train_sentiment_model(DATA_PATH)
    test_text = "Биткоин продолжает стремительно расти на фоне интереса институционалов"
    prediction = predict_sentiment(test_text)
    logging.info("Предсказание настроения: %s", prediction)
