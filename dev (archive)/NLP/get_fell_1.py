import os
import time
import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_sentiment(text: str):
    """
    Анализирует тональность текста с использованием модели Twitter-RoBERTa.

    :param text: Строка текста для анализа
    :return: Класс (строка) и вероятности классов
    """
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]

    labels = ['Negative', 'Neutral', 'Positive']
    return labels[np.argmax(scores)], scores


if __name__ == "__main__":
    start = time.time()
    text = os.getenv("SENTIMENT_TEXT", "Цена биткоина резко упала за сутки на 10 процентов.")
    label, probs = get_sentiment(text)
    logging.info("Настроение: %s | Вероятности: %s", label, probs)
    logging.info("Затраченное время: %.2f сек.", time.time() - start)
