from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import time

time_start = time.time()

def get_sentiment(text):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]

    labels = ['Negative', 'Neutral', 'Positive']
    return labels[np.argmax(scores)], scores

text = "Цена биткоина резко упала за сутки на 10 процентов."
label, probs = get_sentiment(text)
res = f"{round(float(probs[0]), 2)}"
print(f"Total time: {round(time.time() - time_start, 2)}")
print(f"Настроение: {label} | Вероятности: {probs}")
print(res)

