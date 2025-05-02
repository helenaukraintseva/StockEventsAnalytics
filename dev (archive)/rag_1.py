from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import torch
import time

start_time = time.time()

# --- 1. Документы sdff---
documents = [
    "Биткоин — это децентрализованная цифровая валюта.",
    "Эфириум позволяет создавать смарт-контракты и dApp-приложения.",
    "Блокчейн — это технология хранения данных в виде цепочки блоков.",
    "Кошелёк используется для хранения криптовалютных активов.",
    "Стейкинг — это способ получать доход от хранения токенов в сети PoS."
]

# --- 2. Создание эмбеддингов для документов ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# --- 3. Индексация (FAISS) ---
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# --- 4. Генеративная модель (для ответа) ---
qa_model = pipeline("text2text-generation", model="sberbank-ai/rugpt3small_based_on_gpt2", max_length=100,
                    device=0 if torch.cuda.is_available() else -1)

# --- 5. Вопрос от пользователя ---
# question = input("❓ Введите вопрос: ")
question = "что такое биткоин"
print("time 1:", round(time.time() - start_time))
# --- 6. Поиск релевантных кусков ---
question_embedding = embedder.encode([question], convert_to_numpy=True)
D, I = index.search(question_embedding, k=2)  # топ-2 релевантных куска

retrieved_docs = "\n".join([documents[i] for i in I[0]])

# --- 7. Составление промпта и генерация ответа ---
prompt = f"Ответь на вопрос на основе текста:\n{retrieved_docs}\n\nВопрос: {question}"
answer = qa_model(prompt)[0]["generated_text"]

print("\n🧠 Ответ:", answer)

question = "что такое Эфириум"
print("time 2:", round(time.time() - start_time))
# --- 6. Поиск релевантных кусков ---
question_embedding = embedder.encode([question], convert_to_numpy=True)
D, I = index.search(question_embedding, k=2)  # топ-2 релевантных куска

retrieved_docs = "\n".join([documents[i] for i in I[0]])

# --- 7. Составление промпта и генерация ответа ---
prompt = f"Ответь на вопрос на основе текста:\n{retrieved_docs}\n\nВопрос: {question}"
answer = qa_model(prompt)[0]["generated_text"]

print("\n🧠 Ответ:", answer)
