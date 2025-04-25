# rag_ask_fixed.py

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time

start_time = time.time()

# --- 1. Загрузка базы знаний и эмбеддингов ---
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

embeddings = np.load("doc_embeddings.npy")
index = faiss.read_index("faiss_index.index")

# --- 2. Загрузка моделей ---
print("🔄 Загружаем SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Загружаем генеративную модель...")
qa_model = pipeline(
    "text2text-generation",
    model="./rut5-small",
    tokenizer="./rut5-small",
    tokenizer_kwargs={"use_fast": False},  # 🔧 Ключевая правка
    device=-1  # Или 0, если есть CUDA
)

# --- 3. Функция поиска и генерации ответа ---
def answer_question(question, top_k=3):
    question_embedding = embedder.encode([question])
    _, indices = index.search(question_embedding, top_k)
    context = "\n".join([documents[i] for i in indices[0]])

    prompt = f"Вопрос: {question}\nКонтекст: {context}\nОтвет:"
    result = qa_model(prompt, max_length=100, do_sample=False)[0]['generated_text']
    return result.strip()

# --- 4. Пример использования ---
user_question = "Что такое блокчейн?"
response = answer_question(user_question)

print("🧠 Ответ:", response)
print(f"⏱️ Total time: {round(time.time() - start_time, 2)} сек.")
