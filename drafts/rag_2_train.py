# rag_train.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- 1. Загружаем или создаём базу знаний ---
documents = [
    "Биткойн — это децентрализованная цифровая валюта.",
    "Ethereum поддерживает смарт-контракты.",
    "Блокчейн — это распределённый реестр данных.",
    "Майнинг — процесс создания новых блоков.",
    "Криптовалюта не контролируется государством."
]

# --- 2. Сохраняем базу знаний ---
with open("knowledge_base.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

# --- 3. Векторизуем документы ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True)

# --- 4. Сохраняем эмбеддинги и индекс ---
np.save("doc_embeddings.npy", embeddings)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")

print("✅ Обучение завершено: база знаний, эмбеддинги и индекс сохранены.")
