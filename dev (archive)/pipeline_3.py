import os
import time
import google.generativeai as genai
from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from config import gemini_api
import json
from langchain_core.documents import Document

def load_jsonl_documents(path):
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(Document(
                page_content=record["content"],
                metadata={"title": record["title"], "source": record["source"]}
            ))
    return documents

# === Конфигурация Gemini ===
genai.configure(api_key=gemini_api)
model = genai.GenerativeModel("models/gemini-2.0-flash-lite-001")

start_time = time.time()

# === Шаг 1: Загрузка текста ===
documents = load_jsonl_documents("rag_data.jsonl")

# === Шаг 2: Разбиение на чанки ===
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# === Шаг 3: Эмбеддинги (локальные) ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Шаг 4: Индексация FAISS ===
vectorstore = FAISS.from_documents(chunks, embeddings)

# === Шаг 5: Поиск ===
retriever = vectorstore.as_retriever()
query = "Что такое криптовалюта"
docs = retriever.invoke(query)

# === Шаг 6: Вывод и генерация через Gemini ===
if not docs:
    print("🤷 Ничего не найдено.")
else:
    print(f"\n🔎 Найдено {len(docs)} релевантных фрагментов:")
    context = ""
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Фрагмент {i}:\n{doc.page_content}")
        context += doc.page_content + "\n"

    # === Шаг 7: Запрос к Gemini ===
    prompt = f"Контекст:\n{context}\n\nВопрос: {query}"
    print("\n⏳ Отправка запроса к Gemini...")
    response = model.generate_content(prompt)
    print("\n🧠 Ответ от Gemini:")
    print(response.text)

print("\n⏱️ Время выполнения:", round(time.time() - start_time, 2), "сек.")
