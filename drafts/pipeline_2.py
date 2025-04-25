import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import time
start_time = time.time()

# === Шаг 1: Загрузка текста ===
loader = TextLoader("text_1.txt", encoding="utf-8")
documents = loader.load()

# === Шаг 2: Разбиение на чанки ===
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# === Шаг 3: Эмбеддинги (локальные) ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Шаг 4: Индексация FAISS ===
vectorstore = FAISS.from_documents(chunks, embeddings)

# === Шаг 5: Поиск и вывод без генерации ===
retriever = vectorstore.as_retriever()
print("Time 1:", round(time.time() - start_time, 2))
# while True:
    # query = input("\n❓ Введите вопрос (или 'exit' для выхода): ")
query = "Что такое RAG"
# if query.lower() in ["exit", "выход", "quit"]:
#     break

docs = retriever.get_relevant_documents(query)

if not docs:
    print("🤷 Ничего не найдено.")
else:
    print(f"\n🔎 Найдено {len(docs)} релевантных фрагментов:")
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Фрагмент {i}:\n{doc.page_content}")

print("Time 2:", round(time.time() - start_time, 2))