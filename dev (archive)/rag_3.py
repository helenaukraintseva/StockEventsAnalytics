import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Пути к файлам ---
EMBEDDINGS_PATH = "doc_embeddings.npy"
INDEX_PATH = "faiss_index.index"
KNOWLEDGE_PATH = "knowledge_base.json"

# --- Определение устройства ---
use_cuda = os.environ.get("CUDA_VISIBLE_DEVICES") is not None
device_id = 0 if use_cuda else -1

# --- Загрузка моделей ---
print("🔄 Загружаем SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if use_cuda else "cpu")

print("🔄 Загружаем генеративную модель...")
qa_model = pipeline(
    "text2text-generation",
    model="./rut5-small",
    tokenizer="./rut5-small",
    tokenizer_kwargs={"use_fast": False},  # 🔥 ВАЖНО: Отключаем fast токенизатор!
    device=device_id
)

# --- Загрузка базы знаний и FAISS индекса ---
print("📚 Загружаем базу знаний...")
with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

print("📦 Загружаем эмбеддинги и индекс...")
doc_embeddings = np.load(EMBEDDINGS_PATH)
index = faiss.read_index(INDEX_PATH)

# --- Функция для генерации ответа ---
def answer_question(question: str, top_k: int = 3, max_ctx_len: int = 300, debug: bool = False) -> str:
    question_embedding = embedder.encode([question])
    _, indices = index.search(question_embedding, top_k)

    # Подрезаем контекст (если нужен)
    context_chunks = [documents[i][:max_ctx_len] for i in indices[0]]
    context = "\n".join(context_chunks)

    if debug:
        print("\n🔍 Вопрос:", question)
        print("📚 Контекст:\n", context)
        return "(Режим отладки — генерация отключена)"

    prompt = f"Вопрос: {question}\nКонтекст: {context}\nОтвет:"
    result = qa_model(prompt, max_length=100, do_sample=False)[0]['generated_text']
    return result.strip()

# --- Интерактивный режим ---
if __name__ == "__main__":
    print("\n🤖 Добро пожаловать в RAG-систему! Напиши 'выход' для завершения.\n")

    while True:
        user_input = input("Вопрос: ").strip()
        if user_input.lower() in ["выход", "exit", "quit"]:
            print("\n👋 До встречи!")
            break

        response = answer_question(user_input)
        print("🧠 Ответ:", response, "\n")
