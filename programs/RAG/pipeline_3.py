import os
import json
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai

from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GEMINI_API = os.getenv("GEMINI_API_KEY")
GEN_QUERY = os.getenv("GEN_QUERY", "Что такое криптовалюта")
JSONL_PATH = os.getenv("JSONL_FILE", "rag_data.jsonl")

genai.configure(api_key=GEMINI_API)
model = genai.GenerativeModel("models/gemini-2.0-flash-lite-001")


def load_jsonl_documents(path):
    """
    Загружает документы из JSONL-файла.
    """
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(Document(
                page_content=record["content"],
                metadata={"title": record["title"], "source": record["source"]}
            ))
    return documents


def main():
    start_time = time.time()
    documents = load_jsonl_documents(JSONL_PATH)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    docs = retriever.invoke(GEN_QUERY)

    if not docs:
        logging.warning("🤷 Ничего не найдено.")
        return

    logging.info("🔎 Найдено %d релевантных фрагментов", len(docs))
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"Контекст:\n{context}\n\nВопрос: {GEN_QUERY}"
    logging.info("⏳ Запрос к Gemini...")
    response = model.generate_content(prompt)

    print("\n🧠 Ответ от Gemini:")
    print(response.text)
    logging.info("⏱️ Время выполнения: %.2f сек.", time.time() - start_time)


if __name__ == "__main__":
    main()
