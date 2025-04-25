import os
import time
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TEXT_FILE = os.getenv("PIPELINE_TEXT_FILE", "text_1.txt")
QUERY = os.getenv("PIPELINE_QUERY", "Что такое RAG")

def main():
    start_time = time.time()
    logging.info("📄 Загрузка документа: %s", TEXT_FILE)
    loader = TextLoader(TEXT_FILE, encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    docs = retriever.get_relevant_documents(QUERY)

    if not docs:
        logging.warning("🤷 Ничего не найдено.")
    else:
        logging.info("🔎 Найдено %d релевантных фрагментов:", len(docs))
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 Фрагмент {i}:\n{doc.page_content}")

    logging.info("⏱️ Время выполнения: %.2f сек.", time.time() - start_time)

if __name__ == "__main__":
    main()
