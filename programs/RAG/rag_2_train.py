import os
import json
import numpy as np
import faiss
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
EMBED_PATH = os.getenv("EMBED_PATH", "doc_embeddings.npy")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.index")
MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def save_knowledge_base(documents, path):
    """
    Сохраняет список документов в JSON-файл.

    :param documents: Список текстовых строк
    :param path: Путь к файлу JSON
    :return: None
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logging.info("🧠 База знаний сохранена: %s", path)


def embed_documents(documents, model_name):
    """
    Векторизует документы с помощью SentenceTransformer.

    :param documents: Список текстов
    :param model_name: Название модели SentenceTransformer
    :return: Массив эмбеддингов
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, convert_to_numpy=True)
    return embeddings


def save_embeddings_and_index(embeddings, embed_path, index_path):
    """
    Сохраняет эмбеддинги в .npy и индекс в FAISS-формате.

    :param embeddings: np.array эмбеддингов
    :param embed_path: Путь для сохранения .npy
    :param index_path: Путь для FAISS-индекса
    :return: None
    """
    np.save(embed_path, embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    logging.info("💾 Эмбеддинги и индекс сохранены.")


if __name__ == "__main__":
    # 1. База знаний
    knowledge_base = [
        "Биткойн — это децентрализованная цифровая валюта.",
        "Ethereum поддерживает смарт-контракты.",
        "Блокчейн — это распределённый реестр данных.",
        "Майнинг — процесс создания новых блоков.",
        "Криптовалюта не контролируется государством."
    ]

    save_knowledge_base(knowledge_base, KB_PATH)

    # 2. Векторизация
    embeddings = embed_documents(knowledge_base, MODEL_NAME)

    # 3. Сохранение эмбеддингов и индекса
    save_embeddings_and_index(embeddings, EMBED_PATH, FAISS_INDEX_PATH)

    logging.info("✅ Обучение завершено: всё сохранено.")
