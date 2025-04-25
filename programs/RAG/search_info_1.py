import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def search_documents(db, query: str, top_k: int = 3):
    """
    Выполняет семантический поиск по базе векторных документов.

    :param db: Объект векторной базы (например, FAISS, Chroma)
    :param query: Запрос пользователя
    :param top_k: Количество релевантных документов
    :return: Список релевантных документов
    """
    results = db.similarity_search(query, k=top_k)
    logging.info("🔍 Найдено %d релевантных документов.", len(results))
    return results


if __name__ == "__main__":
    # Предполагается, что db инициализирован ранее
    query = "Что такое RAG модель?"
    docs = search_documents(db, query)
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Документ {i}:\n{doc.page_content}")
