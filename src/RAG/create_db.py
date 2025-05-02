import logging
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.Client()

    collection = client.create_collection(name="with_embeddings", embedding_function=embedder)
    collection.add(
        documents=["Москва — столица России", "Париж — столица Франции"],
        metadatas=[{"country": "Russia"}, {"country": "France"}],
        ids=["doc1", "doc2"]
    )

    res = collection.query(query_texts=["Столица Франции"], n_results=1)
    logging.info("Результат запроса: %s", res["documents"])

if __name__ == "__main__":
    main()
