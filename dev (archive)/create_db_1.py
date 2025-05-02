from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="with_embeddings", embedding_function=embedder)

collection.add(
    documents=["Москва — столица России", "Париж — столица Франции"],
    metadatas=[{"country": "Russia"}, {"country": "France"}],
    ids=["doc1", "doc2"]
)

res = collection.query(query_texts=["Столица Франции"], n_results=1)
print(res["documents"])
