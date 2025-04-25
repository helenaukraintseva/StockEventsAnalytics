import chromadb
from chromadb.utils import embedding_functions

# Инициализация клиента (по умолчанию — локально)
client = chromadb.Client()

# Создаем коллекцию
collection = client.create_collection(name="my_collection")

# Добавим эмбеддинги (для примера - просто числа)
collection.add(
    documents=["Привет, как дела?", "Сегодня хорошая погода", "Погода ужасная", "Привет, чем занят?"],
    metadatas=[{"source": "chat"}] * 4,
    ids=["id1", "id2", "id3", "id4"]
)

# Поиск по коллекции — передаем текст, Chroma сам генерирует эмбеддинг
results = collection.query(
    query_texts=["Привет!"],
    n_results=2
)

print("Найденные документы:")
for doc in results['documents'][0]:
    print("-", doc)
