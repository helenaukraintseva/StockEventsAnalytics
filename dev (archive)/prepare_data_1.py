import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

load_dotenv()

DATA_PATH = os.getenv("TEXT_DATA_PATH", "your_data.txt")

def load_documents(path: str):
    """
    Загружает текстовые документы с помощью LangChain TextLoader.

    :param path: Путь к текстовому файлу
    :return: Список документов
    """
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()
    return documents


if __name__ == "__main__":
    docs = load_documents(DATA_PATH)
    print(f"📄 Загружено документов: {len(docs)}")
