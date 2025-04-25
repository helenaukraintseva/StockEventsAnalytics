from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from typing import List, Dict  # ✅ Используем typing для совместимости с Python < 3.9

# Создание хранилища документов
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

# Подготовка документов
docs: List[Dict[str, str]] = [
    {"content": "Python — это язык программирования с открытым исходным кодом."},
    {"content": "RAG — это архитектура, которая объединяет поиск и генерацию текста."},
    {"content": "Haystack — это библиотека для создания систем вопрос-ответ."}
]

# Загрузка документов в хранилище
document_store.write_documents(docs)

# Ретривер (поисковик)
retriever = BM25Retriever(document_store=document_store)

# Читатель (LLM, отвечает на вопросы)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Создание пайплайна
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Вопрос пользователя
question: str = "Что такое RAG?"

# Запуск пайплайна
prediction = pipeline.run(
    query=question,
    params={
        "Retriever": {"top_k": 2},
        "Reader": {"top_k": 1}
    }
)

# Вывод ответа
print("Ответ:", prediction["answers"][0].answer)
