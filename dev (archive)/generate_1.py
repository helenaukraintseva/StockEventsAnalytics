from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

response = qa.run("Что такое RAG модель?")
print(response)