from langchain.document_loaders import TextLoader

loader = TextLoader("your_data.txt")
documents = loader.load()
