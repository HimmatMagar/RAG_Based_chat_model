from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(file_path="document/ai.pdf")
docs = data.load()
print(docs[-1])