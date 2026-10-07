from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vectorStore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

def retrive_document(query: str, k: int = 3):
    retriver = vectorStore.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k": k,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    document = retriver.invoke(query)
    return document
