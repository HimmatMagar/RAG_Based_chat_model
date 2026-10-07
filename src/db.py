from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from loader import load_document, split_text

load_dotenv()

PERSIST_DIR = "chroma_db"
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


def BuildDB(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )


def LoadDB():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding
    )

