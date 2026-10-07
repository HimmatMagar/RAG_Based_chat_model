from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



PERSIST_DIR = "chroma_db"
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def load_document(path):
    data = PyPDFLoader(path)
    docs = data.load()
    return docs


def clean_text(text):
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def split_text(docs, chunk_size, chunk_overlap):
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks



def BuildDB(chunks):
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk.page_content, str):
            raise TypeError(
                f"Chunk {i} contains {type(chunk.page_content)} "
                "instead of str"
            )

    vectorestore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )
    return vectorestore



if __name__ == "__main__":
    docs = load_document("document/deep_learning.pdf")
    chunks = split_text(docs, chunk_size=600, chunk_overlap=100)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    print(chunks[0].page_content[:300])
    BuildDB(chunks)