from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(path):
    data = PyPDFLoader(path)
    docs = data.load()
    return docs


def split_text(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks



if __name__ == "__main__":
    docs = load_document("document/deep_learning.pdf")
    chunks = split_text(docs, chunk_size=600, chunk_overlap=100)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    print(chunks[0].page_content[:300])