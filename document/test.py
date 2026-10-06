from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter


# splitter = CharacterTextSplitter(
#     chunk_size = 10,
#     chunk_overlap = 1
# )

# splitter = TokenTextSplitter(
#     chunk_size = 30,
#     chunk_overlap = 2
# )

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 40,
    chunk_overlap = 5
)
data = TextLoader("document/text.txt")
docs = data.load()
chunk = splitter.split_documents(docs)
print(chunk)