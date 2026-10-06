import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm = llm)

data = PyPDFLoader("document/ai.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that summarize text"),
    ("human", "{data}")
])

prompt = template.format_messages(data=docs[0].page_content)

result = model.invoke(prompt)

print(result.content)