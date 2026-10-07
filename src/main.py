import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from retriver import retrive_document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm= llm)

def ask_question(question: str):
    document = retrive_document(question)
    context = "\n\n".join(document.page_content for document in document)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
            Use only provided context to give the answer.
            if answer is not present in context.
            say: I am unable to find the answer in the context"""
        ),
        ("human", """
            context: {context},
            question: {question}
        """)
    ])

    message = prompt.invoke({
        "context": context,
        "question": question
    })

    response = model.invoke(message)
    return {
        "answer": response.content,
        "sources": [
            doc.page_content
            for doc in document
        ]
    }

while True:
    query = input("User: ")
    if query == "exit":
        break

    response = ask_question(query)
    print(response.answer)