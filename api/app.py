from fastapi import FastAPI
from pydantic import BaseModel
from src.main import ask_question


app = FastAPI(
    title="Simple RAG Application"
)

class UserRequest(BaseModel):
    question: str


class ModelResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def home():
    return {"message": "Welcome to simple rag application. you can ask about deep learning concept"}

@app.get("/health")
def health():
    return {
        "status_code": 200,
        "Health": "Ok"
    }


@app.post("/ask", response_model=ModelResponse)
def get_response(request: UserRequest):
    response = ask_question(request.question)
    return response