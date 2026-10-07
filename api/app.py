from fastapi import FastAPI, HTTPException, status
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
    try:
        response = ask_question(request.question)
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Model timed out",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error",
        )
