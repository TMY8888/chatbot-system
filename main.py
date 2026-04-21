import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from agents import graph

app = FastAPI()

class Query(BaseModel):
    session_id: str
    question: str

@app.post("/chat")
def chat(query: Query):
    state = graph.invoke({
        "session_id": query.session_id,
        "question": query.question,
        "intent": None,
        "context": None,
        "answer": None,
        "history": []
    })
    return {"answer": state["answer"], "intent": state["intent"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)