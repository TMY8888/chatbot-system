import os
import uuid
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from loguru import logger
import uvicorn
from dotenv import load_dotenv

from intent import IntentClassifier
from agents import AgentOrchestrator
from memory import MemoryManager

load_dotenv()

# 初始化组件
intent_classifier = IntentClassifier()
orchestrator = AgentOrchestrator()
memory_manager = MemoryManager()

app = FastAPI(title="智能客服知识库系统", version="2.0.0")


# ---------- 日志中间件 ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} | {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response {request_id} | Status {response.status_code}")
    return response


# ---------- 请求/响应模型 ----------
class ChatRequest(BaseModel):
    session_id: str = None  # 可选，如果不提供则创建新会话
    user_id: str = "anonymous"
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_req: ChatRequest):
    # 生成或使用已有session_id
    session_id = chat_req.session_id or str(uuid.uuid4())
    user_id = chat_req.user_id
    query = chat_req.message

    # 1. 意图识别
    intent = intent_classifier.classify(query)
    logger.info(f"Session {session_id} | Intent: {intent}")

    # 2. 路由到对应Agent
    answer = orchestrator.route_to_agent(intent, query, session_id, user_id)

    # 3. 可选：记录用户偏好（长期记忆）
    if intent == "order_query" and "订单号" in query:
        # 简单提取订单号（演示）
        import re
        match = re.search(r'\d{6}', query)
        if match:
            memory_manager.set_user_preference(user_id, "last_order", match.group())

    return ChatResponse(session_id=session_id, answer=answer, intent=intent)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = memory_manager.get_session_history(session_id)
    return {"session_id": session_id, "history": history}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)