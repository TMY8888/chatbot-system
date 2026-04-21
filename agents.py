from retrieval import HybridRetriever
from memory import MemoryManager
import os


class AgentOrchestrator:
    def __init__(self):
        # 初始化 RAG 检索器
        self.retriever = HybridRetriever()
        # 记忆管理器
        self.memory = MemoryManager()

    def _rag_answer(self, query: str) -> str:
        """使用 RAG 检索知识库并生成回答"""
        relevant_chunks = self.retriever.retrieve_with_rerank(query)
        if not relevant_chunks:
            return "抱歉，知识库中没有找到相关信息。"
        context = "\n".join(relevant_chunks)
        # 注意：这里需要 LLM 来生成回答，但我们暂时没有引入 llm
        # 为了简化，直接返回检索到的内容片段
        return f"根据知识库：{context[:200]}..."

    def route_to_agent(self, intent: str, query: str, session_id: str, user_id: str) -> str:
        # 获取会话历史
        history = self.memory.get_session_history(session_id)

        if intent == "order_query":
            answer = "订单123456已发货，预计3天后到达。"
        elif intent == "policy":
            answer = self._rag_answer(query)
        elif intent == "product_qa":
            answer = self._rag_answer(query)
        elif intent == "complaint":
            answer = "很抱歉给您带来不便，我会记录您的投诉并转交专员处理。"
        else:
            answer = self._rag_answer(query)

        # 存储会话历史
        self.memory.add_to_session(session_id, query, answer)
        return answer