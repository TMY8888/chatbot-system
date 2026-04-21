# memory.py
import uuid
from datetime import datetime
from typing import List, Tuple

class MemoryManager:
    def __init__(self):
        self.session_memory = {}  # session_id -> list of (role, content)

    def get_session_history(self, session_id: str) -> List[Tuple[str, str]]:
        """获取会话历史"""
        return self.session_memory.get(session_id, [])

    def add_to_session(self, session_id: str, user_msg: str, assistant_msg: str):
        """添加对话到会话历史"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = []
        self.session_memory[session_id].append(("user", user_msg))
        self.session_memory[session_id].append(("assistant", assistant_msg))

    def clear_session(self, session_id: str):
        """清除会话历史"""
        if session_id in self.session_memory:
            del self.session_memory[session_id]