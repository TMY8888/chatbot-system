import redis
import json
from datetime import datetime
import os


class MemoryManager:
    def __init__(self):
        self.redis_client = None
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        except:
            self.redis_client = None
            self.memory_store = {}

    def get_session_history(self, session_id: str) -> list:
        key = f"session:{session_id}"
        if self.redis_client:
            data = self.redis_client.lrange(key, -10, -1)  # 最近10条
            return [json.loads(item) for item in data]
        else:
            return self.memory_store.get(key, [])

    def add_to_session(self, session_id: str, user_msg: str, bot_msg: str):
        key = f"session:{session_id}"
        record = {"user": user_msg, "bot": bot_msg, "timestamp": datetime.now().isoformat()}
        if self.redis_client:
            self.redis_client.rpush(key, json.dumps(record))
            self.redis_client.expire(key, 3600)  # 1小时过期
        else:
            if key not in self.memory_store:
                self.memory_store[key] = []
            self.memory_store[key].append(record)

    def set_user_preference(self, user_id: str, key: str, value: str):
        pref_key = f"pref:{user_id}"
        if self.redis_client:
            self.redis_client.hset(pref_key, key, value)
        else:
            pass  # 内存版本简化

    def get_user_preference(self, user_id: str, key: str) -> str:
        pref_key = f"pref:{user_id}"
        if self.redis_client:
            return self.redis_client.hget(pref_key, key)
        return None