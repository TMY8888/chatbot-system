import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate


class IntentClassifier:
    def __init__(self):
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        self.llm = ChatZhipuAI(
            model="glm-4-flash",
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个意图分类器。请将用户的问题分类为以下之一：order_query（订单查询）、policy（政策咨询）、complaint（投诉）、chat（闲聊）、product_qa（产品问答）。只输出类别名称，不要输出其他内容。"),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm

    def classify(self, query: str) -> str:
        response = self.chain.invoke({"query": query})
        return response.content.strip()