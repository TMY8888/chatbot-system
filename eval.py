import json
from agents import graph
from tqdm import tqdm

# 测试集
test_cases = [
    {"question": "我的订单12345什么时候到？", "expected_intent": "order_query", "expected_tool": "query_order_status"},
    {"question": "电子产品退货政策是什么？", "expected_intent": "policy", "expected_tool": "get_return_policy"},
    {"question": "北京配送要多久？", "expected_intent": "policy", "expected_tool": "check_delivery_time"},
    {"question": "你们的产品真垃圾，我要投诉！", "expected_intent": "complaint", "expected_tool": None},
    {"question": "如何设置手环的心率监测？", "expected_intent": "faq", "expected_tool": None},
]

def run_eval():
    correct_intent = 0
    correct_tool = 0
    for case in tqdm(test_cases):
        state = graph.invoke({
            "session_id": "test",
            "question": case["question"],
            "intent": None,
            "context": None,
            "answer": None,
            "history": []
        })
        if state["intent"] == case["expected_intent"]:
            correct_intent += 1
        if case["expected_tool"] is None and not state.get("tool_result"):
            correct_tool += 1
        elif case["expected_tool"] and state.get("tool_result"):
            correct_tool += 1
    print(f"意图识别准确率: {correct_intent/len(test_cases)*100:.1f}%")
    print(f"工具调用准确率: {correct_tool/len(test_cases)*100:.1f}%")

if __name__ == "__main__":
    run_eval()