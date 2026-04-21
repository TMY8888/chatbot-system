from langchain.tools import tool
import requests
import json

# 模拟订单查询工具
@tool
def query_order_status(order_id: str) -> str:
    """根据订单号查询订单状态（已发货、配送中、已签收等）"""
    # 模拟数据库查询
    mock_orders = {
        "12345": "已发货，预计明天送达",
        "67890": "已签收，签收时间：2026-04-19",
        "11111": "处理中，请耐心等待"
    }
    return mock_orders.get(order_id, "未找到该订单，请确认订单号")

@tool
def get_return_policy(product_category: str) -> str:
    """查询不同品类的退换货政策"""
    policies = {
        "电子产品": "7天无理由退货，15天内质量问题换货",
        "服装": "30天无理由退货，需保持吊牌完好",
        "食品": "不支持无理由退货，质量问题请联系客服"
    }
    return policies.get(product_category, "请联系人工客服咨询具体政策")

@tool
def check_delivery_time(city: str) -> str:
    """查询配送时效"""
    times = {
        "北京": "次日达",
        "上海": "次日达",
        "广州": "2-3天",
        "深圳": "2-3天"
    }
    return times.get(city, "3-5天，具体以物流为准")