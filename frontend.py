import streamlit as st
import requests
import uuid

st.set_page_config(page_title="智能客服", layout="wide")
st.title("🤖 智能客服知识库系统")
st.markdown("支持订单查询、政策咨询、投诉、闲聊等")

API_BASE_URL = "http://localhost:8000"  # 后端地址

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("思考中..."):
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": prompt}
        )
        if resp.status_code == 200:
            data = resp.json()
            answer = data["answer"]
            intent = data["intent"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"意图: {intent}")
        else:
            st.error(f"服务异常: {resp.status_code}")