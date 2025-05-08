import streamlit as st
from llm import ChatBot
from langchain_core.messages import HumanMessage, AIMessage

# 设置页面配置
st.set_page_config(
    page_title="AI 聊天助手",
    page_icon="🤖",
    layout="centered"
)

# 初始化会话状态
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatBot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# 页面标题
st.title("🤖 AI 聊天助手")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 获取AI响应
    with st.chat_message("assistant"):
        try:
            # 创建一个空的占位符
            message_placeholder = st.empty()
            full_response = ""
            
            # 流式获取响应
            for chunk in st.session_state.chatbot.chat_stream(prompt):
                full_response += chunk
                # 更新占位符内容
                message_placeholder.markdown(full_response + "▌")
            
            # 最终更新，移除光标
            message_placeholder.markdown(full_response)
            
            # 添加AI响应到聊天历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.session_state.messages.pop()  # 移除失败的用户消息 