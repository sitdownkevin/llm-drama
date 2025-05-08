from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Iterator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# 加载环境变量
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "对话历史"]
    next: str

class ChatBot:
    def __init__(self):
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            streaming=True  # 启用流式输出
        )
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的AI助手。"),
            ("human", "{input}"),
            ("ai", "{chat_history}")
        ])
        
        # 创建链
        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 创建 LangGraph
        self.workflow = self._create_graph()
    
    def _create_graph(self):
        # 定义节点
        def chat(state: AgentState) -> AgentState:
            # 获取最后一条用户消息
            last_message = state["messages"][-1]
            # 获取历史消息
            history = state["messages"][:-1]
            
            # 准备输入
            chain_input = {
                "input": last_message.content,
                "chat_history": "\n".join([msg.content for msg in history])
            }
            
            # 获取响应
            response = self.chain.invoke(chain_input)
            
            # 更新状态
            state["messages"].append(AIMessage(content=response))
            return state
        
        # 创建图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("chat", chat)
        
        # 设置边
        workflow.set_entry_point("chat")
        workflow.add_edge("chat", END)
        
        # 编译图
        return workflow.compile()
    
    def chat(self, user_input: str) -> str:
        # 初始化状态
        state = {
            "messages": [HumanMessage(content=user_input)],
            "next": "chat"
        }
        
        # 运行图
        result = self.workflow.invoke(state)
        
        # 返回最后一条消息
        return result["messages"][-1].content
    
    def chat_stream(self, user_input: str) -> Iterator[str]:
        """流式聊天方法"""
        # 获取最后一条用户消息
        last_message = HumanMessage(content=user_input)
        # 获取历史消息
        history = []  # 这里可以添加历史消息的处理
        
        # 准备输入
        chain_input = {
            "input": last_message.content,
            "chat_history": "\n".join([msg.content for msg in history])
        }
        
        # 流式获取响应
        for chunk in self.chain.stream(chain_input):
            yield chunk

# 测试函数
def main():
    chatbot = ChatBot()
    response = chatbot.chat("你好，请介绍一下你自己。")
    print(response)

if __name__ == "__main__":
    main()
