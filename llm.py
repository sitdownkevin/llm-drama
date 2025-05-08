from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Iterator, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import re

# 加载环境变量
load_dotenv()

# 定义LLM返回的结构化数据模型
class QuestionWithOptions(BaseModel):
    question: str = Field(description="向用户提出的问题")
    options: List[str] = Field(description="提供给用户的选项列表，至少2个选项")

class GuidedQuestions(BaseModel):
    questions: List[QuestionWithOptions] = Field(description="引导用户生成剧本的5个问题列表")

# 新增：定义生成初始和结束状态的Pydantic模型
class GeneratedStates(BaseModel):
    initial_state: str = Field(description="冒险游戏开始时的初始状态描述")
    final_state: str = Field(description="冒险游戏可能达到的一个结束状态描述")

# 接下来我们可以定义使用这些模型的LLM链

# 1. 初始化 OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7) # 修改: 指定模型为 gpt-4o

# 2. 创建 PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=GuidedQuestions)

# 3. 创建提示模板
prompt_template = """
你是一个游戏剧本创作助手。
用户的目标是创作一个冒险游戏的剧本，该剧本需要联结两个已知的游戏状态。

初始状态: {initial_state}
结束状态: {final_state}

请你生成5个引导性的问题，每个问题提供至少2个选项，帮助用户一步步构建这个剧本。
确保这些问题和选项能够自然地引导用户从初始状态过渡到结束状态。

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(
    template=prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 创建并连接LLM调用链
guiding_questions_chain = prompt | llm | parser

# --- 新增：生成初始和结束状态的LLM链 ---

# 1. 初始化 LLM (复用之前的llm实例)

# 2. 创建 PydanticOutputParser for GeneratedStates
states_parser = PydanticOutputParser(pydantic_object=GeneratedStates)

# 3. 创建提示模板 for GeneratedStates
states_prompt_template = """
你是一个富有想象力的游戏设定生成器。
请为用户的无限流冒险游戏生成两个随机且有趣的的游戏状态：一个是初始状态，一个是潜在的结束状态。
这两个状态应该能够激发有趣的故事情节。

确保状态描述简洁且引人入胜。

{format_instructions}
"""

states_prompt = ChatPromptTemplate.from_template(
    template=states_prompt_template,
    partial_variables={"format_instructions": states_parser.get_format_instructions()}
)

# 4. 创建并连接LLM调用链 for GeneratedStates
state_generation_chain = states_prompt | llm | states_parser

# --- 新增：根据用户选择生成最终剧本的LLM链 ---

# 1. 定义最终剧本的Pydantic模型
class FinalStory(BaseModel):
    story: str = Field(description="根据初始状态、结束状态和用户选择的五个情节片段串联起来的完整冒险剧本")

# 2. 创建 PydanticOutputParser for FinalStory
story_parser = PydanticOutputParser(pydantic_object=FinalStory)

# 3. 创建提示模板 for FinalStory
# 用户选择的格式将会是: {"user_choice_0": "选项A", "user_choice_1": "选项B", ...}
# 我们需要将这些选择在提示中清晰地列出来
story_prompt_template = """
你是一位才华横溢的剧作家。
请根据以下信息创作一个连贯、引人入胜的冒险游戏短篇剧本：

初始状态: 
{initial_state}

结束状态: 
{final_state}

用户选择的五个关键情节转折点是：
1. {user_choice_0}
2. {user_choice_1}
3. {user_choice_2}
4. {user_choice_3}
5. {user_choice_4}

请将这些元素巧妙地编织进一个完整的故事中，确保故事流畅，逻辑清晰，并能够从初始状态自然发展到结束状态。
故事应该生动有趣，富有冒险色彩。

{format_instructions}
"""

story_prompt = ChatPromptTemplate.from_template(
    template=story_prompt_template,
    partial_variables={"format_instructions": story_parser.get_format_instructions()}
)

# 4. 创建并连接LLM调用链 for FinalStory
# 注意：这里的llm实例是复用的
story_generation_chain = story_prompt | llm | story_parser


# 示例用法 (这部分代码后续会整合到streamlit应用中)
# async def generate_final_story(initial_state: str, final_state: str, user_choices: dict) -> FinalStory:
#     # 从 user_choices 字典中提取选项值，确保顺序正确
#     # user_choices 的 key 是类似 'user_choice_0', 'user_choice_1' ...
#     choices_for_prompt = {
#         f"user_choice_{i}": user_choices.get(f"user_choice_{i}", "[未选择]") for i in range(5)
#     }
#     payload = {
#         "initial_state": initial_state,
#         "final_state": final_state,
#         **choices_for_prompt
#     }
#     return await story_generation_chain.ainvoke(payload)


# # 用于测试的异步函数运行 (实际使用时不需要在llm.py中直接运行)
# async def main_test_full_flow():
#     # 1. 生成状态
#     print("--- 生成状态 ---")
#     generated_states = await generate_initial_final_states()
#     print(f"初始状态: {generated_states.initial_state}")
#     print(f"结束状态: {generated_states.final_state}")
#     
#     if not (generated_states.initial_state and generated_states.final_state):
#         print("状态生成失败，测试中止")
#         return
# 
#     # 2. 生成问题
#     print("\n--- 生成引导问题 ---")
#     questions_data = await generate_questions(generated_states.initial_state, generated_states.final_state)
#     if not questions_data.questions:
#         print("问题生成失败，测试中止")
#         return
#         
#     dummy_user_choices = {}
#     for i, q_item in enumerate(questions_data.questions):
#         print(f"问题 {i+1}: {q_item.question}")
#         for j, option in enumerate(q_item.options):
#             print(f"  选项 {j+1}: {option}")
#         # 假设用户选择了每个问题的第一个选项
#         if q_item.options:
#             dummy_user_choices[f"user_choice_{i}"] = q_item.options[0]
#         else:
#             dummy_user_choices[f"user_choice_{i}"] = "[默认选项]"
#     print(f"\n模拟用户选择: {dummy_user_choices}")
# 
#     # 3. 生成最终剧本
#     print("\n--- 生成最终剧本 ---")
#     final_story_data = await generate_final_story(generated_states.initial_state, generated_states.final_state, dummy_user_choices)
#     print(f"\n最终剧本:\n{final_story_data.story}")
#
# if __name__ == '__main__':
#     import asyncio
#     # asyncio.run(main_test_states()) # 测试状态和问题生成
#     asyncio.run(main_test_full_flow()) # 测试完整流程