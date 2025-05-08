import streamlit as st
from llm import state_generation_chain, guiding_questions_chain, story_generation_chain, GeneratedStates, GuidedQuestions, QuestionWithOptions, FinalStory # 修改: 导入新内容
import asyncio
from langchain.chat_models import ChatOpenAI

# 设置页面标题
st.set_page_config(page_title="无限冒险剧本生成器", layout="wide")

st.title("🎲 无限冒险剧本生成器")
st.caption("根据随机生成的起点和终点，通过一系列选择来构建你自己的冒险故事！")

# 初始化 session state
if 'initial_state' not in st.session_state:
    st.session_state.initial_state = ""
if 'final_state' not in st.session_state:
    st.session_state.final_state = ""
if 'guided_questions' not in st.session_state:
    st.session_state.guided_questions = [] # List[QuestionWithOptions]
if 'user_choices' not in st.session_state:
    st.session_state.user_choices = {} # Dict[str, str] e.g. {"user_choice_0": "selected_option_for_q0"}
if 'final_story' not in st.session_state: # 新增
    st.session_state.final_story = ""
if 'story_generated' not in st.session_state:
    st.session_state.story_generated = False
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False

async def generate_states_and_questions():
    """生成初始/结束状态，然后生成引导问题"""
    try:
        # 重置状态
        st.session_state.initial_state = ""
        st.session_state.final_state = ""
        st.session_state.guided_questions = []
        st.session_state.user_choices = {}
        st.session_state.final_story = ""
        st.session_state.story_generated = False
        st.session_state.questions_generated = False

        with st.spinner("正在生成初始和结束状态..."):
            generated_states: GeneratedStates = await state_generation_chain.ainvoke({})
            st.session_state.initial_state = generated_states.initial_state
            st.session_state.final_state = generated_states.final_state

        if st.session_state.initial_state and st.session_state.final_state:
            with st.spinner("正在生成引导问题..."):
                questions_data: GuidedQuestions = await guiding_questions_chain.ainvoke({
                    "initial_state": st.session_state.initial_state,
                    "final_state": st.session_state.final_state
                })
                st.session_state.guided_questions = questions_data.questions
                # 初始化用户的选择字典的key，确保后续能直接赋值
                for i in range(len(st.session_state.guided_questions)):
                    st.session_state.user_choices[f"user_choice_{i}"] = None 
                st.session_state.questions_generated = True
        else:
            st.error("未能成功生成初始或结束状态，请重试。")

    except Exception as e:
        st.error(f"生成过程中发生错误: {e}")
        st.session_state.questions_generated = False

async def generate_the_final_story():
    """根据用户选择生成最终剧本"""
    if not st.session_state.initial_state or not st.session_state.final_state or not st.session_state.user_choices:
        st.warning("请先生成初始设定并回答所有问题。")
        return

    # 检查用户是否已回答所有问题 (简单的检查方式，可以根据需要完善)
    # GuidedQuestions 固定为5个问题
    if len(st.session_state.user_choices) < 5 or any(v is None for v in st.session_state.user_choices.values()):
        # 确保guided_questions 也被正确加载了
        if len(st.session_state.guided_questions) == 5 and len(st.session_state.user_choices) < len(st.session_state.guided_questions):
            st.warning("请回答所有的引导问题后再生成剧本。")
            return
        elif any(st.session_state.user_choices.get(f"user_choice_{i}") is None for i in range(len(st.session_state.guided_questions))):
             st.warning("请回答所有的引导问题后再生成剧本。")
             return

    try:
        with st.spinner("正在融合您的选择，创作最终剧本..."):
            choices_for_prompt = {
                f"user_choice_{i}": st.session_state.user_choices.get(f"user_choice_{i}", "[未选择]") 
                for i in range(len(st.session_state.guided_questions)) # 使用实际问题数量
            }
            payload = {
                "initial_state": st.session_state.initial_state,
                "final_state": st.session_state.final_state,
                **choices_for_prompt
            }
            final_story_data: FinalStory = await story_generation_chain.ainvoke(payload)
            st.session_state.final_story = final_story_data.story
            st.session_state.story_generated = True
            st.balloons() # 庆祝一下
    except Exception as e:
        st.error(f"生成最终剧本时发生错误: {e}")

# --- 按钮和界面布局 ---
col_button1, col_button2 = st.columns(2)

with col_button1:
    if st.button("✨ 生成剧本初始设定", type="primary", use_container_width=True, disabled=st.session_state.questions_generated and not st.session_state.story_generated):
        asyncio.run(generate_states_and_questions())

with col_button2:
    # 只有在问题生成后且故事尚未生成时，才启用生成最终剧本按钮
    if st.button("📜 生成最终剧本", type="primary", use_container_width=True, disabled=not st.session_state.questions_generated or st.session_state.story_generated):
        asyncio.run(generate_the_final_story())

if st.session_state.story_generated: # 提供一个重新开始的选项
    if st.button("🔄 重新开始一段新冒险", use_container_width=True):
        # 重置所有相关状态以重新开始
        st.session_state.initial_state = ""
        st.session_state.final_state = ""
        st.session_state.guided_questions = []
        st.session_state.user_choices = {}
        st.session_state.final_story = ""
        st.session_state.story_generated = False
        st.session_state.questions_generated = False
        st.rerun() # 重新运行脚本以刷新界面

if st.session_state.questions_generated and not st.session_state.story_generated:
    st.subheader("🚀 你的冒险起点和可能的终点：")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**初始状态:** {st.session_state.initial_state}")
    with col2:
        st.warning(f"**结束状态:** {st.session_state.final_state}")

    st.subheader("🗺️ 请通过以下问题引导剧本创作：")
    
    for i, q_item in enumerate(st.session_state.guided_questions):
        question_text = f"{i+1}. {q_item.question}"
        options = [opt for opt in q_item.options]
        choice_key = f"user_choice_{i}"
        
        # 从 session_state 获取当前为此问题保存的答案
        current_choice_val = st.session_state.user_choices.get(choice_key)
        current_choice_idx = options.index(current_choice_val) if current_choice_val in options else 0

        selected_option = st.radio(
            question_text,
            options,
            key=f"radio_{choice_key}", # 确保key是唯一的，并且在重新生成问题时能正确更新
            index=current_choice_idx,
        )
        st.session_state.user_choices[choice_key] = selected_option

if st.session_state.story_generated and st.session_state.final_story:
    st.subheader("🎉 恭喜！你的冒险剧本已生成：")
    st.markdown(st.session_state.final_story)
    st.download_button(
        label="📥 下载剧本 (.txt)",
        data=st.session_state.final_story,
        file_name="我的冒险剧本.txt",
        mime="text/plain"
    )


# 添加一些说明和页脚
st.markdown("---")
st.markdown("由 Langchain 和 Streamlit 驱动 | 一个AI剧本小助手")