from dotenv import load_dotenv, find_dotenv
import asyncio

load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI

with open('prompts/scene_change.xml', 'r') as f:
    prompt_scene_change = f.read()


async def main():
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    result = await llm.ainvoke(prompt_scene_change)
    print(result)
    
    with open('export/scene_change.xml', 'w') as f:
        f.write(result.content)


if __name__ == "__main__":
    asyncio.run(main())