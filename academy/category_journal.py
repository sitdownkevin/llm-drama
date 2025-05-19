from dotenv import load_dotenv, find_dotenv
import asyncio
import json
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_fixed
import os
import logging

load_dotenv(find_dotenv())

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.5,
)


response_schemas = [
    ResponseSchema(name="title", description="The title of the journal"),
    ResponseSchema(name="issn", description="The ISSN of the journal", type='string'),
    ResponseSchema(name="category", description="The category of the journal", type='string'),
    ResponseSchema(name="publisher", description="The publisher of the journal", type='string'),
]


output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# print(format_instructions)

prompt_template = """
<format_instructions>
{format_instructions}
</format_instructions>

<query>
I am a librarian. I need to classify the following journal into a category.

{journal}
</query>
"""

prompt = PromptTemplate.from_template(
    prompt_template,
    partial_variables={'format_instructions': format_instructions}
)

chain = prompt | llm | output_parser

semaphore = asyncio.Semaphore(5)

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "category_journal.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def category_journal(journal):
    async with semaphore:
        try:
            logger.info(f"开始处理: {journal}")
            result = await chain.ainvoke({"journal": journal})
            logger.info(f"处理成功: {journal} -> {result}")
            return result
        except Exception as e:
            logger.error(f"处理失败: {journal}，错误: {e}")
            raise

async def main():
    logger.info("任务开始")
    with open("data/Table II.json", "r") as f:
        journals = json.load(f)
        journals = [journal["title"] for journal in journals]

    results = []
    for coro in tqdm([category_journal(journal) for journal in journals], desc="分类中", total=len(journals)):
        try:
            result = await coro
        except Exception:
            logger.error("最终失败: 任务兜底空结果")
            result = {"title": "", "issn": "", "category": "", "publisher": ""}
        results.append(result)

    logger.info("任务结束")
    return results


if __name__ == "__main__":
    results = asyncio.run(main())

    with open("data/tjsem_table2.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)