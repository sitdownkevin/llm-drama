from dotenv import load_dotenv, find_dotenv
import asyncio
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
    ResponseSchema(name="match", description="Whether the journal matches the query", type='boolean'),
    ResponseSchema(name="reason", description="The reason for the match or mismatch", type='string'),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# print(format_instructions)

prompt_template = """
<format_instructions>
{format_instructions}
</format_instructions>

<query>
I am doing research on the topic of "{topic}". Can you help me judge whether the following journal matches my research topic:

{journal}
</query>

<constraints>
1. The journal should be a peer-reviewed journal.
2. The journal should be published in the last 10 years.
3. The journal should be published in the United States.
4. The journal should be published in the field of "{topic}".
5. Match means the journal is relevant to the research topic.
6. Other factors that may affect the match are the journal.
7. Give a reason for your judgement.
</constraints>
"""

prompt = PromptTemplate.from_template(
    prompt_template,
    partial_variables={'format_instructions': format_instructions}
)

chain = prompt | llm | output_parser

async def judge_journal(topic, journal):
    return await chain.ainvoke({"topic": topic, "journal": journal})
    
if __name__ == "__main__":
    result = asyncio.run(judge_journal("IBD", "Internet Research"))
    print(result)