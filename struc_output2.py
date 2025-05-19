from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.5,
)


response_schemas = [
    ResponseSchema(name="city", description="The city name"),
    ResponseSchema(name="country", description="The country name"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

prompt_template = """
<format_instructions>
{format_instructions}
{format_instructions}
</format_instructions>

<question>
{question}
</question>
"""

prompt = PromptTemplate.from_template(
    prompt_template,
    partial_variables={'format_instructions': format_instructions}
)

chain = prompt | llm | output_parser

result = chain.invoke({"question": "What is the capital of France?"})

print(result)