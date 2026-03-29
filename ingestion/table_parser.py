import re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemma-3-1b-it",
    temperature=0
    ) 

TABLE_DESCRIPTION_PROMPT = """You are a data analyst. Given the table below, write a 2-3 sentence description that explains:
1. What the table is about
2. What the columns and rows represent
3. Any notable values, patterns, or metrics a user might search for

Rules:
- Be specific — use actual column names, row labels, and values from the table
- Do NOT write generic statements like "this table shows some data"
- Output ONLY the description, no preamble, no explanation

Table:
{table}
"""

class TableParser:

    def __init__(self):
        self.llm = llm
        self.parser = StrOutputParser()
        prompt = PromptTemplate.from_template(TABLE_DESCRIPTION_PROMPT)
        self.chain = prompt | self.llm | self.parser
    
    @staticmethod
    def split_blocks(markdown_text: str):
        blocks = re.split(r"\n\s*\n", markdown_text)

        text_blocks = []
        table_blocks = []
        skip_indices = set()

        for i, block in enumerate(blocks):
            if "|" in block and "---" in block:
                preceding_text = ""
                if i > 0 and not ("|" in blocks[i - 1] and "---" in blocks[i - 1]):
                    preceding_text = blocks[i - 1].strip()
                    skip_indices.add(i - 1)  # mark as absorbed

                table_blocks.append({
                    "table": block,
                    "preceding_text": preceding_text
                })
            else:
                if i not in skip_indices:
                    text_blocks.append(block)

        return text_blocks, table_blocks, blocks

    def parse_tables(self, table_blocks: list[dict]):
        table_docs = []

        for item in table_blocks:
            table = item["table"]
            preceding_text = item["preceding_text"]

            full_input = f"{preceding_text}\n\n{table}" if preceding_text else table
            description = self.chain.invoke({"table": full_input}).strip()

            page_content = f"{description}\n\n{preceding_text}\n\n{table}".strip()

            table_docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "type": "table",
                        "context": description
                    }
                )
            )

        return table_docs