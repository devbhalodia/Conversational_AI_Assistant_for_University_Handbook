from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import time

from dotenv import load_dotenv
load_dotenv() 
llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

QUERY_GEN_PROMPT = """You are evaluating a retrieval system.

{instruction}

Rules:
- Question must be answerable ONLY from this chunk
- Be specific, not generic
- Return ONLY the question, nothing else

Chunk:
{chunk_text}"""

prompt = PromptTemplate.from_template(QUERY_GEN_PROMPT)
chain  = prompt | llm | StrOutputParser()

def generate_single_query(chunk_text: str, chunk_type: str) -> str:
    if chunk_type == "table":
        instruction = "Generate 1 specific question about values, metrics, or comparisons found in this table."
    else:
        instruction = "Generate 1 specific question whose answer is clearly found in this text."

    return chain.invoke({
        "instruction": instruction,
        "chunk_text": chunk_text
    }).strip()


def build_eval_dataset(collection, min_text_length: int = 1500):
    results = collection.get(include=["documents", "metadatas"])

    documents = results["documents"]
    metadatas = results["metadatas"]
    ids = results["ids"]

    dataset = []

    for doc, meta, chunk_id in zip(documents, metadatas, ids):
        meta       = meta or {}
        chunk_type = meta.get("type", "text")

        # Filter out small text chunks
        if chunk_type != "table" and len(doc.strip()) < min_text_length:
            print(f"⏭️  Skipping {chunk_id} — too short ({len(doc.strip())} chars)")
            continue

        print(f"⚙️  Generating query for {chunk_id}...")
        query = generate_single_query(doc, chunk_type)

        dataset.append({
            "query": query,
            "relevant_chunk_id": chunk_id,
            "chunk_type": chunk_type
        })

        time.sleep(3)

    with open("eval_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    text_count  = sum(1 for d in dataset if d["chunk_type"] != "table")
    table_count = sum(1 for d in dataset if d["chunk_type"] == "table")
    print(f"\n✅ Dataset saved — {text_count} text queries + {table_count} table queries = {len(dataset)} total")

    return dataset

import chromadb

client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="university_handbook")

dataset = build_eval_dataset(collection, min_text_length=1500)