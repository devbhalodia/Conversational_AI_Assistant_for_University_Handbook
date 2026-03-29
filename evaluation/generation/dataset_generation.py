import json
import time
import logging
import chromadb
from collections import defaultdict

from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)

with open("evaluation/retrieval/eval_dataset.json") as f:
    eval_dataset = json.load(f)

client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="university_handbook")

chunk_lookup = {}
results = collection.get(include=["documents", "metadatas"])
for doc, meta, cid in zip(results["documents"], results["metadatas"], results["ids"]):
    chunk_lookup[cid] = doc

pipeline = RAGPipeline()

samples = []
skipped = 0

for i, item in enumerate(eval_dataset):
    query = item["query"]
    ground_truth_id = item["relevant_chunk_id"]
    ground_truth_text = chunk_lookup.get(ground_truth_id, "")

    if not ground_truth_text:
        print(f"[{i+1}] Skipping — chunk {ground_truth_id} not found")
        skipped += 1
        continue

    print(f"[{i+1}/{len(eval_dataset)}] Running: {query[:60]}")
    start_time = time.time()
    try:
        result = pipeline.run(query, session_id=f"eval_{i}")
    except Exception as e:
        print(f"[{i+1}] Pipeline failed: {e}")
        skipped += 1
        continue
    end_time = time.time()
    latency = end_time - start_time

    answer = result["answer"]
    context = result["context"]
    contexts = context.split("\n\n") if context else [""]

    relevant_rank = None
    try:
        retrieved_docs = pipeline.retriever.retrieve(result["rewritten_query"])
        reranked_docs  = pipeline.reranker.rerank(result["rewritten_query"], retrieved_docs)

        for rank, doc in enumerate(reranked_docs, start=1):
            if doc.metadata.get("chunk_id") == ground_truth_id:
                relevant_rank = rank
                break
    except Exception:
        pass

    samples.append({
        "question": query,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth_text,
        "_latency": latency,
        "_relevant_rank": relevant_rank,
        "_retrieval_ok": result["retrieval_ok"],
        "_fallback_used": result["fallback_used"],
        "_was_rewritten": result["was_rewritten"],
        "_chunk_type": item.get("chunk_type", "text"),
    })

    time.sleep(4)

latencies = [s["_latency"] for s in samples if "_latency" in s]

if latencies:
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage latency: {avg_latency:.2f} sec")

print(f"\nCollected {len(samples)} samples ({skipped} skipped)")

with open("evaluation/generation/generated_dataset.json", "w") as f:
    json.dump(samples, f, indent=2)

print("\nSaved to evaluation/generation/generated_dataset.json")