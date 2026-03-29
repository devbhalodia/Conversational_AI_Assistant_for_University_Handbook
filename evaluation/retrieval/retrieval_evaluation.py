import json
import numpy as np
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker

def evaluate(dataset_path: str = "evaluation/retrieval/eval_dataset.json"):
    retriever = Retriever(k=20)
    reranker = Reranker(top_n=4)

    with open(dataset_path) as f:
        dataset = json.load(f)

    recall_20_scores = []
    recall_4_scores  = []
    mrr_4_scores = []
    #
    ndcg_4_scores = [] 
    def dcg(relevances):
        return sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)])
    #

    for i, item in enumerate(dataset):
        query = item["query"]
        ground_truth_id = item["relevant_chunk_id"]

        retrieved_docs = retriever.retrieve(query)
        retrieved_ids = [doc.metadata.get("chunk_id") for doc in retrieved_docs]

        recall_20 = 1.0 if ground_truth_id in retrieved_ids else 0.0
        recall_20_scores.append(recall_20)

        reranked_docs = reranker.rerank(query, retrieved_docs)
        reranked_ids = [doc.metadata.get("chunk_id") for doc in reranked_docs]

        recall_4 = 1.0 if ground_truth_id in reranked_ids else 0.0
        recall_4_scores.append(recall_4)

        mrr = 0.0
        for rank, doc_id in enumerate(reranked_ids, start=1):
            if doc_id == ground_truth_id:
                mrr = 1.0 / rank
                break
        mrr_4_scores.append(mrr)
        #
        relevances = [1 if doc_id == ground_truth_id else 0 for doc_id in reranked_ids]
        dcg_score = dcg(relevances)

        ideal_relevances = sorted(relevances, reverse=True) 
        idcg = dcg(ideal_relevances)

        ndcg_4 = dcg_score / idcg if idcg > 0 else 0.0
        ndcg_4_scores.append(ndcg_4)
        #

        print(f"[{i+1}/{len(dataset)}] {query[:60]:<60} | R@20={recall_20:.0f} R@4={recall_4:.0f} MRR={mrr:.2f}")

    print("\n" + "="*50)
    print(f"{'Recall@20':<20} {np.mean(recall_20_scores):.4f}  ← is retriever fetching the right chunk?")
    print(f"{'Recall@4':<20} {np.mean(recall_4_scores):.4f}  ← is reranker keeping the right chunk?")
    print(f"{'MRR@4':<20} {np.mean(mrr_4_scores):.4f}  ← is reranker pushing it to the top?")
    print(f"{'nDCG@4':<20} {np.mean(ndcg_4_scores):.4f}  ← ranking quality across all positions")
    print("="*50)

    r20 = np.mean(recall_20_scores)
    r4  = np.mean(recall_4_scores)
    mrr = np.mean(mrr_4_scores)

    print("\nDiagnosis:")
    if r20 < 0.7:
        print("Retriever is weak — correct chunk not found in top 20 often enough. Consider better embeddings or larger k.")
    else:
        print("Retriever is healthy.")

    if r20 >= 0.7 and r4 < 0.6:
        print("Reranker is too aggressive — dropping correct chunks. Consider increasing top_n.")
    elif r4 >= 0.6:
        print("Reranker is retaining correct chunks.")

    if r4 >= 0.6 and mrr < 0.4:
        print("Reranker finds the chunk but doesn't rank it first. Cross-encoder may need a better model.")
    elif mrr >= 0.4:
        print("Reranker is ranking correctly.")

evaluate()