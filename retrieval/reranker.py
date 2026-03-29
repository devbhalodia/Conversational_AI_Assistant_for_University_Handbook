from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

TOP_N = 4

class Reranker:
    def __init__(self, top_n: int = TOP_N):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_n = top_n

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Takes a query and a list of retrieved docs,
        scores each (query, doc) pair,
        returns top_n docs sorted by relevance score.
        """
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for _, doc in scored_docs[:self.top_n]]