from langchain_community.vectorstores import Chroma
from ingestion.embeddings import get_embedding_model



class Retriever:

    def __init__(self, collection_name="university_handbook", k=20):
        embedding_model = get_embedding_model()

        self.vector_store = Chroma(
            persist_directory='vector_db',
            embedding_function=embedding_model,
            collection_name=collection_name
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "lambda_mult": 0.5}
        )

    def retrieve(self, query: str):
        return self.retriever.invoke(query)