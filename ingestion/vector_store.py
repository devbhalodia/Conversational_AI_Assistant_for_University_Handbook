from langchain_community.vectorstores import Chroma
import os


class VectorStore:
    os.makedirs('vector_db', exist_ok=True)
    def __init__(self, persist_directory="vector_db"):
        self.persist_directory = persist_directory

    def create(self, documents, embedding_model, collection_name):
        return Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )