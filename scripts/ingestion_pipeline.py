from ingestion.doc_converter import DocumentProcessor
from ingestion.text_splitter import TextSplitter
from ingestion.table_parser import TableParser
from ingestion.embeddings import get_embedding_model
from ingestion.vector_store import VectorStore
from core.logger import get_logger
import os

logger = get_logger()

def run_ingestion(pdf_path: str):
    # =========================
    # 1. Convert PDF → Markdown
    # =========================
    processor = DocumentProcessor()
    os.makedirs('data/processed', exist_ok=True)
    logger.info("converting pdf to markdown")
    markdown_text = processor.pdf_to_markdown(
        input_path=pdf_path,
        output_path="data/processed/handbook.md"
    )
    logger.info("converted pdf to markdown")
    # =========================
    # 2. Split blocks
    # =========================
    logger.info("splitting markdown into blocks")
    text_blocks, table_blocks, all_blocks = TableParser.split_blocks(markdown_text)
    logger.info("block splitting successful.")

    # =========================
    # 3. Text splitting
    # =========================
    logger.info("starting to split the text blocks into chunks.")
    splitter = TextSplitter()
    text_chunks = splitter.split(text_blocks)
    logger.info("text blocks split to chunks successfully.")

    # =========================
    # 4. Table parsing
    # =========================
    logger.info("extracting full tables.")
    table_parser = TableParser()
    table_docs = table_parser.parse_tables(table_blocks)
    logger.info("extracted full tables successfullly")

    # =========================
    # 5. Combine documents
    # =========================
    final_docs = text_chunks + table_docs

    # =========================
    # 6. Embeddings
    # =========================
    embedding_model = get_embedding_model()

    # =========================
    # 7. Store in Vector DB
    # =========================
    logger.info("converting chunks into embeddings")
    vector_store = VectorStore(persist_directory="vector_db")

    vector_store.create(
        documents=final_docs,
        embedding_model=embedding_model,
        collection_name="university_handbook"
    )
    logger.info("converted chunks into embeddings successfully.")

    logger.info("Ingestion COMPLETE")
    logger.debug(f"Text chunks: {len(text_chunks)}")
    logger.debug(f"Table docs: {len(table_docs)}")
    logger.debug(f"Total stored: {len(final_docs)}")

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    try:
        run_ingestion("data/raw/student_handbook.pdf")
    except FileNotFoundError:
        logger.error("file not found")