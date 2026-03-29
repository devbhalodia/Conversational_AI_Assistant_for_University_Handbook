from docling.document_converter import DocumentConverter
from core.logger import get_logger
import pypdf

logger = get_logger()

BATCH_SIZE = 10

class DocumentProcessor:
    def __init__(self):
        self.converter = DocumentConverter()

    def pdf_to_markdown(self, input_path: str, output_path: str) -> str:
        try:
            total_pages = len(pypdf.PdfReader(input_path).pages)
            logger.info(f"Total pages: {total_pages}")

            all_markdown = []

            for start in range(0, total_pages, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total_pages)
                logger.info(f"Converting pages {start + 1}–{end}...")

                result = self.converter.convert(
                    input_path,
                    page_range=(start + 1, end)
                )
                all_markdown.append(result.document.export_to_markdown())

            markdown_output = "\n\n".join(all_markdown)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_output)

            return markdown_output

        except Exception as e:
            raise RuntimeError(f"Document conversion failed: {e}")