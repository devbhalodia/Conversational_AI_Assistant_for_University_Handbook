from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size=2500, chunk_overlap=250):
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ],
            strip_headers=False
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def split(self, text_blocks: list[str]):
        sections = self.header_splitter.split_text("\n\n".join(text_blocks))
        return self.text_splitter.split_documents(sections)