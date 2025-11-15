import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

root_path = Path(__file__).parent.parent
docs_path = root_path / "docs"


@dataclass
class ConversionConfig:
    """Configuration for document conversion"""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    db_path: Path = root_path / "livekit_agents_db"
    collection_name: str = "livekit_agents_docs"


class Markdown2VectorDB:

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self.vectorstore = None
        self.documents = list()

    def load_markdown(
        self,
        docs_path: Optional[Path] = None,
    ) -> List[Document]:
        loader = DirectoryLoader(
            path=docs_path, glob="*.md", loader_cls=UnstructuredMarkdownLoader
        )
        documents = loader.load()
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            # Detect code blocks
            chunk.metadata["has_code"] = "```" in chunk.page_content

        self.documents = chunks
        return chunks

    def create_vectorstore(self, docs_path: Path) -> None:
        self.vectorstore = Chroma(
            persist_directory=str(self.config.db_path),
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
        )

        documents = self.load_markdown(docs_path)
        splits = self.split_documents(documents)

        self.vectorstore.add_documents(splits)

        self.vectorstore.persist()


if __name__ == "__main__":
    vectordb = Markdown2VectorDB()
    vectordb.create_vectorstore(docs_path)
