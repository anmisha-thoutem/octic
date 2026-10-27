from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

from config.settings import settings


class DocumentSearchEngine:
    def __init__(self) -> None:
        self.index = None
        self.documents: list[Document] = []

    def _get_embeddings(self):
        if not settings.has_openai_key:
            raise ValueError("OPENAI_API_KEY is required for document search.")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

    def index_file(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        text = self._extract_text(file_path)
        if not text.strip():
            raise ValueError("The uploaded file does not contain any readable text.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata={"source": str(file_path)}) for chunk in chunks]

        if not docs:
            raise ValueError("No text chunks were generated from the uploaded document.")

        embeddings = self._get_embeddings()
        self.documents.extend(docs)
        if self.index is None:
            self.index = FAISS.from_documents(docs, embeddings)
        else:
            self.index.add_documents(docs)

        return {
            "status": "indexed",
            "source": str(file_path),
            "chunks": len(docs),
        }

    def _extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        if suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        raise ValueError("Unsupported document type. Please upload a PDF or TXT file.")

    def search(self, query: str, k: int = 3) -> list[str]:
        if not query or not str(query).strip():
            raise ValueError("Search query cannot be empty.")
        if self.index is None:
            return []
        results = self.index.similarity_search(str(query), k=k)
        return [result.page_content for result in results]


engine = DocumentSearchEngine()


def upload_and_index_document(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    if not file_name:
        raise ValueError("A file name is required.")
    if not file_bytes:
        raise ValueError("No file content was provided.")

    file_path = Path("/tmp") / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(file_bytes)
    return engine.index_file(file_path)


def document_search_tool(query: str) -> str:
    matches = engine.search(query)
    if not matches:
        return "No relevant information was found in the uploaded documents."
    return "\n\n---\n\n".join(matches)
