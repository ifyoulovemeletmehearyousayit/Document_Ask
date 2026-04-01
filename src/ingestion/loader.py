from __future__ import annotations
import logging
from pathlib import Path
import PyPDF2 as pypdf

logger = logging.getLogger(__name__)

CHUNK_SIZE = 250
OVERLAP = 50

def load_pdf(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages: list[str] = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {file_path}")
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def load_and_chunk(file_path: str) -> list[dict]:
    text = load_pdf(file_path)
    chunks = chunk_text(text)
    source_name = Path(file_path).name
    return [{"text": chunk, "source": source_name, "chunk_index": i} for i, chunk in enumerate(chunks)]