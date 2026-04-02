from __future__ import annotations

import logging
from pathlib import Path

import PyPDF2 as pypdf

logger = logging.getLogger(__name__)

CHUNK_SIZE = 250  # words ต่อ chunk
OVERLAP = 50      # words ที่ซ้อนกันระหว่าง chunk → ป้องกันตัดประโยคกลางคัน


def load_pdf(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected .pdf, got: {path.suffix}")

    pages: list[str] = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    if not pages:
        raise ValueError(f"No extractable text in: {file_path}")
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
        start += chunk_size - overlap  # slide window โดยถอย overlap ไว้
    return chunks


def load_and_chunk(file_path: str) -> list[dict]:
    text = load_pdf(file_path)
    chunks = chunk_text(text)
    source_name = Path(file_path).name
    return [
        {"text": chunk, "source": source_name, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]