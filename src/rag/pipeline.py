from __future__ import annotations

import logging
import os
from typing import Optional
from ollama import Client

from src.ingestion.embedder import get_all_chunks, query_similar

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_CONTEXT_TOKENS = 3000   
AVG_CHARS_PER_TOKEN = 4     
SUMMARY_CHUNK_LIMIT = 30    

SUMMARY_KEYWORDS = frozenset(["สรุป", "summary", "summarize", "overview", "ภาพรวม"])

ollama_client = Client(host=OLLAMA_HOST)

def _is_thai(text: str) -> bool:
    return any("\u0e00" <= c <= "\u0e7f" for c in text)

def _is_summary_request(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in SUMMARY_KEYWORDS)

def _truncate_to_token_limit(chunks: list[dict], max_tokens: int) -> list[dict]:
    max_chars = max_tokens * AVG_CHARS_PER_TOKEN
    total, kept = 0, []
    for chunk in chunks:
        total += len(chunk["text"])
        if total > max_chars:
            break
        kept.append(chunk)
    return kept

def _build_context(chunks: list[dict]) -> str:
    parts = [
        f"[{i+1}] (Source: {c['source']}, chunk #{c['chunk_index']})\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    return "\n\n".join(parts)

def _build_prompt(question: str, context: str, is_thai: bool, is_summary: bool) -> str:
    lang_inst = "IMPORTANT: You MUST answer in Thai (ภาษาไทย)." if is_thai else "Answer in English."
    task_hint = (
        "Provide a comprehensive, structured SUMMARY of all provided context."
        if is_summary
        else "Answer the question accurately based only on the provided context."
    )
    return f"""You are a professional Document Intelligence Assistant.
{lang_inst}

Task: {task_hint}

Rules:
- Use bullet points or numbered lists for readability.
- If the source language differs from the question language, translate your answer.
- Cite source filenames when mentioning specific facts.
- If the context does not contain enough information, say so clearly — do not hallucinate.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer:"""

def ask_question(question: str, source_filter: Optional[str] = None) -> str:
    is_summary = _is_summary_request(question)
    is_thai = _is_thai(question)

    if is_summary:
        chunks = get_all_chunks(source_filter=source_filter)
        chunks.sort(key=lambda c: (c["source"], c["chunk_index"]))
        chunks = chunks[:SUMMARY_CHUNK_LIMIT]
    else:
        chunks = query_similar(question, n_results=5, source_filter=source_filter)

    if not chunks:
        return "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่เลือก" if is_thai else "Sorry, no relevant information found in the selected documents."

    chunks = _truncate_to_token_limit(chunks, MAX_CONTEXT_TOKENS)
    context = _build_context(chunks)
    prompt = _build_prompt(question, context, is_thai, is_summary)

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc)
        return "เกิดข้อผิดพลาดในการเชื่อมต่อกับ LLM กรุณาลองใหม่อีกครั้ง" if is_thai else "Failed to connect to LLM. Please try again."