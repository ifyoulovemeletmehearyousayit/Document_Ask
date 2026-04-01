from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.ingestion.embedder import delete_source, embed_and_store, list_sources, query_similar
from src.ingestion.loader import load_and_chunk
from src.ocr.processor import image_to_chunks
from src.rag.pipeline import ask_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

@asynccontextmanager
async def lifespan(app: FastAPI):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Document Intelligence API started. Raw data dir: %s", RAW_DIR)
    yield
    logger.info("API shutting down.")

app = FastAPI(title="Document Intelligence API", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    source_filter: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

def _save_upload(file: UploadFile) -> Path:
    dest = RAW_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest

# ใช้ def ธรรมดา เพื่อไม่ให้บล็อกการทำงานเวลาประมวลผลไฟล์หนักๆ
@app.post("/upload/pdf", status_code=200)
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="รับเฉพาะไฟล์ PDF เท่านั้น")
    dest = _save_upload(file)
    try:
        chunks = load_and_chunk(str(dest))
        stored = embed_and_store(chunks)
    except Exception as exc:
        logger.exception("Failed to process PDF '%s'", file.filename)
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": f"Processed {stored} chunks", "filename": file.filename}

@app.post("/upload/image", status_code=200)
def upload_image(file: UploadFile = File(...)):
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}
    if Path(file.filename).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"ไม่รองรับไฟล์รูปภาพประเภทนี้: {allowed}")
    dest = _save_upload(file)
    try:
        chunks = image_to_chunks(str(dest))
        stored = embed_and_store(chunks)
    except Exception as exc:
        logger.exception("Failed to process image '%s'", file.filename)
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": f"Processed {stored} chunks", "filename": file.filename}

@app.get("/documents")
def get_documents():
    sources = list_sources()
    return {"documents": sources, "count": len(sources)}

@app.delete("/documents/{filename}")
def remove_document(filename: str):
    deleted_count = delete_source(filename)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"ไม่พบไฟล์ '{filename}' ใน database")
    raw_path = RAW_DIR / filename
    if raw_path.exists():
        raw_path.unlink()
    return {"message": f"ลบ '{filename}' ออกแล้ว ({deleted_count} chunks)"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="คำถามต้องไม่เป็นค่าว่าง")
    answer = ask_question(request.question, source_filter=request.source_filter)
    chunks = query_similar(request.question, n_results=3, source_filter=request.source_filter)
    sources = sorted(set(c["source"] for c in chunks))
    return QueryResponse(answer=answer, sources=sources)

@app.get("/health")
async def health():
    return {"status": "ok"}