from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

CHUNK_SIZE = 200
OVERLAP = 20
OCR_LANG = "eng+tha"
OCR_CONFIG = "--dpi 300 --oem 3 --psm 6"
# oem 3 = LSTM engine, psm 6 = assume uniform text block


def _configure_tesseract() -> None:
    if os.path.exists("/.dockerenv"):
        # ใน Docker → tesseract อยู่ใน PATH อยู่แล้ว
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
    else:
        win_path = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        pytesseract.pytesseract.tesseract_cmd = win_path

    if not shutil.which(pytesseract.pytesseract.tesseract_cmd) and \
       not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        logger.warning("Tesseract not found — OCR will fail.")


_configure_tesseract()


def _preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # adaptive threshold รับมือภาพที่แสงไม่สม่ำเสมอได้ดีกว่า global threshold
    return cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )


def extract_text(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    try:
        processed = _preprocess(img)
        text = pytesseract.image_to_string(processed, lang=OCR_LANG, config=OCR_CONFIG)
        logger.info("OCR extracted %d chars from '%s'", len(text), Path(image_path).name)
        return text.strip()
    except Exception as exc:
        logger.error("OCR failed for %s: %s", image_path, exc)
        raise RuntimeError(f"OCR failed: {image_path}") from exc


def image_to_chunks(image_path: str) -> list[dict]:
    text = extract_text(image_path)
    if not text:
        raise ValueError(f"No text found in: {image_path}")

    words = text.split()
    source_name = Path(image_path).name
    chunks: list[dict] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append({
            "text": " ".join(words[start:end]),
            "source": source_name,
            "chunk_index": len(chunks),
        })
        if end == len(words):
            break
        start += CHUNK_SIZE - OVERLAP
    return chunks