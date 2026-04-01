from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

def _configure_tesseract() -> None:
    if os.path.exists("/.dockerenv"):
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
    else:
        win_path = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        pytesseract.pytesseract.tesseract_cmd = win_path

    if not shutil.which(pytesseract.pytesseract.tesseract_cmd) and not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        logger.warning("Tesseract not found. OCR will fail.")

_configure_tesseract()

CHUNK_SIZE = 200  
OVERLAP = 20      
OCR_LANG = "eng+tha"  
OCR_CONFIG = "--dpi 300 --oem 3 --psm 6"  

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _preprocess_adaptive(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

def extract_text(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = _to_gray(img)
    try:
        processed = _preprocess_adaptive(gray) 
        text = pytesseract.image_to_string(processed, lang=OCR_LANG, config=OCR_CONFIG)
        logger.info("Extracted %d characters from '%s'", len(text), Path(image_path).name)
        return text.strip()
    except Exception as exc:
        logger.error("OCR failed for %s: %s", image_path, exc)
        raise RuntimeError(f"OCR failed for: {image_path}")

def image_to_chunks(image_path: str) -> list[dict]:
    text = extract_text(image_path)
    if not text:
        raise ValueError(f"No text found in image: {image_path}")
    words = text.split()
    source_name = Path(image_path).name
    chunks: list[dict] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({"text": chunk, "source": source_name, "chunk_index": len(chunks)})
        if end == len(words):
            break
        start += CHUNK_SIZE - OVERLAP
    return chunks