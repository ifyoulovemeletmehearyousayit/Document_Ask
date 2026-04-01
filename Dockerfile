FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-tha \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ติดตั้ง Library ก่อน
RUN pip install --no-cache-dir -r requirements.txt

# โหลด Model เก็บไว้ใน Image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

EXPOSE 8000

# รัน API โดยชี้ไปที่โฟลเดอร์ src/api/main.py
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]