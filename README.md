# 📄 Document Ask

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)

ระบบ Document Intelligence สำหรับการถาม-ตอบและสกัดข้อมูลจากเอกสาร (PDF) และรูปภาพ (OCR) ด้วยเทคโนโลยี **Retrieval-Augmented Generation (RAG)** โดยระบบทั้งหมดถูกออกแบบมาให้รันแบบ **Local 100%** เพื่อรักษาความลับของข้อมูล (Data Privacy) ไม่มีการส่งข้อมูลออกไปยังเซิร์ฟเวอร์ภายนอก

## Key Features

- 📄 **PDF Document Processing:** อัปโหลดและสกัดข้อความจากไฟล์ PDF พร้อมระบบหั่นข้อความ (Sliding Window Chunking) อัตโนมัติ
- 🖼️ **Image OCR (Thai & English):** รองรับการอัปโหลดรูปภาพ (PNG, JPG) พร้อมระบบสกัดข้อความด้วยเทคนิค Adaptive Thresholding และ Tesseract OCR
- 🧠 **Local LLM Integration:** ประมวลผลและตอบคำถามด้วยโมเดล `llama3.2` ผ่าน Ollama พร้อมรองรับการสั่งให้ "สรุป" (Summarize) เนื้อหา
- 🔍 **Semantic Search:** ค้นหาเนื้อหาที่เกี่ยวข้องด้วย Vector Database (ChromaDB) และ Embedding Model (`paraphrase-multilingual-MiniLM-L12-v2`)
- 💬 **Modern UI:** หน้าต่างสนทนาแบบ Chat Interface ธีม Dark Mode ที่เรียบหรู ใช้งานง่าย พัฒนาด้วย Streamlit
- 📎 **Source Citation:** ทุกคำตอบของ AI จะมีการอ้างอิงชื่อไฟล์และแหล่งที่มาเสมอ เพื่อลดปัญหา Hallucination

---

## System Architecture

ระบบถูกแบ่งออกเป็น 2 ส่วนหลัก เพื่อลดคอขวดในการประมวลผล:
1. **Frontend (Client):** รันด้วย Streamlit บนเครื่อง Local เพื่อจัดการ UI และรับส่งคำสั่งกับผู้ใช้
2. **Backend (Server):** รันใน Docker Container ประกอบด้วย FastAPI, ChromaDB และกระบวนการสกัดข้อความ (OCR/Embedding) 

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI, Uvicorn
- **AI & NLP:** - Ollama (LLM)
  - Sentence-Transformers (Embedding)
  - PyPDF2 (PDF Parsing)
  - OpenCV + PyTesseract (Image OCR)
- **Database:** ChromaDB (Vector Store v0.5.23)
- **Infrastructure:** Docker, Docker Compose

---

## 🚀 วิธีติดตั้งและรันโปรเจกต์ (Installation & Setup)

### 📋 สิ่งที่ต้องเตรียม (Prerequisites)
1. ติดตั้ง **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
2. ติดตั้ง **[Ollama](https://ollama.ai/)**
3. ติดตั้ง **Python 3.11** ขึ้นไป

### ⚙️ ขั้นตอนการรันระบบ (Step-by-Step Guide)

**1. เตรียมความพร้อมของ LLM (Ollama)**
ก่อนเริ่มรันระบบ ต้องแน่ใจว่าเครื่องของคุณมีโมเดล AI สำหรับประมวลผลข้อความแล้ว
* เปิด Terminal (หรือ Command Prompt) 
* พิมพ์คำสั่งด้านล่างเพื่อดาวน์โหลดและรันโมเดล (ระบบจะโหลดไฟล์โมเดลในครั้งแรก)
```bash
ollama run llama3.2
```

**2. ดาวน์โหลดโค้ดโปรเจกต์
เปิด Terminal ตัวใหม่ แล้วดาวน์โหลดโค้ดลงมาที่เครื่องของคุณ
```bash
git clone https://github.com/ifyoulovemeletmehearyousayit/Document_Ask.git
cd Document_Ask
```

**3. สตาร์ทระบบ Backend
ระบบ Backend และ Database จะถูกจัดการผ่าน Docker Compose
```bash
docker-compose up --build -d
```

**4. ตั้งค่าและรัน Frontend
เปิด Terminal ตัวใหม่ (อยู่ในโฟลเดอร์โปรเจกต์) สร้าง Virtual Environment และติดตั้งไลบรารีสำหรับ UI
```bash
python -m venv venv
venv\Scripts\activate
pip install streamlit requests
streamlit run app.py
```
```bash
python -m venv venv
venv\Scripts\activate
pip install streamlit requests
streamlit run app.py
```
