# 📌 Multi-Modal Retrieval-Augmented Generation (RAG) System

A powerful AI-driven **Multi-Modal Retrieval-Augmented Generation (RAG) System** that extracts, embeds, and retrieves insights from **documents, images, and videos**—enhancing searchability and knowledge retrieval.

---

## 🚀 Features

✅ Extracts text from **PDFs** and **DOCX** files.  
✅ Uses **OCR** to extract text from **images**.  
✅ Processes **videos** by extracting and transcribing **audio**.  
✅ Stores and retrieves data efficiently using **FAISS indexing**.  
✅ Generates AI-powered responses using **Mistral-7B**.  

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/Multi-Modal-RAG.git
cd Multi-Modal-RAG
```

### 2️⃣ Install Dependencies
```bash
pip install torch transformers faiss-cpu sentence-transformers opencv-python pytesseract PyPDF2 python-docx moviepy
```

⚠ **Note:** If using GPU, install `faiss-gpu` instead of `faiss-cpu` for better performance.

### 3️⃣ Set Up OCR (Tesseract)
- Install **Tesseract-OCR** (Required for image text extraction).  
- [Download Tesseract](https://github.com/tesseract-ocr/tesseract) and set the path in your environment.  

---

## 📂 File Processing Overview

### 1️⃣ Text Extraction
| File Type  | Processing Method |
|------------|------------------|
| **PDFs**   | Extracted using `PyPDF2`. |
| **DOCX**   | Extracted using `python-docx`. |
| **Images** | Processed via OCR using `pytesseract`. |
| **Videos** | Extracted via `MoviePy`, transcribed using an STT model. |

---

## 🚀 Running the System

### 1️⃣ Process Files in a Folder
```python
folder_path = "your_data_folder"
texts = process_folder(folder_path)
index, embeddings, texts = embed_texts(texts)
```

### 2️⃣ Ask AI Questions
```python
if index is not None:
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = retrieve_and_generate(index, embeddings, texts, query)
        print("AI Assistant:", response)
```

---

## 📌 Future Enhancements
🚀 **Integrate OpenAI Whisper for real-time speech transcription.**  
🚀 **Improve document parsing with layout-aware models like LayoutLM.**  
🚀 **Optimize FAISS for large-scale knowledge retrieval.**  

---

## 👨‍💻 Author
📌 **Darshan kumar jajoria**   
📌 **[Your GitHub]([https://github.com/your-username](https://github.com/Darshanjajoria18))**  

---
