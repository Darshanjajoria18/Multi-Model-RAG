import os
import torch
import pytesseract
import cv2
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from moviepy.editor import VideoFileClip

# ✅ Use an open-access model (No authentication required)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# ✅ Load the model with memory optimization
def load_model(model_name):
    """Loads the language model with quantization (8-bit) for low memory usage."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error: Could not load model '{model_name}' - {e}")
        return None, None

# ✅ Text extraction functions
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except:
        return ""

def extract_text_from_docx(docx_path):
    """Extracts text from a Word document."""
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""

def extract_text_from_image(image_path):
    """Extracts text from an image using OCR."""
    try:
        img = cv2.imread(image_path)
        return pytesseract.image_to_string(img).strip()
    except:
        return ""

def extract_audio_from_video(video_path):
    """Extracts and transcribes audio from a video (Placeholder)."""
    try:
        clip = VideoFileClip(video_path)
        audio_path = "temp_audio.wav"
        clip.audio.write_audiofile(audio_path)
        return "(Speech-to-Text Transcription here)"  # Replace with an actual STT model
    except:
        return ""

# ✅ Folder processing function
def process_folder(folder_path):
    """Processes a folder with PDFs, Docs, Images, and Videos."""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found!")
        return []

    text_data = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            text_data.append(extract_text_from_pdf(file_path))
        elif file.endswith(".docx"):
            text_data.append(extract_text_from_docx(file_path))
        elif file.endswith(('.png', '.jpg', '.jpeg')):
            text_data.append(extract_text_from_image(file_path))
        elif file.endswith(('.mp4', '.avi', '.mov')):
            text_data.append(extract_audio_from_video(file_path))

    if not text_data:
        print("❌ Error: No valid files found in the folder!")
    return text_data

# ✅ Embedding function
def embed_texts(texts):
    """Embeds text using Sentence Transformers and stores them in FAISS."""
    if not texts:
        print("❌ Error: No text data available for embedding!")
        return None, None, None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings, texts

# ✅ Retrieval & Generation function
def retrieve_and_generate(index, embeddings, texts, query):
    """Retrieves the most relevant text and generates a response."""
    if index is None or not texts:
        print("❌ Error: No indexed data available!")
        return "I don't have enough data to answer."

    query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query], convert_to_numpy=True)
    _, closest_idx = index.search(query_embedding, k=1)

    relevant_text = texts[closest_idx[0][0]]

    tokenizer, model = load_model(MODEL_NAME)
    if model is None:
        return "❌ Error: Model could not be loaded."

    input_text = f"Context: {relevant_text}\nQuery: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Run the pipeline
folder_path = "your_data_folder"  # Change this to your actual folder

texts = process_folder(folder_path)
index, embeddings, texts = embed_texts(texts)

if index is not None:
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = retrieve_and_generate(index, embeddings, texts, query)
        print("AI Assistant:", response)