import os
import tempfile
import requests
import pdfplumber
import google.generativeai as genai
import concurrent.futures
import logging
import re
from typing import List
from numpy import dot
from numpy.linalg import norm
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.status import HTTP_204_NO_CONTENT

# Load env
load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Logging
logging.basicConfig(level=logging.INFO)

# Auth
security = HTTPBearer()
VALID_TOKEN = os.getenv("API_TOKEN")

# FastAPI app
app = FastAPI()

# ===== Helpers =====
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def extract_keywords(text: str):
    """Extract simple keywords from question text."""
    words = re.findall(r"\b\w+\b", text.lower())
    stopwords = {"the", "is", "are", "a", "an", "in", "on", "for", "to", "with", "and", "of"}
    return [w for w in words if w not in stopwords and len(w) > 2]

def clean_text(text: str):
    """Remove extra newlines and spaces."""
    return re.sub(r"\s+", " ", text).strip()

def download_and_extract_pdf(url: str) -> str:
    """Download PDF and return cleaned extracted text."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF: {url}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            file_path = tmp_file.name
        
        with pdfplumber.open(file_path) as pdf:
            extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        
        os.remove(file_path)
        return clean_text(extracted_text)
    except Exception as e:
        logging.error(f"Error processing PDF {url}: {e}")
        return ""

def chunk_text(text: str, chunk_size=1000, overlap=200):
    """Split cleaned text into overlapping chunks."""
    start, chunks = 0, []
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(clean_text(text[start:end]))
        start += chunk_size - overlap
    return chunks

# ===== Request Schema =====
class QueryRequest(BaseModel):
    documents: List[str]  # Multiple PDF URLs
    questions: List[str]

# ===== Auth dependency =====
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# ===== Main API =====
@app.post("/api/v2/hackrx/run", dependencies=[Depends(verify_token)])
def run_query(payload: QueryRequest):
    pdf_urls = payload.documents
    questions = payload.questions

    logging.info(f"Processing {len(pdf_urls)} PDFs with {len(questions)} questions.")

    # Step 1: Extract text from all PDFs
    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_texts = list(executor.map(download_and_extract_pdf, pdf_urls))
    
    # Combine all texts into one big corpus
    full_text = "\n".join(all_texts)
    if not full_text.strip():
        raise HTTPException(status_code=HTTP_204_NO_CONTENT, detail="No text extracted from PDFs.")

    # Step 2: Chunk the text
    all_chunks = chunk_text(full_text)
    logging.info(f"Total chunks created: {len(all_chunks)}")

    # Step 3: Embed chunks once
    def embed_chunk(chunk):
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )["embedding"]
        return {"chunk": chunk, "embedding": embedding}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunk_embeddings = list(executor.map(embed_chunk, all_chunks))

    logging.info("All chunks embedded successfully.")

    # Step 4: Answer each question
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    chat = model.start_chat()

    def answer_question(question):
        keywords = extract_keywords(question)
        logging.info(f"Keywords for '{question}': {keywords}")

        # Filter chunks by keyword match
        keyword_matched_chunks = [
            c for c in chunk_embeddings if any(kw in c["chunk"].lower() for kw in keywords)
        ]
        logging.info(f"Chunks after keyword filter: {len(keyword_matched_chunks)}")

        if not keyword_matched_chunks:
            return "Not mentioned in the context."

        # Embed the question
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=question,
            task_type="retrieval_query"
        )["embedding"]

        # Get top relevant chunks by cosine similarity
        top_chunks = sorted(
            keyword_matched_chunks,
            key=lambda c: cosine_similarity(query_embedding, c["embedding"]),
            reverse=True
        )[:3]

        # Clean up context before sending to Gemini
        context = clean_text("\n\n".join(c["chunk"] for c in top_chunks))

        prompt = f"""
You are a PDF content expert. 
Answer the question strictly using the provided context below. 
Do not use any outside knowledge. 
If the answer is not in the context, say exactly: "Not mentioned in the context."

Question: {question}

Context:
{context}

Answer:
""".strip()

        reply = chat.send_message(prompt)
        return clean_text(reply.text)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        answers = list(executor.map(answer_question, questions))

    return {"answers": answers}
