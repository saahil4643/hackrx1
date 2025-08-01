import os
import uuid
import tempfile
import requests
import pdfplumber
import google.generativeai as genai
from numpy import dot
from numpy.linalg import norm
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import get_db
# from models import Document, DocumentChunk  # Uncomment once defined

from pinecone import Pinecone
from starlette.status import HTTP_204_NO_CONTENT

# Load environment variables
load_dotenv()

# Initialize external services
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Token Auth
security = HTTPBearer()
VALID_TOKEN = os.getenv("API_TOKEN")

# FastAPI instance
app = FastAPI()

# Utility
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Auth Dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# Request Schemas
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class PDFBlobRequest(BaseModel):
    blob_url: str


@app.post("/hackrx/run", dependencies=[Depends(verify_token)])
def run_query(payload: QueryRequest):
    blob_url = payload.documents
    questions = payload.questions

    try:
        # Step 1: Download PDF
        response = requests.get(blob_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            file_path = tmp_file.name

        # Step 2: Extract text
        with pdfplumber.open(file_path) as pdf:
            extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        os.remove(file_path)

        if not extracted_text.strip():
            raise HTTPException(status_code=HTTP_204_NO_CONTENT, detail="No text could be extracted.")

        # Step 3: Chunk
        chunk_size, overlap = 1000, 200
        start, chunks = 0, []
        while start < len(extracted_text):
            end = min(start + chunk_size, len(extracted_text))
            chunks.append(extracted_text[start:end])
            start += chunk_size - overlap
        chunks = chunks[:20]

        # Step 4: Embed chunks
        chunk_embeddings = []
        for chunk in chunks:
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )["embedding"]
            chunk_embeddings.append({"chunk": chunk, "embedding": embedding})

        # Step 5: Answer Questions
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        chat = model.start_chat()
        results = []

        for question in questions:
            query_embedding = genai.embed_content(
                model="models/embedding-001",
                content=question,
                task_type="retrieval_query"
            )["embedding"]

            top_chunks = sorted(
                chunk_embeddings,
                key=lambda c: cosine_similarity(query_embedding, c["embedding"]),
                reverse=True
            )[:3]

            context = "\n\n".join(c["chunk"] for c in top_chunks)
            prompt = f"""You are an expert insurance policy assistant. Answer the question strictly based on the context below.

            Question: {question}

            Context:
            {context}

            Answer concisely as if writing for a customer-facing FAQ. If the answer is present, give a clear, specific response. If the information is not in the context, respond: "Not mentioned in the context."

            Return only the answer. No introductions, no extra explanation."""
            reply = chat.send_message(prompt)
            results.append(reply.text.strip())

        return {
            "message": "Questions answered successfully.",
            "answers": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import concurrent.futures

@app.post("/api/v1/hackrx/run", dependencies=[Depends(verify_token)])
def run_query(payload: QueryRequest):
    blob_url = payload.documents
    questions = payload.questions

    try:
        # Step 1: Download PDF
        response = requests.get(blob_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            file_path = tmp_file.name

        # Step 2: Extract text
        with pdfplumber.open(file_path) as pdf:
            extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        os.remove(file_path)

        if not extracted_text.strip():
            raise HTTPException(status_code=HTTP_204_NO_CONTENT, detail="No text could be extracted.")

        # Step 3: Chunk (reduced to 10 chunks)
        chunk_size, overlap = 1000, 200
        start, chunks = 0, []
        while start < len(extracted_text):
            end = min(start + chunk_size, len(extracted_text))
            chunks.append(extracted_text[start:end])
            start += chunk_size - overlap
        chunks = chunks[:10]

        # Step 4: Parallel Embedding of chunks
        def embed_chunk(chunk):
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )["embedding"]
            return {"chunk": chunk, "embedding": embedding}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            chunk_embeddings = list(executor.map(embed_chunk, chunks))

        # Step 5: Parallel Question Answering
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        chat = model.start_chat()
        results = []

        def answer_question(question):
            query_embedding = genai.embed_content(
                model="models/embedding-001",
                content=question,
                task_type="retrieval_query"
            )["embedding"]

            top_chunks = sorted(
                chunk_embeddings,
                key=lambda c: cosine_similarity(query_embedding, c["embedding"]),
                reverse=True
            )[:3]

            context = "\n\n".join(c["chunk"] for c in top_chunks)
            prompt = f"""You are a customer-facing insurance assistant. Based on the context below, give a **direct, specific answer** to the question.

Only return the answer. If the answer is not found in the context, say exactly: "Not mentioned in the context."

Question: {question}

Context:
{context}

Answer:"""

            reply = chat.send_message(prompt)   
            return {"question": question, "answer": reply.text.strip()}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(answer_question, questions))

        only_answers = [result["answer"] for result in results]
        return {
            "answers": only_answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
