import os
import pandas as pd
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
FRONTEND_TOKEN = os.getenv("RAASHI_FRONTEND_TOKEN") or ""

embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)
documents = []

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), authorization: str = Header(None)):
    if authorization != f"Bearer {FRONTEND_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid token")

    if file.filename.lower().endswith(".xlsx"):
        df = pd.read_excel(file.file)
    elif file.filename.lower().endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type (use .xlsx or .csv)")

    if df.shape[1] < 1:
        raise HTTPException(status_code=400, detail="File must have at least one column with questions")

    questions = df.iloc[:, 0].astype(str).tolist()
    embeddings = []

    for q in questions:
        resp = openai.Embedding.create(
            input=q,
            model="text-embedding-ada-002"
        )
        embeddings.append(resp["data"][0]["embedding"])

    global documents, index
    documents = questions
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype="float32"))

    return {"status": "ingested", "count": len(questions)}

@app.post("/query")
async def query(data: QueryRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {FRONTEND_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid token")

    if not documents:
        raise HTTPException(status_code=400, detail="No data ingested. Use /ingest first.")

    resp = openai.Embedding.create(
        input=data.question,
        model="text-embedding-ada-002"
    )
    query_embedding = np.array([resp["data"][0]["embedding"]], dtype="float32")

    distances, indices = index.search(query_embedding, 1)
    best_idx = int(indices[0][0])
    best_match = documents[best_idx]
    best_distance = float(distances[0][0])

    return {"answer_like": best_match, "distance": best_distance}
