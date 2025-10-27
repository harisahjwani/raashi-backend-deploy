# app.py
import os
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# load .env if present
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set environment variable OPENAI_API_KEY in the deployment settings.")
openai.api_key = OPENAI_API_KEY

# simple frontend token to protect API from public abuse
FRONTEND_TOKEN = os.getenv("RAASHI_FRONTEND_TOKEN", "")

# paths
DATA_DIR = Path("./qbank")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = Path("./data/raashi_meta.db")
INDEX_PATH = Path("./data/faiss.index")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
TOP_K = 6
Path("./data").mkdir(exist_ok=True)

# sqlite
conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS docs (
    doc_id INTEGER PRIMARY KEY,
    file_name TEXT,
    row_index INTEGER,
    question TEXT,
    answer TEXT,
    difficulty TEXT,
    topic TEXT,
    chunk TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS processed_files (
    file_name TEXT PRIMARY KEY,
    processed_at INTEGER
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS vectors_map (
    vector_id INTEGER PRIMARY KEY,
    doc_id INTEGER
)
""")
conn.commit()

# faiss index
index = None
if INDEX_PATH.exists():
    try:
        index = faiss.read_index(INDEX_PATH.as_posix())
        print("Loaded existing FAISS index.")
    except Exception as e:
        print("Could not load index, creating new one:", e)

if index is None:
    base_index = faiss.IndexFlatL2(EMBED_DIM)
    index = faiss.IndexIDMap(base_index)
    print("Created new FAISS index.")

def save_index():
    faiss.write_index(index, INDEX_PATH.as_posix())
    print("FAISS index saved to disk.")

# helpers
def get_embeddings(texts: List[str]) -> List[List[float]]:
    results = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = openai.Embedding.create(model=EMBED_MODEL, input=batch)
        for item in resp["data"]:
            results.append(item["embedding"])
    return results

def normalize_row(row: pd.Series) -> Dict[str, str]:
    q = a = difficulty = topic = ""
    for k in ['question', 'Question', 'Q']:
        if k in row.index and pd.notna(row[k]):
            q = str(row[k]); break
    for k in ['answer', 'Answer', 'A']:
        if k in row.index and pd.notna(row[k]):
            a = str(row[k]); break
    for k in ['difficulty', 'Difficulty', 'level']:
        if k in row.index and pd.notna(row[k]):
            difficulty = str(row[k]); break
    for k in ['topic', 'Topic', 'chapter']:
        if k in row.index and pd.notna(row[k]):
            topic = str(row[k]); break
    return {"question": q.strip(), "answer": a.strip(),
            "difficulty": difficulty.strip(), "topic": topic.strip()}

def ingest_file(filepath: Path):
    print(f"Processing: {filepath.name}")
    try:
        if filepath.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        print("Failed to read file:", e)
        return

    cur = conn.cursor()
    cur.execute("SELECT file_name FROM processed_files WHERE file_name = ?", (filepath.name,))
    if cur.fetchone():
        print(f"Already processed {filepath.name}")
        return

    texts_to_embed = []
    metadata_rows = []
    for idx, row in df.iterrows():
        norm = normalize_row(row)
        q, a = norm["question"], norm["answer"]
        if not q and not a:
            continue
        chunk = f"Q: {q}\nA: {a}\nDifficulty: {norm['difficulty']}\nTopic: {norm['topic']}"
        texts_to_embed.append(chunk)
        metadata_rows.append((filepath.name, int(idx), q, a,
                              norm['difficulty'], norm['topic'], chunk))

    if not texts_to_embed:
        print("No valid rows in", filepath.name)
        cur.execute("INSERT OR REPLACE INTO processed_files VALUES (?,?)",
                    (filepath.name, int(time.time())))
        conn.commit()
        return

    print(f"Generating embeddings for {len(texts_to_embed)} rows...")
    embeddings = [np.array(e, dtype='float32') for e in get_embeddings(texts_to_embed)]

    cur.execute("SELECT MAX(vector_id) FROM vectors_map")
    r = cur.fetchone()[0]
    next_vector_id = (r + 1) if r is not None else 0

    for i, meta in enumerate(metadata_rows):
        file_name, row_index, q, a, diff, topic, chunk = meta
        cur.execute("INSERT INTO docs (file_name,row_index,question,answer,difficulty,topic,chunk)"
                    "VALUES (?,?,?,?,?,?,?)", (file_name, row_index, q, a, diff, topic, chunk))
        doc_id = cur.lastrowid
        vec_id = next_vector_id + i
        index.add_with_ids(np.array([embeddings[i]]), np.array([vec_id], dtype='int64'))
        cur.execute("INSERT INTO vectors_map (vector_id, doc_id) VALUES (?,?)",
                    (int(vec_id), int(doc_id)))

    cur.execute("INSERT OR REPLACE INTO processed_files VALUES (?,?)",
                (filepath.name, int(time.time())))
    conn.commit()
    save_index()
    print(f"Ingested {len(metadata_rows)} rows from {filepath.name}")

def ingest_all():
    files = [f for f in DATA_DIR.glob("*") if f.suffix.lower() in ['.xlsx', '.xls', '.csv']]
    for f in files:
        ingest_file(f)

class QbankHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory: return
        path = Path(event.src_path)
        if path.suffix.lower() in ['.xlsx', '.xls', '.csv']:
            time.sleep(1)
            ingest_file(path)

def start_watcher():
    observer = Observer()
    handler = QbankHandler()
    observer.schedule(handler, DATA_DIR.as_posix(), recursive=False)
    observer.start()
    print("Watching qbank folder:", DATA_DIR)
    return observer

def retrieve_similar(question: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    emb = np.array(get_embeddings([question])[0], dtype='float32').reshape(1, -1)
    if index.ntotal == 0:
        return []
    D, I = index.search(emb, k)
    results = []
    for vid in I[0].tolist():
        if vid == -1: continue
        cur.execute("SELECT doc_id FROM vectors_map WHERE vector_id = ?", (int(vid),))
        row = cur.fetchone()
        if not row: continue
        doc_id = row[0]
        cur.execute("SELECT question,answer,difficulty,topic,file_name FROM docs WHERE doc_id=?",
                    (doc_id,))
        r = cur.fetchone()
        if r:
            results.append({"vector_id": vid, "doc_id": doc_id, "question": r[0],
                            "answer": r[1], "difficulty": r[2], "topic": r[3],
                            "file_name": r[4]})
    return results

# FastAPI app + CORS + simple auth
app = FastAPI(title="RAASHI Backend")
origins = os.getenv("CORS_ORIGINS", "https://www.virtualself-tutor.com").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

class QueryIn(BaseModel):
    user_id: str = "anon"
    question: str

class QueryOut(BaseModel):
    answer: str
    suggestions: List[str]
    sources: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status":"ok","indexed_vectors": int(index.ntotal)}

@app.post("/ingest")
def trigger_ingest():
    ingest_all()
    return {"status":"ok","indexed_vectors": int(index.ntotal)}

@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn, authorization: str = Header(None)):
    # simple token auth
    if FRONTEND_TOKEN:
        if not authorization or not authorization.startswith("Bearer ") or authorization.split(" ",1)[1] != FRONTEND_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    neighbors = retrieve_similar(payload.question)
    context_parts = []
    sources = []
    for n in neighbors:
        context_parts.append(f"Topic: {n['topic']} | Difficulty: {n['difficulty']}\nQ: {n['question']}\nA: {n['answer']}\n---")
        sources.append({"doc_id": n["doc_id"], "file": n["file_name"], "difficulty": n["difficulty"]})
    context = "\n".join(context_parts) or "No context found."

    system_prompt = (
        "You are RAASHI, a friendly tutor assistant. "
        "Use the provided context to answer clearly in 2–6 sentences. "
        "Then give 3 short suggestions: (1) easier question, (2) similar difficulty, (3) study tip."
    )
    user_prompt = f"User: {payload.question}\n\nContext:\n{context}"

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    reply = resp["choices"][0]["message"]["content"].strip()
    lines = reply.splitlines()
    answer_lines = []; suggestions = []
    for ln in lines:
        if ln.strip().startswith(("1.", "-", "*")):
            suggestions.append(ln.strip())
        else:
            answer_lines.append(ln.strip())
    answer = " ".join(answer_lines[:5]).strip()
    if not suggestions:
        suggestions = [
            "Try a simpler question from the same chapter.",
            "Attempt another question of similar difficulty.",
            "Review the key concept summary before reattempting."
        ]
    return {"answer": answer, "suggestions": suggestions, "sources": sources}

if __name__ == "__main__":
    print("Starting RAASHI backend (local mode)...")
    ingest_all()
    observer = start_watcher()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
