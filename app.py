import os
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import timedelta
from openai import OpenAI
from dotenv import load_dotenv
import difflib
import re

# ===== LOAD ENV =====
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXCEL_QA_PATH = os.getenv("EXCEL_QA_PATH")
EXCEL_LAN_PATH = os.getenv("LAN_DATA_PATH")
LOG_FILE      = os.getenv("LOG_FILE", "error_log.txt")




if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found in .env file")
    st.stop()
if not EXCEL_QA_PATH or not os.path.exists(EXCEL_QA_PATH):
    st.error(f"⚠️ EXCEL_QA_PATH is missing or invalid: {EXCEL_QA_PATH}")
    st.stop()
if not EXCEL_LAN_PATH or not os.path.exists(EXCEL_LAN_PATH):
    st.warning(f"⚠️ LAN data file not found: {EXCEL_LAN_PATH}")
    EXCEL_LAN_PATH = None  # handle gracefully later

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== UTILS =====
def _norm(s: str) -> str:
    return str(s).strip().lower()

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

# ===== LOAD Q&A =====
def load_qa(path: str):
    df = pd.read_excel(path)
  #  df = df.rename(columns={"questions":"Question","answers":"Answer","Business":"Business"})
    if "Business" not in df.columns:
        df["Business"] = ""
    #df["id"] = np.arange(len(df))
    return df[["id","Question","Answer","Business"]]

# ===== LOAD LAN STATUS =====
def load_lan_status(path: str):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_excel(path, dtype={"Lan Id": str})
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Business"] = df["Business"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], dayfirst=True, errors="coerce")
    return df

# ===== EMBEDDINGS =====
EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
TOP_K       = 5
MAX_WORDS   = 150

def embed_texts(texts):
    BATCH = 128
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors

def build_or_load_embeddings(excel_path=EXCEL_QA_PATH, cache_path=EMBED_CACHE):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)
        df = load_qa(excel_path)
        if saved.get("csv_len") == len(df):
            return saved["df"], saved["embeddings"]

    df = load_qa(excel_path)
    corpus = [f"Business: {b}\nQ: {q}\nA: {a}" for q,a,b in zip(df["Question"], df["Answer"], df["Business"])]
    vecs = np.array(embed_texts(corpus), dtype=np.float32)

    with open(cache_path, "wb") as f:
        pickle.dump({"df": df, "embeddings": vecs, "csv_len": len(df)}, f)

    return df, vecs

# ===== RAG RETRIEVE =====
def retrieve(query, df, embeddings, top_k=TOP_K):
    q_vec = np.array(embed_texts([query])[0], dtype=np.float32)
    sims = np.array([_cosine_sim(q_vec, emb) for emb in embeddings])
    top_idx = sims.argsort()[::-1][:top_k]
    return [{"id": int(df.iloc[i]["id"]),
             "Question": df.iloc[i]["Question"],
             "Answer": df.iloc[i]["Answer"],
             "Business": df.iloc[i]["Business"],
             "Score": float(sims[i])} for i in top_idx]

# ===== STREAMLIT APP =====
st.title("NBFC Legal Advocate RAG Bot 🤖")
st.write("Ask about LAN status, notices, or general legal queries!")

# Load embeddings and LAN data once
qa_df, qa_embeddings = build_or_load_embeddings(EXCEL_QA_PATH)
lan_df = load_lan_status(EXCEL_LAN_PATH)

query = st.text_input("Enter your query:")

if query:
    # Check if user is asking about LAN ID
    lan_id_match = re.search(r"\b\d{3,}\b", query)
    if lan_id_match and lan_df is not None:
        lan_id = lan_id_match.group(0)
        subset = lan_df[lan_df["Lan Id"].str.strip() == lan_id]
        if not subset.empty:
            row = subset.sort_values("Notice Sent Date", ascending=False).iloc[0]
            st.write(f"**LAN ID:** {row['Lan Id']}")
            st.write(f"**Business:** {row['Business']}")
            st.write(f"**Last Notice:** {row['Status']} on {row['Notice Sent Date'].strftime('%d/%m/%Y') if pd.notna(row['Notice Sent Date']) else 'N/A'}")
        else:
            st.warning(f"No LAN record found for {lan_id}")
    else:
        # Normal RAG query
        contexts = retrieve(query, qa_df, qa_embeddings)
        context_text = "\n\n".join([f"Q: {c['Question']}\nA: {c['Answer']}" for c in contexts])
        st.text_area("Retrieved Context", value=context_text, height=200)
