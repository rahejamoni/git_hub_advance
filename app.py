import os
import re
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI, OpenAIError
# abcs
# ==============================
# Streamlit App Title
# ==============================
st.title("NBFC Legal Advocate RAG Bot 🤖")
st.write("Ask about LAN status, notices, or general legal queries!")

# ==============================
# Load Secrets
# ==============================
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    EXCEL_QA_PATH = st.secrets["EXCEL_QA_PATH"]
    EXCEL_LAN_PATH = st.secrets["EXCEL_LAN_PATH"]
except KeyError as e:
    st.error(f"⚠️ Missing secret: {e}")
    st.stop()

LOG_FILE = st.secrets.get("LOG_FILE", "error_log.txt")

# ---------- Initialize OpenAI Client ----------
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except OpenAIError as e:
    st.error(f"⚠️ Failed to initialize OpenAI client: {e}")
    st.stop()

# ==============================
# Utility Functions
# ==============================
def _norm(s: str) -> str:
    return str(s).strip().lower()

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

# ==============================
# Load QA Excel
# ==============================
def load_qa(path: str):
    if not os.path.exists(path):
        st.error(f"⚠️ QA file not found: {path}")
        st.stop()
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    required_columns = {"id", "Business", "Question", "Answer"}
    missing = required_columns - set(df.columns)
    if missing:
        st.error(f"Missing required columns in QA file: {missing}")
        st.stop()
    return df[["id", "Business", "Question", "Answer"]]

# ==============================
# Load LAN Excel
# ==============================
def load_lan_status(path: str):
    if not path or not os.path.exists(path):
        st.warning(f"⚠️ LAN file not found: {path}")
        return None
    df = pd.read_excel(path, dtype={"Lan Id": str})
    df.columns = df.columns.str.strip()
    required_columns = {"Lan Id", "Status", "Business", "Notice Sent Date"}
    if not required_columns.issubset(set(df.columns)):
        st.error(f"LAN file must have columns: {required_columns}")
        st.stop()
    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Business"] = df["Business"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], dayfirst=True, errors="coerce")
    return df

# ==============================
# Embeddings
# ==============================
EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5

def embed_texts(texts):
    vectors = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        except OpenAIError as e:
            st.error(f"⚠️ OpenAI Embedding API failed: {e}")
            return []
    return vectors

def build_or_load_embeddings(excel_path=EXCEL_QA_PATH, cache_path=EMBED_CACHE):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)
        df = load_qa(excel_path)
        if saved.get("csv_len") == len(df):
            return saved["df"], saved["embeddings"]
    df = load_qa(excel_path)
    corpus = [f"Business: {b}\nQ: {q}\nA: {a}" for q, a, b in zip(df["Question"], df["Answer"], df["Business"])]
    vecs = np.array(embed_texts(corpus), dtype=np.float32)
    with open(cache_path, "wb") as f:
        pickle.dump({"df": df, "embeddings": vecs, "csv_len": len(df)}, f)
    return df, vecs

# ==============================
# RAG Retrieval
# ==============================
def retrieve(query, df, embeddings, top_k=TOP_K):
    q_vecs = embed_texts([query])
    if not q_vecs:
        return []
    q_vec = np.array(q_vecs[0], dtype=np.float32)
    sims = np.array([_cosine_sim(q_vec, emb) for emb in embeddings])
    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {
            "id": int(df.iloc[i]["id"]),
            "Question": df.iloc[i]["Question"],
            "Answer": df.iloc[i]["Answer"],
            "Business": df.iloc[i]["Business"],
            "Score": float(sims[i]),
        }
        for i in top_idx
    ]

# ==============================
# Load data once
# ==============================
qa_df, qa_embeddings = build_or_load_embeddings(EXCEL_QA_PATH)
lan_df = load_lan_status(EXCEL_LAN_PATH)

# ==============================
# User Input
# ==============================
query = st.text_input("Enter your query:")

if query:
    lan_id_match = re.search(r"\b\d{3,}\b", query)
    if lan_id_match and lan_df is not None:
        lan_id = lan_id_match.group(0)
        subset = lan_df[lan_df["Lan Id"].str.strip() == lan_id]
        if not subset.empty:
            row = subset.sort_values("Notice Sent Date", ascending=False).iloc[0]
            st.write(f"**LAN ID:** {row['Lan Id']}")
            st.write(f"**Business:** {row['Business']}")
            last_notice = row['Notice Sent Date'].strftime('%d/%m/%Y') if pd.notna(row['Notice Sent Date']) else 'N/A'
            st.write(f"**Last Notice:** {row['Status']} on {last_notice}")
        else:
            st.warning(f"No LAN record found for {lan_id}")
    else:
        contexts = retrieve(query, qa_df, qa_embeddings)
        if contexts:
            context_text = "\n\n".join([f"Q: {c['Question']}\nA: {c['Answer']}" for c in contexts])
            st.text_area("Retrieved Context", value=context_text, height=200)
        else:
            st.warning("⚠️ Could not retrieve embeddings. Check your OpenAI API key or connectivity.")
