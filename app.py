import re
import pickle
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI, OpenAIError

# ==============================
# Streamlit App Title
# ==============================
st.set_page_config(page_title="NBFC Legal Advocate RAG Bot", layout="wide")
st.title("NBFC Legal Advocate RAG Bot 🤖")
st.write("Ask about LAN status, notices, or general legal queries!")

# ==============================
# Load Secrets
# ==============================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.error("⚠️ Please add OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

# ---------- Initialize OpenAI Client ----------
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except OpenAIError as e:
    st.error(f"⚠️ Failed to initialize OpenAI client: {e}")
    st.stop()

# ==============================
# Config
# ==============================
EMBED_CACHE = "qa_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_WORDS = 150

# ==============================
# Utility Functions
# ==============================
def _norm(s: str) -> str:
    return str(s).strip().lower()

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def _truncate_to_words(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words])

def log_error(msg: str):
    with open("error_log.txt", "a") as f:
        f.write(msg + "\n")

# ==============================
# File Upload (Streamlit Cloud compatible)
# ==============================
st.sidebar.header("Upload Excel Files (optional)")
qa_file = st.sidebar.file_uploader("Upload QA Excel", type=["xlsx"])
lan_file = st.sidebar.file_uploader("Upload LAN Excel", type=["xlsx"])

# ==============================
# Load QA & LAN Data
# ==============================
@st.cache_data
def load_qa(file):
    df = pd.read_excel(file)
    required_columns = {"id", "Business", "Question", "Answer"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        st.error(f"❌ Missing QA columns: {missing}")
        st.stop()
    return df[["id", "Business", "Question", "Answer"]]

@st.cache_data
def load_lan(file):
    df = pd.read_excel(file, dtype={"Lan Id": str})
    df.columns = df.columns.str.strip()
    required_columns = {"Lan Id", "Status", "Business", "Notice Sent Date"}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ LAN file missing columns: {required_columns}")
        st.stop()
    df["Lan Id"] = df["Lan Id"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Business"] = df["Business"].astype(str).str.strip()
    df["Notice Sent Date"] = pd.to_datetime(df["Notice Sent Date"], dayfirst=True, errors="coerce")
    return df

# Use uploaded files or default placeholders
qa_df = load_qa(qa_file) if qa_file else None
lan_df = load_lan(lan_file) if lan_file else None

# ==============================
# Embeddings
# ==============================
@st.cache_data(show_spinner=False)
def embed_texts(texts):
    vectors = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        except OpenAIError as e:
            st.error(f"Embedding API failed: {e}")
            log_error(f"Embedding API Error: {e}")
            return []
    return vectors

@st.cache_data(show_spinner=True)
def build_embeddings(df):
    corpus = [f"Business: {b}\nQ: {q}\nA: {a}" for q, a, b in zip(df["Question"], df["Answer"], df["Business"])]
    vecs = np.array(embed_texts(corpus), dtype=np.float32)
    return vecs

if qa_df is not None:
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE, "rb") as f:
            cache = pickle.load(f)
            qa_embeddings = cache["embeddings"]
    else:
        qa_embeddings = build_embeddings(qa_df)
        with open(EMBED_CACHE, "wb") as f:
            pickle.dump({"df": qa_df, "embeddings": qa_embeddings}, f)

# ==============================
# RAG Retrieval
# ==============================
def retrieve(query, df, embeddings, top_k=TOP_K):
    if embeddings.shape[0] == 0:
        return []
    q_vecs = embed_texts([query])
    if not q_vecs:
        return []
    q_vec = np.array(q_vecs[0], dtype=np.float32)
    sims = np.array([_cosine_sim(q_vec, emb) for emb in embeddings])
    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {"id": int(df.iloc[i]["id"]), "Question": df.iloc[i]["Question"], "Answer": df.iloc[i]["Answer"], "Business": df.iloc[i]["Business"], "Score": float(sims[i])}
        for i in top_idx
    ]

# ==============================
# LLM Answer
# ==============================
SYSTEM_ROLE = (
    "You are a Senior Legal Advocate with 15 years of experience advising NBFCs. "
    f"Answer questions using retrieved context concisely (≤{MAX_WORDS} words). "
    "Do not invent facts."
)

def llm_answer(query, contexts):
    if not contexts:
        return "No relevant context found."
    context_text = "\n\n".join([f"[DOC {c['id']}] Q: {c['Question']}\nA: {c['Answer']}" for c in contexts])
    user_prompt = f"User Query:\n{query}\n\nRetrieved Context:\n{context_text}\n\nAnswer in ≤{MAX_WORDS} words."
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": SYSTEM_ROLE}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=220
        )
        text = resp.choices[0].message.content.strip()
        return _truncate_to_words(text)
    except OpenAIError as e:
        st.error(f"LLM API call failed: {e}")
        log_error(f"LLM API Error: {e}")
        return "LLM call failed."

# ==============================
# LAN Helper
# ==============================
STAIRCASE_OFFSETS = {
    "pre arbitration notice": ("Arbitration Notice", 4),
    "arbitration notice": ("Arbitral Award", 5),
    "arbitral award": ("Execution Notice", 7),
    "reminder notice": ("Legal Follow-up", 3),
    "pre-sales": ("Post Sales", 4),
    "post sales": ("Closure", 5),
}

LEGAL_INITIATED_STATUSES = {"pre arbitration notice", "arbitration notice", "arbitral award", "execution notice", "reminder notice"}

def summarize_lan_record(row):
    last_status = row["Status"]
    last_date = row["Notice Sent Date"]
    next_name, next_offset = STAIRCASE_OFFSETS.get(_norm(last_status), (None, None))
    return {
        "lan_id": row["Lan Id"],
        "business": row["Business"],
        "current_legal_status": "Legal initiated" if _norm(last_status) in LEGAL_INITIATED_STATUSES else "Pre-legal",
        "last_notice_name": last_status,
        "last_notice_date": None if pd.isna(last_date) else last_date.strftime("%d/%m/%Y"),
        "next_notice_name": next_name,
        "next_notice_date": None if not next_offset or pd.isna(last_date) else (last_date + timedelta(days=next_offset)).strftime("%d/%m/%Y")
    }

# ==============================
# Main App Logic
# ==============================
query = st.text_input("Enter your query:")

if query:
    lan_id_match = re.search(r"\b\d{3,}\b", query)
    if lan_id_match and lan_df is not None:
        lan_id = lan_id_match.group(0)
        subset = lan_df[lan_df["Lan Id"].str.strip() == lan_id]
        if not subset.empty:
            row = subset.sort_values("Notice Sent Date", ascending=False).iloc[0]
            summary = summarize_lan_record(row)
            st.write(f"**LAN ID:** {summary['lan_id']}")
            st.write(f"**Business:** {summary['business']}")
            st.write(f"**Current Legal Status:** {summary['current_legal_status']}")
            st.write(f"**Last Notice:** {summary['last_notice_name']} on {summary['last_notice_date'] or 'N/A'}")
            st.write(f"**Next Notice:** {summary['next_notice_name'] or 'N/A'} on {summary['next_notice_date'] or 'N/A'}")
        else:
            st.warning(f"No LAN record found for {lan_id}")
    elif qa_df is not None:
        contexts = retrieve(query, qa_df, qa_embeddings)
        if contexts:
            st.text_area("Retrieved Context", value="\n\n".join([f"Q: {c['Question']}\nA: {c['Answer']}" for c in contexts]), height=250)
            answer_text = llm_answer(query, contexts)
            st.write("**Advocate Answer:**", answer_text)
        else:
            st.warning("⚠️ No relevant QA embeddings found.")
