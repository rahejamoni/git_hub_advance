import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
from openai import OpenAI, OpenAIError

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
except KeyError:
    st.error("⚠️ Missing secret: OPENAI_API_KEY. Please add it under Streamlit Cloud → Settings → Secrets.")
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
TOP_K = 5
MAX_WORDS = 150
CHAT_MODEL = "gpt-4o-mini"

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

LEGAL_INITIATED_STATUSES = {
    "pre arbitration notice", "arbitration notice",
    "arbitral award", "execution notice", "reminder notice"
}

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
# Load Uploaded Files
# ==============================
uploaded_qa = st.file_uploader("Upload QA Excel", type=["xlsx"])
uploaded_emb = st.file_uploader("Upload Embeddings (.pkl)", type=["pkl"])
uploaded_lan = st.file_uploader("Upload LAN Excel", type=["xlsx"])

qa_df, qa_embeddings, lan_df = None, None, None

if uploaded_qa and uploaded_emb:
    qa_df = pd.read_excel(uploaded_qa)
    with open(uploaded_emb, "rb") as f:
        emb_cache = pickle.load(f)
    qa_embeddings = emb_cache["embeddings"]
    st.success("✅ QA and embeddings loaded successfully")

if uploaded_lan:
    lan_df = pd.read_excel(uploaded_lan, dtype={"Lan Id": str})
    st.success("✅ LAN data loaded successfully")

# ==============================
# RAG Retrieval
# ==============================
def retrieve(query, df, embeddings, top_k=TOP_K):
    if embeddings.shape[0] == 0:
        st.error("⚠️ No QA embeddings available.")
        return []

    # Generate query embedding
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
    except OpenAIError as e:
        st.error(f"⚠️ OpenAI Embedding API failed: {e}")
        return []

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
# LLM Answer Generation
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
            messages=[
                {"role": "system", "content": SYSTEM_ROLE},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=220
        )
        text = resp.choices[0].message.content.strip()
        if len(text.split()) > MAX_WORDS:
            text = _truncate_to_words(text, MAX_WORDS)
        return text
    except OpenAIError as e:
        st.error(f"⚠️ OpenAI LLM call failed: {e}")
        return "LLM call failed."

# ==============================
# Main Query Logic
# ==============================
query = st.text_input("Enter your query:")

if query:
    # Check for LAN ID in query
    if lan_df is not None:
        lan_id_match = re.search(r"\b\d{3,}\b", query)
        if lan_id_match:
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

    # RAG QA retrieval
    if qa_df is not None and qa_embeddings is not None:
        contexts = retrieve(query, qa_df, qa_embeddings)
        if contexts:
            st.text_area(
                "Retrieved Context",
                value="\n\n".join([f"Q: {c['Question']}\nA: {c['Answer']}" for c in contexts]),
                height=250
            )
            answer_text = llm_answer(query, contexts)
            st.write("**Advocate Answer:**", answer_text)
        else:
            st.warning("⚠️ No relevant QA embeddings found.")
