import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# Load from .env (local)
load_dotenv()

# Load from Streamlit secrets (cloud)
LAN_DATA_PATH = st.secrets.get("LAN_DATA_PATH", os.getenv("LAN_DATA_PATH"))
EXCEL_QA_PATH = st.secrets.get("EXCEL_QA_PATH", os.getenv("EXCEL_QA_PATH"))
LOG_FILE = st.secrets.get("LOG_FILE", os.getenv("LOG_FILE"))

st.write(f"LAN data path loaded: {LAN_DATA_PATH}")  # Debugging
