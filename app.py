import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

st.title("I love you bubbu")
st.write("mohit")  


# Load the .env file
load_dotenv()

# Fetch environment variables
lan_path = os.getenv("LAN_DATA_PATH")
qa_path = os.getenv("EXCEL_QA_PATH")

# Debugging - show on Streamlit page
st.write(f"LAN_DATA_PATH loaded as: {lan_path}")
st.write(f"EXCEL_QA_PATH loaded as: {qa_path}")

# Try to read the LAN data Excel file
if lan_path and os.path.exists(lan_path):
    df = pd.read_excel(lan_path)
    st.write("First 5 rows of LAN Data:")
    st.dataframe(df.head())
else:
    st.error(f"LAN data file not found at: {lan_path}")
