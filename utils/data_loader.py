import streamlit as st
import pandas as pd
import os
import tempfile

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to temp dir and returns path."""
    try:
        # Re-create if deleted
        if 'temp_dir' not in st.session_state or not os.path.exists(st.session_state.temp_dir):
             st.session_state.temp_dir = tempfile.mkdtemp()
             
        file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

@st.cache_data
def load_dataset(file_path):
    """Loads dataset from path with caching."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    return None
