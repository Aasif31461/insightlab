import streamlit as st
import pickle
import io

# Keys that define the core data state
KEYS_TO_PERSIST = [
    # Data
    "df", 
    "clean_df", 
    "all_datasets",
    "data_snapshots",
    "active_dataset_name",
    
    # ML
    "model", 
    "X_train", 
    "X_test",
    "y_train", 
    "y_test",
    "target_col", 
    "problem_type",
    
    # Navigation/State
    "nav_mode",
    "nav_pipeline",
    "nav_toolkit"
]

def load_session_from_bytes(uploaded_file):
    """Load session state from an uploaded bytes-like object."""
    try:
        # Check if it's a file-like object or raw bytes
        if hasattr(uploaded_file, 'read'):
            content = uploaded_file.read()
        else:
            content = uploaded_file
            
        saved_state = pickle.loads(content)
        
        for k, v in saved_state.items():
            st.session_state[k] = v
            
        return True
    except Exception as e:
        print(f"Failed to load session: {e}")
        return False

def get_session_as_bytes():
    """Serialize current session state to bytes for download."""
    state_to_save = {}
    for k in KEYS_TO_PERSIST:
        if k in st.session_state:
            state_to_save[k] = st.session_state[k]
    
    try:
        buffer = io.BytesIO()
        pickle.dump(state_to_save, buffer)
        return buffer.getvalue()
    except Exception as e:
        print(f"Failed to serialize session: {e}")
        return None

def clear_session():
    """Clear session logic (to be called by button)."""
    import shutil
    import tempfile
    import os
    
    # 1. Cleanup Temp Directory
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except: pass
        
    # 2. Clear Streamlit Cache
    st.cache_data.clear()
    
    # 3. Clear Session State
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    
    # 4. Reset Temp Dir (Start fresh immediately)
    st.session_state.temp_dir = tempfile.mkdtemp()
    
    st.rerun()
