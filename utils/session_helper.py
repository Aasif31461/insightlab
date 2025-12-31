import streamlit as st
import pickle
import os

SESSION_FILE = ".insightlab_session.pkl"

# Keys that define the core data state
KEYS_TO_PERSIST = [
    # Data
    "df", 
    "clean_df", 
    "all_datasets",
    "data_snapshots",
    
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

def load_session():
    """Load session state from disk if available."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "rb") as f:
                saved_state = pickle.load(f)
            
            for k, v in saved_state.items():
                st.session_state[k] = v
                
            # st.toast("🔄 Session Restored!", icon="💾") 
            # Commented out toast to avoid spam on every refresh
            return True
        except Exception as e:
            # If load fails (e.g. corruption), just ignore
            print(f"Failed to load session: {e}")
    return False

def save_session():
    """Save current session state to disk."""
    state_to_save = {}
    for k in KEYS_TO_PERSIST:
        if k in st.session_state:
            state_to_save[k] = st.session_state[k]
    
    try:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(state_to_save, f)
    except Exception as e:
        print(f"Failed to save session: {e}")

def clear_session():
    """Clear session logic (to be called by button)."""
    import shutil
    import tempfile
    
    # 1. Delete Persistence File
    if os.path.exists(SESSION_FILE):
        try:
            os.remove(SESSION_FILE)
        except: pass

    # 2. Cleanup Temp Directory
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except: pass
        
    # 3. Clear Streamlit Cache
    st.cache_data.clear()
    
    # 4. Clear Session State
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    
    # 5. Reset Temp Dir (Start fresh immediately)
    st.session_state.temp_dir = tempfile.mkdtemp()
    
    st.rerun()
