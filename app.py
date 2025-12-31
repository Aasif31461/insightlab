import streamlit as st
import shutil
import tempfile
import os
import time

# --- Import Modules ---
from modules.data_upload import render_data_upload
from modules.merge import render_merge_studio
from modules.profiling import render_deep_profiling
from modules.preprocessing import render_preprocessing
from modules.ml_studio import render_ml_studio
from modules.conclusion import render_conclusion
from modules.model_runner import render_model_runner
from modules.smart_tools import render_smart_tools
from utils.session_helper import load_session_from_bytes, get_session_as_bytes, clear_session

# --- Page Config ---
st.set_page_config(
    page_title="InsightLab AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem;}
    h1 {color: #2c3e50;}
    h2 {color: #34495e;}
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)
    
def init_session_state():
    # Initialize Session State
    if 'df' not in st.session_state: st.session_state.df = None
    if 'clean_df' not in st.session_state: st.session_state.clean_df = None
    if 'model' not in st.session_state: st.session_state.model = None
    if 'all_datasets' not in st.session_state: st.session_state.all_datasets = {}
    if 'active_dataset_name' not in st.session_state: st.session_state.active_dataset_name = None
    if 'data_snapshots' not in st.session_state: st.session_state.data_snapshots = {}
    if 'external_model' not in st.session_state: st.session_state.external_model = None
    
    # Initialize Temp Dir
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

init_session_state()

# --- App Constants ---
PAGES = {
    "Data Loader": "1. Data Loader",
    "Data Cleaning": "2. Data Cleaning & Editor",
    "Exploratory Analysis": "3. Exploratory Analysis",
    "Model Builder": "4. Model Builder",
    "Model Runner": "🏃 Model Runner",
    "Merge Studio": "🛠️ Merge Studio",
    "Smart Tools": "🧰 Smart Tools",
    "Conclusion": "🏁 Conclusion"
}

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.title("InsightLab AI")
    st.caption("v2.0 • Data to Insight")
    
    if st.session_state.active_dataset_name:
        st.caption(f"Active: **{st.session_state.active_dataset_name}**")
        st.markdown("---")
        
        mode_type = st.radio("Mode", ["Pipeline", "Toolkit"], horizontal=True, label_visibility="collapsed", key="nav_mode")
        
        if mode_type == "Pipeline":
             final_selection = st.radio("Step", 
                 [PAGES["Data Loader"], PAGES["Data Cleaning"], PAGES["Exploratory Analysis"], PAGES["Model Builder"], PAGES["Conclusion"]],
                 key="nav_pipeline")
        else:
             final_selection = st.radio("Tool", 
                 [PAGES["Merge Studio"], PAGES["Model Runner"], PAGES["Smart Tools"]],
                 key="nav_toolkit")
                 
        st.info("💡 **Tip**: Follow the Pipeline steps 1-4.")
        
    else:
        # Default to Data Loader if no dataset
        final_selection = PAGES["Data Loader"]
        st.info("👆 Please upload a dataset to begin.")

    st.divider()
    
    # --- Session Manager ---
    with st.expander("💾 Session Manager"):
        st.caption("Save your progress to return later.")
        
        # Download
        session_data = get_session_as_bytes()
        if session_data:
            st.download_button(
                label="Download Session",
                data=session_data,
                file_name="insightlab_session.pkl",
                mime="application/octet-stream",
                help="Download your current work as a file."
            )
            
        # Upload
        uploaded_session = st.file_uploader("Restore Session", type=["pkl"], label_visibility="collapsed")
        if uploaded_session:
            if st.button("Load Session Data"):
                if load_session_from_bytes(uploaded_session):
                    st.success("Session loaded!")
                    time.sleep(0.5)
                    st.rerun()

    st.divider()
    if st.button("🔴 Reset App", help="Deletes all data and resets the app."):
        clear_session()

    # --- Versioning Sidebar ---
    if st.session_state.data_snapshots:
        st.divider()
        with st.expander("🗂️ Dataset Versioning", expanded=True):
            st.markdown("Switch between data stages:")
            versions = list(st.session_state.data_snapshots.keys())
            selected_version = st.radio("Current Version", versions, index=len(versions)-1, key="version_radio")
            
            if st.button("Restore Version"):
                st.session_state.df = st.session_state.data_snapshots[selected_version].copy()
                st.session_state.clean_df = st.session_state.df.copy() # Sync clean_df
                st.success(f"Restored {selected_version}!")
                time.sleep(0.5)
                st.rerun()

# ==========================
# ROUTING (Main Area)
# ==========================
try:
    if final_selection == PAGES["Data Loader"]:
        render_data_upload()
    
    elif final_selection == PAGES["Exploratory Analysis"]:
        render_deep_profiling()
    
    elif final_selection == PAGES["Data Cleaning"]:
        render_preprocessing()
    
    elif final_selection == PAGES["Model Builder"]:
        render_ml_studio()

    elif final_selection == PAGES["Merge Studio"]:
        render_merge_studio()

    elif final_selection == PAGES["Model Runner"]:
        render_model_runner()
        
    elif final_selection == PAGES["Smart Tools"]:
        render_smart_tools()
    
    elif final_selection == PAGES["Conclusion"]:
        render_conclusion()

except Exception as e:
    import traceback
    traceback.print_exc()
    st.error("An internal error occurred. Please check the logs.")