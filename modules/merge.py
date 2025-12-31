import streamlit as st
import pandas as pd
from utils.data_loader import save_uploaded_file, load_dataset

def render_merge_studio():
    st.title("🔗 Merge Studio")
    st.markdown("Combine your main dataset with another file.")
    
    if st.session_state.df is None:
        st.warning("Please upload a Main Dataset in 'Data Upload' first.")
        st.stop()
        
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Main Dataset (Left)")
        st.dataframe(st.session_state.df.head(), height=150, width="stretch")
        st.caption(f"Rows: {st.session_state.df.shape[0]}, Cols: {st.session_state.df.shape[1]}")
        
    with c2:
        st.subheader("Secondary Dataset (Right)")
        
        # Option to select from existing uploads
        existing_files = [f for f in st.session_state.all_datasets.keys() if f != st.session_state.get('active_dataset_name')]
        
        merge_source = st.radio("Source", ["Existing File", "New Upload"], horizontal=True, label_visibility="collapsed")
        
        df2 = None
        if merge_source == "Existing File":
            if existing_files:
                selected_file_2 = st.selectbox("Select File", existing_files)
                path2 = st.session_state.all_datasets[selected_file_2]
                df2 = load_dataset(path2)
            else:
                st.info("No other files uploaded.")
        else:
            uploaded_file_2 = st.file_uploader("Upload CSV to merge", type=["csv", "xlsx"])
            if uploaded_file_2:
                try:
                    path_new = save_uploaded_file(uploaded_file_2)
                    if path_new:
                        df2 = load_dataset(path_new)
                        if uploaded_file_2.name not in st.session_state.all_datasets:
                                st.session_state.all_datasets[uploaded_file_2.name] = path_new
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        if df2 is not None:
            st.dataframe(df2.head(), height=150, width="stretch")
            st.caption(f"Rows: {df2.shape[0]}, Cols: {df2.shape[1]}")

    if st.session_state.df is not None and df2 is not None:
        st.divider()
        st.subheader("Configuration")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"], help="inner: Only match. left: Keep all main. right: Keep all secondary. outer: Keep all.")
        with col_cfg2:
            left_on = st.multiselect("Key Column (Main)", st.session_state.df.columns)
        with col_cfg3:
            right_on = st.multiselect("Key Column (Secondary)", df2.columns)
        
        if left_on and right_on: 
            if len(left_on) != len(right_on):
                st.error("Number of key columns must match.")
            else:
                if st.button("🔍 Preview Merge"):
                    try:
                        # Perform merge preview
                        merged_preview = pd.merge(st.session_state.df, df2, left_on=left_on, right_on=right_on, how=join_type)
                        st.subheader("Preview Result")
                        st.write(f"New Shape: {merged_preview.shape}")
                        st.dataframe(merged_preview.head(), width="stretch")
                    except Exception as e:
                        st.error(f"Merge failed: {e}")
                        
                if st.button("🚀 Apply Merge"):
                    try:
                        # Apply merge deeply
                        st.session_state.df = pd.merge(st.session_state.df, df2, left_on=left_on, right_on=right_on, how=join_type)
                        st.session_state.clean_df = st.session_state.df.copy()
                        st.success(f"Merged successfully! New shape: {st.session_state.df.shape}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Merge failed: {e}")
