import streamlit as st
import pandas as pd
import os
import time
from utils.data_loader import save_uploaded_file, load_dataset
import numpy as np

def render_data_upload():
    st.title("📂 Data Ingestion")
    st.markdown("Start by uploading your dataset(s). Supported formats: **CSV, Excel**.")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader("Upload Files", type=["csv", "xlsx"], accept_multiple_files=True)
    with col2:
        url = st.text_input("Or Import via URL")
        if url:
            if st.button("Load URL"):
                try:
                    url_df = pd.read_csv(url)
                    fname = url.split("/")[-1] or "url_dataset"
                    # Save dataframe to temp csv
                    path = os.path.join(st.session_state.temp_dir, f"{fname}.csv")
                    url_df.to_csv(path, index=False)
                    st.session_state.all_datasets[fname] = path
                    st.session_state.all_datasets[fname] = path # Redundant logic in original, keeping safe
                    st.success(f"Loaded {fname}!")
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")

    # Process Uploads
    if uploaded_files:
        st.subheader("Processing Files...")
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            if file.name not in st.session_state.all_datasets:
                    path = save_uploaded_file(file)
                    if path:
                        st.session_state.all_datasets[file.name] = path
            
            # Update Progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        time.sleep(0.5)
        progress_bar.empty()
        st.toast(f"Processed {len(uploaded_files)} files successfully!", icon="✅")

    # Dataset Selector & Preview
    dataset_names = list(st.session_state.all_datasets.keys())
    
    if not dataset_names:
        st.info("👋 Welcome! Please upload a dataset to begin.")
    else:
        st.divider()
        col_sel, col_info = st.columns([2, 1])
        
        with col_sel:
            # Determine index
            current_idx = 0
            if st.session_state.active_dataset_name in dataset_names:
                current_idx = dataset_names.index(st.session_state.active_dataset_name)
            
            selected_name = st.selectbox("Choose Active Dataset", dataset_names, index=current_idx)
            
            # Switch Logic
            if 'active_dataset_name' not in st.session_state or st.session_state.active_dataset_name != selected_name:
                try:
                    path = st.session_state.all_datasets[selected_name]
                    st.session_state.df = load_dataset(path)
                    st.session_state.active_dataset_name = selected_name
                    st.session_state.clean_df = st.session_state.df.copy()
                    # Snapshot
                    st.session_state.data_snapshots = {'Original': st.session_state.df.copy()}
                    
                    st.session_state.model = None
                    st.session_state.model_history = [] 
                    st.success(f"Switched to {selected_name}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load {selected_name}: {e}")
        
        # Preview (Active DF)
        if st.session_state.df is not None:
                st.divider()
                st.subheader("📊 Data Snapshot")
                
                # 1. High-Level Metrics
                r, c = st.session_state.df.shape
                duplicates = st.session_state.df.duplicated().sum()
                total_cells = r * c
                missing_cells = st.session_state.df.isnull().sum().sum()
                missing_pct = (missing_cells / total_cells) * 100
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Rows", f"{r:,}")
                m2.metric("Columns", f"{c}")
                m3.metric("Duplicates", f"{duplicates}", delta="⚠️" if duplicates > 0 else "Good", delta_color="inverse")
                m4.metric("Missing Values", f"{missing_cells:,} ({missing_pct:.1f}%)", delta="⚠️" if missing_cells > 0 else "Clean", delta_color="inverse")
                m5.metric("Memory", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                
                # 2. Detailed Column Analysis (The "Beautiful" Table)
                st.write("### 🔍 Column Insights")
                
                # Build Summary Data
                summary_data = []
                for col in st.session_state.df.columns:
                    series = st.session_state.df[col]
                    dtype = str(series.dtype)
                    n_unique = series.nunique()
                    n_missing = series.isnull().sum()
                    pct_missing = (n_missing / len(st.session_state.df)) * 100
                    
                    # Calculate Stats
                    if pd.api.types.is_numeric_dtype(series):
                        min_v = series.min()
                        max_v = series.max()
                        mean_v = series.mean()
                        median_v = series.median()
                        mode_v = series.mode()[0] if not series.mode().empty else np.nan
                        
                        stats = f"Min: {min_v:.2f} | Max: {max_v:.2f} | Mean: {mean_v:.2f}"
                    else:
                        min_v, max_v, mean_v, median_v = None, None, None, None
                        # Top value as mode for categorical
                        try:
                            mode_v = series.mode()[0] if not series.mode().empty else "-"
                        except:
                            mode_v = "-"
                        stats = f"Top: {mode_v} (Freq: {series.value_counts().iloc[0] if not series.value_counts().empty else 0})"

                    # Format Missing
                    missing_str = f"{n_missing} ({pct_missing:.1f}%)"
                    
                    # Trend Data (Downsampled for Sparkline)
                    trend_data = []
                    if pd.api.types.is_numeric_dtype(series):
                        # Take approx 30 points
                        step = max(1, len(series) // 30)
                        trend_data = series.iloc[::step].fillna(0).tolist()

                    summary_data.append({
                        "Column": col,
                        "Trend": trend_data,
                        "Type": dtype,
                        "Unique": n_unique,
                        "Missing": missing_str,
                        "Mode": str(mode_v),
                        "Min": float(min_v) if min_v is not None else None,
                        "Max": float(max_v) if max_v is not None else None,
                        "Mean": float(mean_v) if mean_v is not None else None,
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                st.dataframe(
                    summary_df,
                    column_config={
                        "Column": st.column_config.TextColumn("Feature", help="Name of the column"),
                        "Trend": st.column_config.LineChartColumn("Trend / Dist", y_min=0, y_max=None, width="small"),
                        "Type": st.column_config.TextColumn("Dtype", width="small"),
                        "Unique": st.column_config.NumberColumn("Unique", help="Number of unique values", format="%d"),
                        "Missing": st.column_config.TextColumn("Missing", help="Count (Percentage)"),
                        "Mode": st.column_config.TextColumn("Mode/Top", width="medium"),
                        "Min": st.column_config.NumberColumn("Min", format="%.2f"),
                        "Max": st.column_config.NumberColumn("Max", format="%.2f"),
                        "Mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                    },
                    hide_index=True,
                    width="stretch"
                )
                
                # Raw Data Expansion
                with st.expander("📄 View Raw Data"):
                    st.dataframe(st.session_state.df, width="stretch")
        
        st.divider()
        col_next = st.columns([3, 1])[1]
        
        def go_next():
            st.session_state.nav_pipeline = "2. Data Cleaning & Editor"
            
        st.button("Proceed to Data Cleaning & Editor ➡️", type="primary", on_click=go_next)
