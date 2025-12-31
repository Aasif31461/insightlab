import streamlit as st
import pandas as pd
import hashlib
import io

def render_smart_tools():
    st.title("🧰 Smart Tools")
    st.caption("Advanced operations for your active dataset.")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the 'Data Loader' first.")
        st.stop()
        
    df = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.df
    
    tab_pivot, tab_compare, tab_pii = st.tabs(["🔄 Pivot Studio", "⚖️ Data Comparer", "🛡️ PII Anonymizer"])
    
    # ==========================
    # 1. Pivot Studio
    # ==========================
    with tab_pivot:
        st.subheader("Pivot Table Studio")
        st.info("Reshape and aggregate your data.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            index_col = st.multiselect("Rows (Index)", df.columns)
        with c2:
            cols_col = st.multiselect("Columns", df.columns)
        with c3:
            values_col = st.multiselect("Values (Metrics)", df.select_dtypes(include='number').columns)
            
        agg_func = st.selectbox("Aggregation Function", ["sum", "mean", "count", "min", "max", "std"])
        margins = st.checkbox("Show Totals (Margins)", value=True)
        
        if index_col and values_col:
            try:
                pivot_df = pd.pivot_table(
                    df, 
                    index=index_col, 
                    columns=cols_col if cols_col else None, 
                    values=values_col, 
                    aggfunc=agg_func,
                    margins=margins
                )
                
                st.write(f"**Result Preview** ({pivot_df.shape[0]} rows)")
                st.dataframe(pivot_df, width="stretch")
                
                # Download
                csv = pivot_df.to_csv().encode('utf-8')
                st.download_button("⬇️ Download Pivot Table", csv, "pivot_table.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Could not generate pivot table: {e}")
        else:
            st.info("Select at least one Row and one Value column to generate a table.")

    # ==========================
    # 2. Data Comparer
    # ==========================
    with tab_compare:
        st.subheader("Dataset Diff Tool")
        st.caption("Compare your current 'Cleaned' data against the Original or another file.")
        
        # Source B Selection
        compare_options = ["Original Upload (Snapshot)"] + (["New File Upload"] if True else [])
        source_b_type = st.radio("Compare against:", compare_options, horizontal=True)
        
        df_b = None
        
        if source_b_type == "Original Upload (Snapshot)":
            if 'Original' in st.session_state.data_snapshots:
                df_b = st.session_state.data_snapshots['Original']
                st.success(f"Loaded Original Snapshot: {df_b.shape}")
            else:
                st.warning("Original snapshot not found. Did you modify the upload flow?")
        else:
            uploaded_file = st.file_uploader("Upload Comparison File", type=['csv', 'xlsx'])
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df_b = pd.read_csv(uploaded_file)
                else:
                    df_b = pd.read_excel(uploaded_file)
        
        if df_b is not None:
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Current Dataset Rows", df.shape[0])
            c2.metric("Comparison Dataset Rows", df_b.shape[0], delta=df.shape[0] - df_b.shape[0])
            
            # Schema Diff
            cols_a = set(df.columns)
            cols_b = set(df_b.columns)
            
            added = cols_a - cols_b
            removed = cols_b - cols_a
            
            if added: st.success(f"🆕 Added Columns: {', '.join(added)}")
            if removed: st.error(f"❌ Removed Columns: {', '.join(removed)}")
            if not added and not removed: st.info("✅ Schema matches (Same columns).")
            
            # Simple content check (if schema matches)
            if not added and not removed and df.shape == df_b.shape:
                if df.equals(df_b):
                    st.success("🎉 datasets are Identical!")
                else:
                    st.warning("⚠️ Content differences detected.")
                    
                    # Highlight diffs (basic)
                    if st.checkbox("Show specific value changes (Slow for large data)"):
                        try:
                            # Align indices? 
                            # Just compares cell by cell if sizes match
                           compare_mask = df.ne(df_b)
                           st.write("Cells that differ:")
                           st.write(compare_mask.sum())
                        except:
                            st.error("Could not run cell-by-cell comparison. Indices might differ.")

    # ==========================
    # 3. PII Anonymizer
    # ==========================
    with tab_pii:
        st.subheader("Data Anonymizer")
        st.caption("Secure your data for sharing.")
        
        target_cols = st.multiselect("Select Sensitive Columns (to mask)", df.columns)
        method = st.radio("Anonymization Method", ["Hash (SHA256)", "Redact ([REDACTED])", "Mask Email/ID (***)"], horizontal=True)
        
        if target_cols and st.button("Generate Safe Dataset"):
            safe_df = df.copy()
            
            for col in target_cols:
                safe_df[col] = safe_df[col].astype(str)
                
                if method == "Hash (SHA256)":
                    safe_df[col] = safe_df[col].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
                elif method == "Redact ([REDACTED])":
                    safe_df[col] = "[REDACTED]"
                elif method.startswith("Mask"):
                    # Keep first 2 chars
                    safe_df[col] = safe_df[col].apply(lambda x: x[:2] + "***" if len(x) > 2 else "***")
            
            st.dataframe(safe_df.head())
            
            # Download
            csv_safe = safe_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Anonymized CSV", csv_safe, "safe_data.csv", "text/csv")
