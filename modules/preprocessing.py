import streamlit as st
import pandas as pd
import numpy as np

def render_preprocessing():
    if st.session_state.clean_df is None:
        if st.session_state.df is not None:
             st.session_state.clean_df = st.session_state.df.copy()
        else:
            st.warning("Upload data first.")
            st.stop()
        
    st.title("🛠️ Data Cleaning & Editor")
    
    # Ensure we are working on the CLEAN dataframe
    df = st.session_state.clean_df
    
    # Tabs for modular cleaning
    tab_editor, tab_clean1, tab_clean2, tab_clean3, tab_clean4 = st.tabs([
        "✏️ Data Editor", 
        "🧹 Duplicate Removal", 
        "🔧 Data Cleaning Toolkit", 
        "⚙️ Feature Selection", 
        "📤 Export Data"
    ])
    
    # --- TAB 0: DATA EDITOR ---
    with tab_editor:
        st.subheader("Interactive Data Editor")
        
        # --- Search & Filter ---
        with st.expander("🔍 Search & Filter", expanded=True):
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                search_cols = st.multiselect("Search In", df.columns, placeholder="All Columns", key="ed_search_cols")
            with f_col2:
                search_term = st.text_input("Search Term", placeholder="Type to search...", key="ed_search_term")
        
        # helper to apply filter
        if search_term:
            try:
                if search_cols:
                    mask = df[search_cols].astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                else:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                df_display = df[mask].copy()
                st.caption(f"Showing {len(df_display)} rows matching '{search_term}'")
            except Exception as e:
                st.error(f"Search Error: {e}")
                df_display = df.copy()
        else:
            df_display = df.copy()

        # --- Sidebar Tools for Editor ---
        with st.expander("🛠️ Column & Request Tools", expanded=False):
            tab_col1, tab_col2, tab_col3 = st.tabs(["Add/Remove", "Rename", "Bulk Edit"])
            
            with tab_col1:
                st.markdown("**Add Column**")
                new_col_name = st.text_input("New Column Name", key="ed_new_col")
                new_col_val = st.text_input("Default Value", value="0", key="ed_new_val")
                
                def add_col():
                    if new_col_name and new_col_name not in st.session_state.clean_df.columns:
                        st.session_state.clean_df[new_col_name] = new_col_val
                        st.session_state.clean_df = st.session_state.clean_df.copy() # Defragment
                        st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                        st.success(f"Added {new_col_name}")
                    elif new_col_name in st.session_state.clean_df.columns:
                        st.error("Column exists.")
                
                st.button("➕ Add", on_click=add_col, key="btn_add_col")
                
                st.divider()
                st.markdown("**Delete Column**")
                col_to_del = st.selectbox("Select Column to Delete", df.columns, index=None, placeholder="Select column...", key="ed_del_col")
                
                def del_col():
                    if col_to_del:
                        st.session_state.clean_df = st.session_state.clean_df.drop(columns=[col_to_del]).copy()
                        st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                        st.success(f"Deleted {col_to_del}")

                st.button("🗑️ Delete", on_click=del_col, key="btn_del_col")

            with tab_col2:
                st.markdown("**Rename Column**")
                col_rename_target = st.selectbox("Select Column", df.columns, key="ed_ren_target")
                col_rename_new = st.text_input("New Name", key="ed_ren_new")
                
                def rename_col():
                    if col_rename_new and col_rename_new not in st.session_state.clean_df.columns:
                         st.session_state.clean_df = st.session_state.clean_df.rename(columns={col_rename_target: col_rename_new}).copy()
                         st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                         st.success(f"Renamed {col_rename_target} -> {col_rename_new}")
                    elif col_rename_new in st.session_state.clean_df.columns:
                         st.error("Name taken.")

                st.button("✏️ Rename", on_click=rename_col, key="btn_ren_col")

            with tab_col3:
                st.markdown("**Find & Replace**")
                target_col = st.selectbox("Target Column", df.columns, key="ed_bulk_col")
                find_val = st.text_input("Find Value", key="ed_find_val")
                replace_val = st.text_input("Replace With", key="ed_rep_val")
                
                def bulk_replace():
                    d = st.session_state.clean_df
                    if d[target_col].dtype == object or str(d[target_col].dtype) == 'string':
                         d[target_col] = d[target_col].replace(find_val, replace_val)
                    else:
                         try:
                             d[target_col] = d[target_col].replace(float(find_val), float(replace_val))
                         except:
                             pass
                    st.session_state.clean_df = d.copy()
                    st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()

                st.button("🔄 Replace All", on_click=bulk_replace, key="btn_bulk_rep")

        # --- Grid ---
        num_rows_mode = "dynamic" if not search_term else "fixed"
        if search_term:
            st.info("💡 Row addition is disabled while searching.")
        
        edited_df = st.data_editor(
            df_display,
            num_rows=num_rows_mode,
            width="stretch",
            key="clean_editor"
        )
        
        # Sync Logic
        if not edited_df.equals(df_display):
            if search_term:
                st.session_state.clean_df.update(edited_df)
            else:
                st.session_state.clean_df = edited_df.copy()
            st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()

    
    # --- TAB 1: DUPLICATE REMOVAL ---
    with tab_clean1:
        st.subheader("Manage Duplicates")
        dup_count = df.duplicated().sum()
        st.info(f"Current Total Duplicate Rows: {dup_count}")
        
        dup_mode = st.radio("Duplicate Check Mode", ["Entire Row", "Specific Columns"], horizontal=True)
        
        if dup_mode == "Entire Row":
                if st.button("Removing All Duplicates"):
                    st.session_state.clean_df = df.drop_duplicates()
                    st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                    st.success(f"Removed {dup_count} duplicates!")
                    st.rerun()
        else:
                subset_cols = st.multiselect("Select columns to identify duplicates", df.columns)
                if subset_cols:
                    subset_dup_count = df.duplicated(subset=subset_cols).sum()
                    st.write(f"Duplicates based on selection: {subset_dup_count}")
                    if st.button("Remove Subset Duplicates"):
                        st.session_state.clean_df = df.drop_duplicates(subset=subset_cols)
                        st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                        st.success(f"Removed {subset_dup_count} duplicates!")
                        st.rerun()
    
    # --- TAB 2: CLEANING TOOLKIT ---
    with tab_clean2:
        st.subheader("Granular Data Cleaning")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### 1. Rename Columns")
            col_to_rename = st.selectbox("Select Column to Rename", df.columns, key="clean_ren_sel")
            new_name = st.text_input("New Name", key="clean_ren_input")
            if st.button("Rename", key="clean_ren_btn"):
                if new_name:
                    st.session_state.clean_df = df.rename(columns={col_to_rename: new_name})
                    st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                    st.success(f"Renamed '{col_to_rename}' to '{new_name}'")
                    st.rerun()
        
        with c2:
            st.markdown("##### 2. Type Conversion")
            col_to_convert = st.selectbox("Select Column to Convert", df.columns, key="clean_conv_sel")
            new_type = st.selectbox("Target Type", ["datetime", "numeric", "category", "string"], key="clean_conv_type")
            if st.button("Convert Type", key="clean_conv_btn"):
                try:
                    temp_df = df.copy()
                    if new_type == "datetime":
                        temp_df[col_to_convert] = pd.to_datetime(temp_df[col_to_convert], errors='coerce')
                    elif new_type == "numeric":
                        temp_df[col_to_convert] = pd.to_numeric(temp_df[col_to_convert], errors='coerce')
                    elif new_type == "category":
                            temp_df[col_to_convert] = temp_df[col_to_convert].astype('category')
                    elif new_type == "string":
                            temp_df[col_to_convert] = temp_df[col_to_convert].astype(str)
                    st.session_state.clean_df = temp_df
                    st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                    st.success(f"Converted {col_to_convert} to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
        
        st.divider()
        st.markdown("##### 3. Granular Missing Value Handling")
        col_impute_select = st.selectbox("Target Column for Imputation", df.columns, key="clean_imp_sel")
        col_impute_strat = st.selectbox("Strategy", ["Mean", "Median", "Mode", "Custom", "Drop Rows"], key="gran_imp")
        
        if col_impute_strat == "Custom":
            custom_val = st.text_input("Enter Custom Value", key="clean_imp_val")
        
        if st.button("Apply to Column", key="clean_imp_btn"):
            temp_df = df.copy() # Use copy to avoid fragmentation
            if col_impute_strat == "Drop Rows":
                temp_df = temp_df.dropna(subset=[col_impute_select])
            elif col_impute_strat == "Mean" and pd.api.types.is_numeric_dtype(temp_df[col_impute_select]):
                temp_df[col_impute_select] = temp_df[col_impute_select].fillna(temp_df[col_impute_select].mean())
            elif col_impute_strat == "Median" and pd.api.types.is_numeric_dtype(temp_df[col_impute_select]):
                temp_df[col_impute_select] = temp_df[col_impute_select].fillna(temp_df[col_impute_select].median())
            elif col_impute_strat == "Mode":
                    temp_df[col_impute_select] = temp_df[col_impute_select].fillna(temp_df[col_impute_select].mode()[0])
            elif col_impute_strat == "Custom" and custom_val:
                    temp_df[col_impute_select] = temp_df[col_impute_select].fillna(custom_val)
            
            # De-fragment by copying if needed, though here we are assigning boolean indexing or full column
            st.session_state.clean_df = temp_df.copy() # Explicit copy to defragment
            st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
            st.success(f"Applied {col_impute_strat} to {col_impute_select}")
            st.rerun()

        st.divider()
        st.markdown("##### 4. Global Missing Value Removal")
        total_missing_rows = df.isnull().any(axis=1).sum()
        
        if total_missing_rows > 0:
            st.warning(f"Found **{total_missing_rows}** rows with missing values.")
            
            with st.expander("👀 Preview Rows to be Removed"):
                missing_rows = df[df.isnull().any(axis=1)]
                st.dataframe(missing_rows, width="stretch")
            
            if st.button("🗑️ Remove All Missing Value Rows", type="primary"):
                 st.session_state.clean_df = df.dropna()
                 st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
                 st.success(f"Removed {total_missing_rows} rows!")
                 st.rerun()
        else:
            st.success("No missing values found in the dataset.")

    # --- TAB 3: FEATURE SELECTION ---
    with tab_clean3:
        st.subheader("Feature Selection")
        all_cols = st.session_state.df.columns.tolist()
        current_cols = [c for c in st.session_state.clean_df.columns if c in all_cols]
        selected_cols = st.multiselect("Keep these columns", all_cols, default=current_cols)
        
        if st.button("Update Columns"):
            cols_to_keep = selected_cols
            restored_cols = [c for c in cols_to_keep if c not in st.session_state.clean_df.columns]
            if restored_cols:
                restored_data = st.session_state.df.loc[st.session_state.clean_df.index, restored_cols]
                df_final = pd.concat([st.session_state.clean_df, restored_data], axis=1)
            else:
                df_final = st.session_state.clean_df
            
            df_final = df_final[cols_to_keep]
            st.session_state.clean_df = df_final.copy() # Copy to defragment
            st.session_state.data_snapshots['Cleaned'] = st.session_state.clean_df.copy()
            st.success("Features Updated!")
            st.rerun()

    # --- TAB 4: EXPORT ---
    with tab_clean4:
        st.subheader("Export Cleaned Data")
        file_name = st.text_input("File Name", "cleaned_data")
        file_format = st.selectbox("Format", ["CSV", "Excel"])
        
        if file_format == "CSV":
            data_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data_csv, f"{file_name}.csv", "text/csv")
        else:
            # Requires openpyxl
            pass 
            # Keeping it simple for now as csv is safer
            st.info("Excel export requires additional dependencies. CSV is recommended.")

    st.divider()
    st.subheader("Current Data Preview")
    st.dataframe(st.session_state.clean_df, width="stretch")

    st.divider()
    col_next = st.columns([3, 1])[1]
    
    def go_next():
        st.session_state.nav_pipeline = "3. Exploratory Analysis"
        
    st.button("Proceed to Exploratory Analysis ➡️", type="primary", on_click=go_next)
