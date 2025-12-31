import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import generate_story

def render_deep_profiling():
    if st.session_state.df is None:
        st.warning("Please upload data first.")
        st.stop()
    
    df = st.session_state.df
    st.title("🚀 Exploratory Data Analysis")

    # Tabs for better organization
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview & Stats", "📈 Chart Builder", "Multivariate Analysis", "AI Insights", "Advanced Visualizations", "✅ Data Quality"])

    # --- TAB 1: OVERVIEW & STATS ---
    with tab1:
        st.subheader("Statistical Snapshot")
        st.info("""
        📊 **Cheat Sheet:**
        - **Count**: Total rows.
        - **Mean**: Average.
        - **Median (50%)**: Middle value.
        - **Std**: Spread (Variance).
        - **Skew**: Asymmetry (0 = Normal).
        - **Kurtosis**: Peakiness (High = Outliers).
        """)
        desc = df.describe().T
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            desc['skew'] = df[numeric_cols].skew()
            desc['kurtosis'] = df[numeric_cols].kurtosis()
        st.dataframe(desc.style.format("{:.2f}").background_gradient(cmap='Blues'), width="stretch")

        st.divider()

        st.subheader("Missing Values")
        if df.isnull().sum().sum() > 0:
            fig_null = px.imshow(df.isnull(), color_continuous_scale='gray', aspect='auto', title="Missing Map")
            st.plotly_chart(fig_null, width="stretch")
            st.warning(f"⚠️ {df.isnull().sum().sum()} missing values detected.")
        else:
            st.metric("Missing Values", "0", delta="Clean")
            st.success("Dataset is clean.")
        
        st.divider()
        
        st.subheader("Correlation Matrix")
        if not numeric_cols.empty:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig_corr, width="stretch")
        else:
            st.info("No numeric columns for correlation.")

        st.divider()

        st.subheader("Categorical Value Counts")
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if cat_cols:
            target_cat = st.selectbox("Select Column to Inspect", cat_cols)
            val_counts = df[target_cat].value_counts().reset_index()
            val_counts.columns = [target_cat, 'Count']
            fig_cat = px.bar(val_counts, x=target_cat, y='Count', title=f"Frequency of {target_cat}")
            st.plotly_chart(fig_cat, width="stretch")
        else:
            st.info("No categorical columns found.")

    # --- TAB 2: CHART BUILDER ---
    with tab2:
        st.subheader("⚡ Quick Visualizer")
        st.markdown("Instantly generate common plots for specific features.")
        
        q_col1, q_col2 = st.columns(2)
        
        # Quick Num
        with q_col1:
            st.markdown("#### 1️⃣ Numerical Analysis")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                q_num = st.selectbox("Select Feature", num_cols, key="q_num_feat")
                q_type_num = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Line Chart"], key="q_num_type")
                
                if q_type_num == "Histogram":
                    fig = px.histogram(df, x=q_num, marginal="box", title=f"Distribution of {q_num}")
                    st.plotly_chart(fig, width="stretch")
                elif q_type_num == "Box Plot":
                    fig = px.box(df, y=q_num, title=f"Box Plot of {q_num}")
                    st.plotly_chart(fig, width="stretch")
                elif q_type_num == "Line Chart":
                    fig = px.line(df, y=q_num, title=f"Trend of {q_num}")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No numerical features.")

        # Quick Cat
        with q_col2:
            st.markdown("#### 2️⃣ Categorical Analysis")
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            if cat_cols:
                q_cat = st.selectbox("Select Feature", cat_cols, key="q_cat_feat")
                q_type_cat = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart"], key="q_cat_type")
                
                if q_type_cat == "Bar Chart":
                    val_counts = df[q_cat].value_counts().reset_index()
                    val_counts.columns = [q_cat, 'Count']
                    fig = px.bar(val_counts, x=q_cat, y='Count', color=q_cat, title=f"Count of {q_cat}")
                    st.plotly_chart(fig, width="stretch")
                elif q_type_cat == "Pie Chart":
                    fig = px.pie(df, names=q_cat, title=f"Proportions of {q_cat}")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No categorical features.")
        
        st.divider()
        
        st.subheader("🎨 Custom Chart Builder")
        st.markdown("Build your own visualizations by selecting variables.")
        
        c_sel1, c_sel2, c_sel3, c_sel4 = st.columns(4)
        
        with c_sel1:
            chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter Plot", "Line Chart", "Bar Chart", "Frequency Plot"])
        with c_sel2:
            x_val = st.selectbox("X-Axis", df.columns)
        with c_sel3:
            # Y-axis is optional for Histogram and Frequency Plot
            if chart_type in ["Histogram", "Box Plot", "Frequency Plot"]:
                y_val = st.selectbox("Y-Axis (Optional)", [None] + df.columns.tolist())
            else:
                y_val = st.selectbox("Y-Axis", df.columns, index=min(1, len(df.columns)-1))
        with c_sel4:
            hue_opt = st.selectbox("Color / Group By", [None] + df.columns.tolist())
        
        # Logic Engine
        st.divider()
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_val, y=y_val, color=hue_opt, marginal="box", barmode="overlay", title=f"Histogram of {x_val}")
        
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_val, y=y_val, color=hue_opt, title=f"Box Plot of {x_val}")
            
        elif chart_type == "Scatter Plot":
            if y_val:
                try:
                    fig = px.scatter(df, x=x_val, y=y_val, color=hue_opt, title=f"{x_val} vs {y_val} Scatter")
                    if pd.api.types.is_numeric_dtype(df[x_val]) and pd.api.types.is_numeric_dtype(df[y_val]):
                        fig = px.scatter(df, x=x_val, y=y_val, color=hue_opt, trendline="ols", title=f"{x_val} vs {y_val} (with Trend)")
                except:
                        fig = px.scatter(df, x=x_val, y=y_val, color=hue_opt, title=f"{x_val} vs {y_val}")
            else:
                st.error("Scatter Plot requires Y-Axis.")
                fig = None
        
        elif chart_type == "Line Chart":
            if y_val:
                    fig = px.line(df, x=x_val, y=y_val, color=hue_opt, title=f"{x_val} vs {y_val} Line")
            else:
                st.error("Line Chart requires Y-Axis.")
                fig = None
        
        elif chart_type == "Bar Chart":
            if y_val:
                agg_method = st.radio("Aggregation", ["Sum", "Mean", "Count"], horizontal=True)
                if agg_method == "Count":
                    df_agg = df.groupby([x_val, hue_opt] if hue_opt else [x_val]).size().reset_index(name='Count')
                    fig = px.bar(df_agg, x=x_val, y='Count', color=hue_opt, title=f"Count of {x_val}")
                else:
                    if pd.api.types.is_numeric_dtype(df[y_val]):
                            if agg_method == "Sum":
                                df_agg = df.groupby([x_val, hue_opt] if hue_opt else [x_val])[y_val].sum().reset_index()
                            else:
                                df_agg = df.groupby([x_val, hue_opt] if hue_opt else [x_val])[y_val].mean().reset_index()
                            fig = px.bar(df_agg, x=x_val, y=y_val, color=hue_opt, title=f"{agg_method} of {y_val} by {x_val}")
                    else:
                            st.warning("Y-Axis must be numeric for Sum/Mean.")
                            fig = None
            else:
                st.error("Bar Chart requires Y-Axis (or just use Count agg).")
                fig = None
        
        elif chart_type == "Frequency Plot":
                if hue_opt and hue_opt != x_val:
                    # Count occurrences of X grouped by Hue
                    val_counts = df.groupby([x_val, hue_opt]).size().reset_index(name='Frequency')
                    fig = px.bar(val_counts, x=x_val, y='Frequency', color=hue_opt, title=f"Frequency Dist. of {x_val} by {hue_opt}")
                else:
                    # Simple Count
                    val_counts = df[x_val].value_counts().reset_index()
                    val_counts.columns = [x_val, 'Frequency']
                    fig = px.bar(val_counts, x=x_val, y='Frequency', color=x_val, title=f"Frequency Dist. of {x_val}")
        
        if fig:
            st.plotly_chart(fig, width="stretch")

    # --- TAB 3: MULTIVARIATE (Legacy Bivariate/Multivariate) ---
    with tab3:
        st.subheader("Comparison Analysis")
        # Reuse Bivariate logic if user wants fast preset views
        c1, c2, c3 = st.columns(3)
        with c1: x_bi = st.selectbox("X Variable", df.columns, key="bi_x")
        with c2: y_bi = st.selectbox("Y Variable", df.columns, index=min(1, len(df.columns)-1), key="bi_y")
        with c3: col_bi = st.selectbox("Color", [None] + df.columns.tolist(), key="bi_col")
        
        fig_bi = px.scatter(df, x=x_bi, y=y_bi, color=col_bi, title=f"Comparison: {x_bi} vs {y_bi}")
        st.plotly_chart(fig_bi, width="stretch")

    # --- TAB 4: AI INSIGHTS ---
    with tab4:
        st.subheader("🤖 Automated Data Story")
        if not numeric_cols.empty:
            corr = df[numeric_cols].corr()
            st.write(generate_story(df, corr))
            st.info("The AI analyzes shape, missingness, and correlation to generate this summary.")
        else:
                st.write(generate_story(df, pd.DataFrame()))

    # --- TAB 5: ADVANCED VISUALIZATIONS ---
    with tab5:
        st.subheader("3D & N-Dimensional Analysis")
        
        adv_plot_type = st.radio("Select Advanced Visualization", ["3D Scatter Plot", "Pair Plot (Scatter Matrix)", "Parallel Coordinates"], horizontal=True)
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(num_cols) < 3:
            st.warning("Need at least 3 numeric columns for advanced visualizations.")
        else:
            if adv_plot_type == "3D Scatter Plot":
                col1, col2, col3, col4 = st.columns(4)
                with col1: x_3d = st.selectbox("X Axis", num_cols, index=0)
                with col2: y_3d = st.selectbox("Y Axis", num_cols, index=1 if len(num_cols)>1 else 0)
                with col3: z_3d = st.selectbox("Z Axis", num_cols, index=2 if len(num_cols)>2 else 0)
                with col4: color_3d = st.selectbox("Color By", [None] + df.columns.tolist())
                
                fig_3d = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d, title=f"3D: {x_3d} x {y_3d} x {z_3d}")
                st.plotly_chart(fig_3d, width="stretch")
                
            elif adv_plot_type == "Pair Plot (Scatter Matrix)":
                selected_dims = st.multiselect("Select Dimensions", num_cols, default=num_cols[:4])
                color_pair = st.selectbox("Color ID", [None] + df.columns.tolist(), key="pair_color")
                
                if len(selected_dims) > 1:
                    fig_pair = px.scatter_matrix(df, dimensions=selected_dims, color=color_pair, title="Scatter Matrix")
                    fig_pair.update_layout(height=800)
                    st.plotly_chart(fig_pair, width="stretch")
                    
            elif adv_plot_type == "Parallel Coordinates":
                selected_para = st.multiselect("Select Dimensions", num_cols, default=num_cols[:5], key="para_dims")
                color_para = st.selectbox("Color Metric (Numeric)", [None] + num_cols, key="para_color")
                
                if len(selected_para) > 1:
                    if color_para:
                        fig_para = px.parallel_coordinates(df, dimensions=selected_para, color=color_para, title="Parallel Coordinates Plot")
                    else:
                        fig_para = px.parallel_coordinates(df, dimensions=selected_para, title="Parallel Coordinates Plot")
                    st.plotly_chart(fig_para, width="stretch")

                    st.plotly_chart(fig_para, width="stretch")

    # --- TAB 6: DATA QUALITY ---
    with tab6:
        st.subheader("🕵️ Data Quality Inspection")
        
        # 1. Duplicates
        dup_count = df.duplicated().sum()
        col_q1, col_q2 = st.columns(2)
        col_q1.metric("Duplicate Rows", dup_count, delta="⚠️" if dup_count > 0 else "Good", delta_color="inverse")
        
        # 2. Negative Values
        num_cols_dq = df.select_dtypes(include=np.number).columns
        neg_dict = {}
        for c in num_cols_dq:
            neg_count = (df[c] < 0).sum()
            if neg_count > 0:
                neg_dict[c] = neg_count
        
        if neg_dict:
            col_q2.warning(f"Negative values detected in: {list(neg_dict.keys())}")
        else:
            col_q2.success("No negative values in numeric columns.")

        st.divider()
        
        # 3. Primary Key Candidates
        st.markdown("#### 🔑 Primary Key Candidates")
        pk_candidates = []
        for c in df.columns:
            if df[c].is_unique:
                pk_candidates.append(c)
        
        if pk_candidates:
            st.success(f"Potential Primary Keys (Unique Columns): {pk_candidates}")
        else:
            st.warning("No single column is unique (No Simple PK found).")

        st.divider()

        # 4. Outliers (Z-score check)
        st.markdown("#### 📉 Potential Outliers (Z-Score > 3)")
        st.caption("Outliers are values that deviate significantly from the mean (more than 3 standard deviations).")
        
        outlier_summary = []
        outlier_details = {} # Store details for drill-down
        
        if len(num_cols_dq) > 0:
            for col in num_cols_dq:
                # Simple Z-score: (x - mean) / std
                col_mean = df[col].mean()
                col_std = df[col].std()
                
                if col_std > 0:
                    lower_bound = col_mean - (3 * col_std)
                    upper_bound = col_mean + (3 * col_std)
                    
                    # Filter outliers
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        min_out = outliers[col].min()
                        max_out = outliers[col].max()
                        
                        outlier_summary.append({
                            "Feature": col, 
                            "Outliers": outlier_count, 
                            "Percentage": f"{(outlier_count/len(df))*100:.2f}%",
                            "Lower Bound": f"{lower_bound:.2f}",
                            "Upper Bound": f"{upper_bound:.2f}",
                            "Min Outlier": f"{min_out:.2f}",
                            "Max Outlier": f"{max_out:.2f}"
                        })
                        outlier_details[col] = outliers
        
        if outlier_summary:
            st.dataframe(pd.DataFrame(outlier_summary), width="stretch")
            
            # Drill Down
            st.divider()
            st.subheader("🔍 Inspect Outlier Rows")
            drill_col = st.selectbox("Select Feature to Inspect", [d['Feature'] for d in outlier_summary])
            
            if drill_col:
                st.warning(f"Showing {len(outlier_details[drill_col])} rows where **{drill_col}** is outside normal range.")
                st.dataframe(outlier_details[drill_col], width="stretch")
        else:
            st.info("No numeric outliers detected (Z-Score > 3). values are within 3 standard deviations.")

    st.divider()
    col_next = st.columns([3, 1])[1]
    
    def go_next():
        st.session_state.nav_pipeline = "4. Model Builder"
        
    st.button("Proceed to Model Builder ➡️", type="primary", on_click=go_next)
