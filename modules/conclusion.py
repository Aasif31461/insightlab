import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import io

def render_conclusion():
    st.title("🏁 Conclusion & Next Steps")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model in **Model Builder** first.")
        st.stop()
    
    st.success("Model is ready for deployment!")
    
    # --- TABBED VIEW FOR CONCLUSION ---
    tab1, tab2 = st.tabs(["🎮 Model Playground", "📦 Export Model"])

    with tab1:
        st.subheader("Interactive Prediction")
        st.markdown("Test your model with new data points.")
        
        # Form for inputs
        with st.form("prediction_form"):
            col_list = st.columns(3)
            input_data = {}
            
            # We need to recreate inputs based on original features (before OneHotEncoding)
            # But we only have X_train (which is encoded). 
            # Strategy: Use clean_df columns (minus target) as input references.
            
            if 'target_col' in st.session_state:
                feature_cols = [c for c in st.session_state.clean_df.columns if c != st.session_state.target_col]
            else:
                feature_cols = st.session_state.X_train.columns.tolist() # Fallback (might be raw encoded)

            for i, col in enumerate(feature_cols):
                with col_list[i % 3]:
                    if pd.api.types.is_numeric_dtype(st.session_state.clean_df[col]):
                        input_data[col] = st.number_input(f"{col}", value=float(st.session_state.clean_df[col].mean()))
                    else:
                        input_data[col] = st.selectbox(f"{col}", st.session_state.clean_df[col].unique())
                        
            submit = st.form_submit_button("🔮 Predict")
            
        if submit:
            # Preprocessing Input to match X_train structure
            input_df = pd.DataFrame([input_data])
            
            # Apply One-Hot Encoding (same logic as training)
            input_df = pd.get_dummies(input_df)
            
            # Align with training columns (fill missing with 0)
            # This ensures all One-Hot columns exist
            # Align with training columns (fill missing with 0)
            # This ensures all One-Hot columns exist
            missing_cols = set(st.session_state.X_train.columns) - set(input_df.columns)
            if missing_cols:
                 missing_df = pd.DataFrame(0, index=input_df.index, columns=list(missing_cols))
                 input_df = pd.concat([input_df, missing_df], axis=1)
            
            # Reorder and filter to match exact X_train lines
            input_df = input_df[st.session_state.X_train.columns]
            
            # Prediction
            prediction = st.session_state.model.predict(input_df)[0]
            
            st.markdown("### 🎯 Prediction Result")
            label = st.session_state.target_col if 'target_col' in st.session_state else "Predicted Value"
            if isinstance(prediction, (float, np.float64)):
                st.metric(f"Predicted {label}", f"{prediction:,.4f}")
            else:
                st.metric(f"Predicted {label}", f"{prediction}")

    with tab2:
        st.subheader("📥 Download Center")
        st.markdown("Access all your project assets here.")
        
        d_c1, d_c2, d_c3 = st.columns(3)
        
        # 1. Model
        with d_c1:
            st.markdown("#### 🧠 Model")
            
            # Smart Export Bundle
            target_col_excl = st.session_state.target_col if 'target_col' in st.session_state else None
            input_feats = [c for c in st.session_state.clean_df.columns if c != target_col_excl]
            
            model_bundle = {
                "model": st.session_state.model,
                "is_insightlab_pkg": True,
                "input_features": input_feats,
                "input_dtypes": st.session_state.clean_df[input_feats].dtypes.astype(str).to_dict(),
                "training_columns": st.session_state.X_train.columns.tolist() if 'X_train' in st.session_state else [],
                "target_col": target_col_excl
            }
            
            model_pkl = pickle.dumps(model_bundle)
            
            st.download_button(
                label="Download Smart Model .pkl",
                data=model_pkl,
                file_name="insightlab_smart_model.pkl",
                mime="application/octet-stream",
                help="Includes model + metadata for easy internal reuse."
            )

        # 2. Cleaned Data
        with d_c2:
            st.markdown("#### 🧹 Cleaned Data")
            if st.session_state.clean_df is not None:
                csv = st.session_state.clean_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download .csv",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    help="Your preprocessed dataset."
                )
            else:
                st.warning("No clean data.")

        # 3. Report
        with d_c3:
            st.markdown("#### 📄 EDA Report")
            if st.button("Generate & Download"):
                buffer = io.StringIO()
                st.session_state.clean_df.info(buf=buffer)
                info_str = buffer.getvalue()
                
                # Simple HTML Report
                html = f"""
                <html>
                <head><style>body{{font-family: sans-serif; padding:20px;}} table{{border-collapse: collapse; width: 100%;}} th, td {{border: 1px solid #ddd; padding: 8px;}} tr:nth-child(even){{background-color: #f2f2f2;}} th {{padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #04AA6D; color: white;}}</style></head>
                <body>
                <h1>📊 InsightLab AI - EDA Report</h1>
                <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>1. Dataset Info</h2>
                <pre>{info_str}</pre>
                
                <h2>2. Statistical Summary</h2>
                {st.session_state.clean_df.describe().to_html()}
                
                <h2>3. Data Sample (First 10 Rows)</h2>
                {st.session_state.clean_df.head(10).to_html()}
                </body>
                </html>
                """
                st.download_button(
                    label="Download Report .html",
                    data=html.encode('utf-8'),
                    file_name="eda_report.html",
                    mime="text/html"
                )
        
        st.divider()
        st.info("💡 **Tip**: Use `pickle.load(open('datasense_model.pkl', 'rb'))` to load your model in Python.")
