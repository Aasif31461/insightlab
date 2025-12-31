import streamlit as st
import pandas as pd
import numpy as np
import pickle

def render_model_runner():
    st.title("🏃 Model Runner")
    st.markdown("Upload a pre-trained `.pkl` model file to use for predictions independently.")
    
    # Initialize state for this specific module if needed
    if 'runner_model' not in st.session_state: st.session_state.runner_model = None
    
    uploaded_model = st.file_uploader("Upload .pkl file", type=["pkl"])
    
    # Load Logic
    if uploaded_model:
        try:
            # We load the model directly without pickling purely for safety if it was a real app, 
            # but for this context pickle.load is fine.
            loaded_model = pickle.load(uploaded_model)
            st.session_state.runner_model = loaded_model
            st.success(f"Successfully loaded model: {type(loaded_model).__name__}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.session_state.runner_model = None
            
    if st.session_state.runner_model is not None:
        st.divider()
        st.subheader("🔮 Make Predictions")
        
        # --- Smart Bundle Logic ---
        is_bundle = isinstance(st.session_state.runner_model, dict) and st.session_state.runner_model.get("is_insightlab_pkg")
        
        if is_bundle:
            bundle = st.session_state.runner_model
            model = bundle["model"]
            feature_names = bundle["input_features"] # Raw features (e.g. 'MAKE')
            feature_dtypes = bundle.get("input_dtypes", {})
            training_columns = bundle.get("training_columns", []) # Encoded columns (e.g. 'MAKE_AUDI')
            
            st.success("✅ InsightLab Smart Model Detected! Generating simplified input form...")
        else:
            # Legacy/Standard Pickle Logic
            model = st.session_state.runner_model
            feature_names = None
            training_columns = None
            
            # 1. Try to get feature names from the model
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
            elif hasattr(model, "get_booster"): # XGBoost
                try:
                    feature_names = model.get_booster().feature_names
                except: pass
                
            # 2. If no names, try getting feature count
            n_features = 0
            if feature_names is not None:
                n_features = len(feature_names)
            elif hasattr(model, "n_features_in_"):
                n_features = model.n_features_in_
            elif hasattr(model, "n_features"):
                n_features = model.n_features
                
            if n_features == 0 and feature_names is None:
                 st.warning("Could not automatically detect the number of features.")
                 n_features = st.number_input("How many features does this model expect?", min_value=1, value=1)
                 feature_names = [f"Feature_{i+1}" for i in range(n_features)]
                 
            if feature_names is None:
                feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
        st.info(f"Input Features: **{len(feature_names)}**")
        
        # --- Input Form ---
        with st.form("runner_form"):
            col_list = st.columns(3)
            input_data = {}
            
            for i, col in enumerate(feature_names):
                with col_list[i % 3]:
                    # Smart Type Handling
                    dtype_str = str(feature_dtypes.get(col, "")) if is_bundle else ""
                    
                    if "int" in dtype_str or "float" in dtype_str:
                         input_data[col] = st.number_input(f"{col}", value=0.0)
                    else:
                        # For categorical, we don't have the original unique values list in the bundle (yet)
                        # unless we add it to the bundle. For now, text input or generic is safer than crashing.
                        # Ideally, we should add unique values to metadata. But let's stick to text input for flexibility
                        # or simple type checking.
                        input_data[col] = st.text_input(f"{col}", value="0")
            
            submit_run = st.form_submit_button("🚀 Run Prediction")
            
        if submit_run:
            try:
                # 1. Create Initial DataFrame
                processed_data = {}
                for k, v in input_data.items():
                    # Attempt conversion for non-bundle or if user entered number in text field
                    try:
                        processed_data[k] = float(v)
                    except:
                        processed_data[k] = v 
                
                input_df = pd.DataFrame([processed_data])
                
                # 2. Handle Encoding (Smart Bundle Only)
                if is_bundle and training_columns:
                    # One Hot Encode
                    input_df = pd.get_dummies(input_df)
                    
                    # Align with Training Columns (add missing 0s)
                    missing_cols = set(training_columns) - set(input_df.columns)
                    if missing_cols:
                        missing_df = pd.DataFrame(0, index=input_df.index, columns=list(missing_cols))
                        input_df = pd.concat([input_df, missing_df], axis=1)
                    
                    # Reorder/Filter
                    input_df = input_df[training_columns]
                else:
                     # Standard Alignment
                     if feature_names is not None:
                         # Ensure columns present
                         for col in feature_names:
                             if col not in input_df.columns:
                                 input_df[col] = 0
                         input_df = input_df[feature_names]

                # 3. Predict
                pred = model.predict(input_df)[0]
                
                target_label = "Predicted Value"
                if is_bundle and "target_col" in bundle and bundle["target_col"]:
                    target_label = f"Predicted {bundle['target_col']}"
                
                st.markdown("### 🎯 Result")
                if isinstance(pred, (float, np.float64)):
                     st.metric(target_label, f"{pred:,.4f}")
                else:
                     st.metric(target_label, str(pred))
                     
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                if not is_bundle:
                    st.caption("Ensure input values match the data types expected by the model.")
