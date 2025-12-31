import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, explained_variance_score, max_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

def render_ml_studio():
    if st.session_state.clean_df is None:
        st.warning("Please preprocess data first.")
        st.stop()
        
    st.title("🧠 Model Builder")
    df = st.session_state.clean_df
    
    # Initialize History
    if 'model_history' not in st.session_state: st.session_state.model_history = []
    
    # Layout: Setup | Results
    setup_col, results_col = st.columns([1, 2])
    
    with setup_col:
        st.subheader("Configuration")
        target = st.selectbox("Select Target Variable", df.columns)
        
        # Problem Type Detection
        if pd.api.types.is_numeric_dtype(df[target]) and len(df[target].unique()) > 20:
            problem_type = "Regression"
        else:
            problem_type = "Classification"
        
        st.badge(f"Detected: {problem_type}")
        
        # Model Zoo
        if problem_type == "Regression":
            model_name = st.selectbox("Select Algorithm", 
                ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR"],
                help="Choose the brain of your model:\n- Linear Regression: Simple line fitting.\n- Random Forest/GB: Powerful ensembles.")
        else:
            model_name = st.selectbox("Select Algorithm", 
                ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVC"],
                help="Choose the brain of your model.")
        
        split_pct = st.slider("Training Split %", 50, 90, 80, help="How much data to learn from? 80% is standard.")
        
        # --- Hyperparameters ---
        st.markdown("---")
        st.write("**Hyperparameters**")
        
        params = {}
        if model_name in ["Ridge", "Lasso"]:
            params['alpha'] = st.slider("Regularization (Alpha)", 0.01, 10.0, 1.0, help="Controls model complexity. Higher = simpler model/less overfitting.")
        
        if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            params['max_depth'] = st.slider("Max Depth", 1, 20, 5, help="How deep the tree can grow. Deeper = catches more details but might overfit.")
            
        if model_name in ["Random Forest", "Gradient Boosting"]:
            params['n_estimators'] = st.slider("Number of Trees", 10, 200, 100, help="More trees = usually better but slower.")
            
        if model_name in ["Gradient Boosting"]:
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, help="Step size for learning. Smaller = slower but potentially more accurate.")
        
        # --- Cross-Validation ---
        st.markdown("---")
        use_cv = st.checkbox("Enable Cross-Validation (5-Folds)", help="Splits data into 5 parts and trains 5 times to check stability. Slower but more reliable.")
        
        train_btn = st.button("🚀 Train Model")
        
        # Reset History
        if st.session_state.model_history:
            if st.button("Clear History"):
                    st.session_state.model_history = []
                    st.rerun()

    with results_col:
        if train_btn:
            with st.spinner("Training & Evaluating..."):
                # 1. Prepare Data
                df_train = df.dropna()
                if len(df_train) < len(df):
                    st.warning(f"⚠️ Dropped {len(df) - len(df_train)} rows containing missing values.")
                
                X = df_train.drop(columns=[target])
                y = df_train[target]
                
                # Robust cleaning for Infinite values (causes RuntimeWarnings)
                X = X.replace([np.inf, -np.inf], np.nan)
                if X.isna().any().any():
                     # Re-align y with dropped X rows
                     valid_idx = X.dropna().index
                     X = X.dropna()
                     y = y.loc[valid_idx]
                     st.warning(f"⚠️ Dropped additional rows containing Infinite values. Final count: {len(X)}")
                
                # Scaling (Fixed: Apply to all numeric X for stability)
                
                numeric_features = X.select_dtypes(include=np.number).columns
                scaler = None
                if not numeric_features.empty:
                    scaler = StandardScaler()
                    X[numeric_features] = scaler.fit_transform(X[numeric_features])
                
                # Cardinally check for Classification
                if problem_type == "Classification" and y.nunique() < 2:
                    st.error(f"🚨 Target variable '{target}' has only {y.nunique()} unique class. Classification requires at least 2 classes.")
                    st.stop()
                
                # Encoder
                X = pd.get_dummies(X, drop_first=True)
                if problem_type == "Classification" and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # Stratify for Classification to avoid "one class" in split
                stratify_col = y if problem_type == "Classification" else None
                
                # Handle single class in split even if overall > 1 (extreme imbalance)
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_pct/100, random_state=42, stratify=stratify_col)
                except ValueError:
                        # Fallback if stratify fails (e.g. class has 1 member)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_pct/100, random_state=42)
                
                
                
                # 2. Select Model & Params
                base_model = None
                if problem_type == "Regression":
                    if model_name == "Linear Regression": base_model = LinearRegression()
                    elif model_name == "Ridge": base_model = Ridge(alpha=params.get('alpha', 1.0))
                    elif model_name == "Lasso": base_model = Lasso(alpha=params.get('alpha', 1.0))
                    elif model_name == "Decision Tree": base_model = DecisionTreeRegressor(max_depth=params.get('max_depth', None))
                    elif model_name == "Random Forest": base_model = RandomForestRegressor(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None))
                    elif model_name == "Gradient Boosting": base_model = GradientBoostingRegressor(n_estimators=params.get('n_estimators', 100), learning_rate=params.get('learning_rate', 0.1), max_depth=params.get('max_depth', 3))
                    elif model_name == "SVR": base_model = SVR()
                    
                    # Wrap
                    model = TransformedTargetRegressor(regressor=base_model, transformer=StandardScaler())
                else:
                    if model_name == "Logistic Regression": model = LogisticRegression(max_iter=1000)
                    elif model_name == "Decision Tree": model = DecisionTreeClassifier(max_depth=params.get('max_depth', None))
                    elif model_name == "Random Forest": model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None))
                    elif model_name == "Gradient Boosting": model = GradientBoostingClassifier(n_estimators=params.get('n_estimators', 100), learning_rate=params.get('learning_rate', 0.1), max_depth=params.get('max_depth', 3))
                    elif model_name == "SVC": model = SVC()

                # 3. Train
                model.fit(X_train, y_train)
                
                # 4. Cross-Validation (Optional)
                cv_score_mean = "-"
                cv_score_std = "-"
                if use_cv:
                    # Note: We use original X, y (scaled/encoded) for CV to use full data or just X_train?
                    # Best practice: CV on X_train to avoid leakage.
                    scores = cross_val_score(model, X_train, y_train, cv=5)
                    cv_score_mean = f"{scores.mean():.4f}"
                    cv_score_std = f"{scores.std():.4f}"

                # Store Model for UI
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.target_col = target
                st.session_state.problem_type = problem_type

                # Compute Test Metrics for History
                y_pred_test = model.predict(X_test)
                
                history_entry = {
                    "Model": model_name,
                    "Split%": f"{split_pct}%",
                    "CV Score (Mean)": cv_score_mean,
                }
                
                if problem_type == "Regression":
                    r2 = r2_score(y_test, y_pred_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    mae = mean_absolute_error(y_test, y_pred_test)
                    history_entry.update({"R2": f"{r2:.4f}", "RMSE": f"{rmse:.4f}", "MAE": f"{mae:.4f}"})
                else:
                    acc = accuracy_score(y_test, y_pred_test)
                    history_entry.update({"Accuracy": f"{acc:.4f}"})
                
                st.session_state.model_history.append(history_entry)

        # --- Results Display ---
        if st.session_state.model is not None and 'target_col' in st.session_state and st.session_state.target_col == target:
            st.subheader("Results Analysis")
            
            # Eval Toggle
            eval_set = st.radio("Evaluate on:", ["Test Data", "Train Data"], horizontal=True)
            if eval_set == "Test Data":
                X_eval, y_eval = st.session_state.X_test, st.session_state.y_test
            else:
                X_eval, y_eval = st.session_state.X_train, st.session_state.y_train
            
            y_pred = st.session_state.model.predict(X_eval)
            
            if st.session_state.problem_type == "Regression":
                mae = mean_absolute_error(y_eval, y_pred)
                rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
                r2 = r2_score(y_eval, y_pred)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE", f"{mae:.4f}")
                c2.metric("RMSE", f"{rmse:.4f}")
                c3.metric("R2 Score", f"{r2:.4f}")
                
                # Dynamic Explanation
                st.info(f"💡 **In Simple Words:**\n- **RMSE/MAE**: On average, the model's predictions are off by roughly **{mae:.2f} to {rmse:.2f}** units.\n- **R2 Score**: The model explains about **{r2*100:.1f}%** of the variation in {target}.")
                
                fig = px.scatter(x=y_eval, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                fig.add_shape(type="line", line=dict(dash='dash', color='orange'), x0=y_eval.min(), y0=y_eval.min(), x1=y_eval.max(), y1=y_eval.max())
                st.plotly_chart(fig, width="stretch")
                
            else: # Classification
                acc = accuracy_score(y_eval, y_pred)
                st.metric("Accuracy", f"{acc:.2%}")
                st.info(f"💡 **In Simple Words:**\n- The model correctly predicts the category **{acc*100:.1f}%** of the time.")
                
                cm = confusion_matrix(y_eval, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
                st.plotly_chart(fig_cm, width="stretch")

            # Feature Importance
            inspect_model = st.session_state.model
            if hasattr(inspect_model, 'regressor_'): inspect_model = inspect_model.regressor_
            if hasattr(inspect_model, 'feature_importances_'):
                st.subheader("Feature Importance")
                imp_df = pd.DataFrame({'Feature': st.session_state.X_train.columns, 'Importance': inspect_model.feature_importances_})
                imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
                fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top Influential Factors")
                st.plotly_chart(fig_imp, width="stretch")
            
            # Explainability (SHAP)
            st.markdown("---")
            st.subheader("🔍 Model Explainability (SHAP)")
            try:
                import shap
                
                # Helper to get model/explainer
                if hasattr(st.session_state.model, 'regressor_'): 
                    model_to_explain = st.session_state.model.regressor_
                else:
                    model_to_explain = st.session_state.model

                # Select Explainer
                explainer = None
                # Tree-based
                if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
                        explainer = shap.TreeExplainer(model_to_explain)
                # Linear
                elif model_name in ["Linear Regression", "Ridge", "Lasso", "Logistic Regression"]:
                        # Linear explainer needs masker (X_train)
                        explainer = shap.LinearExplainer(model_to_explain, st.session_state.X_train)
                
                if explainer:
                    with st.spinner("Generating Explanations..."):
                        # Calculate SHAP values (subset of test data for speed)
                        # Use first 50 samples to keep it fast
                        X_shap = st.session_state.X_test.iloc[:50]
                        shap_values = explainer.shap_values(X_shap)
                        
                        # Prepare data
                        shap_df = pd.DataFrame(shap_values, columns=X_shap.columns)
                        
                        # Calculate Feature Importance (Mean Absolute SHAP)
                        feature_importance = shap_df.abs().mean().sort_values(ascending=False)
                        
                        # Tabs for different views
                        tab_summary, tab_detail, tab_dependence = st.tabs(["📊 Global Importance", "🐝 Detailed View", "📈 Feature Dependence"])
                        
                        # --- Tab 1: Global Importance (Bar Chart) ---
                        with tab_summary:
                            st.caption("Which features have the biggest average impact on predictions?")
                            fi_df = feature_importance.reset_index()
                            fi_df.columns = ['Feature', 'Mean |SHAP Value|']
                            
                            fig_bar = px.bar(
                                fi_df,
                                x='Mean |SHAP Value|',
                                y='Feature',
                                orientation='h',
                                title="<b>Feature Importance</b> (Mean Absolute SHAP)",
                                template="plotly_dark",
                                color='Mean |SHAP Value|',
                                color_continuous_scale='Blues'
                            )
                            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig_bar, width="stretch")

                        # --- Tab 2: Detailed View (Beeswarm-style) ---
                        with tab_detail:
                            st.caption("How does each feature value affect the prediction? (Red = High Value, Blue = Low Value)")
                            
                            # Prepare Layout Data
                            shap_long = shap_df.reset_index().melt(id_vars='index', var_name='Feature', value_name='SHAP Value')
                            X_long = X_shap.reset_index().melt(id_vars='index', var_name='Feature', value_name='Feature Value')
                            plot_df = pd.merge(shap_long, X_long, on=['index', 'Feature'])
                            plot_df['Feature Value'] = pd.to_numeric(plot_df['Feature Value'], errors='coerce')
                            
                            # Sort by importance
                            plot_df['Feature'] = pd.Categorical(plot_df['Feature'], categories=feature_importance.index[::-1], ordered=True)
                            plot_df = plot_df.sort_values('Feature')

                            fig_bee = px.scatter(
                                plot_df,
                                x='SHAP Value',
                                y='Feature',
                                color='Feature Value',
                                color_continuous_scale='RdBu_r', # Red=High
                                template="plotly_dark",
                                title="<b>SHAP Summary</b> (Distribution of Impacts)",
                                hover_data=['Feature Value']
                            )
                            
                            # Add some jitter manually? px.strip or just scatter with opacity
                            fig_bee.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0)))
                            fig_bee.update_layout(
                                xaxis_title="SHAP Value (Impact)",
                                yaxis_title=None,
                                coloraxis_colorbar=dict(title="Feature Val"),
                                height=400 + (len(X_shap.columns) * 20)
                            )
                            fig_bee.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.2)")
                            
                            st.plotly_chart(fig_bee, width="stretch")

                        # --- Tab 3: Dependence Plot ---
                        with tab_dependence:
                            st.caption("Explore the relationship between a feature's value and its impact.")
                            dep_col = st.selectbox("Select Feature to Inspect", X_shap.columns)
                            
                            # Data for dependence
                            dep_data = pd.DataFrame({
                                'Feature Value': X_shap[dep_col],
                                'SHAP Value': shap_values[:, X_shap.columns.get_loc(dep_col)]
                            })
                            
                            # Try to find an interaction feature (most correlated with SHAP val? or just color by itself)
                            # Simple version: Color by the feature value itself
                            
                            fig_dep = px.scatter(
                                dep_data,
                                x='Feature Value',
                                y='SHAP Value',
                                color='Feature Value',
                                color_continuous_scale='Bluered',
                                title=f"<b>Dependence Plot</b>: {dep_col}",
                                template="plotly_dark",
                                trendline="lowess" # Add trendline to see pattern
                            )
                            st.plotly_chart(fig_dep, width="stretch")

                else:
                    st.info("SHAP explainer not supported for this model type yet (e.g., SVR, SVC).")

            except ImportError:
                st.warning("⚠️ The `shap` library is not installed. To see explanations, install it: `pip install shap matplotlib`")
            except Exception as e:
                # Specific error handling for additive models or dimension mismatch
                st.error(f"Could not generate SHAP plot: {e}")
        
        # --- Model Comparison Table ---
        if st.session_state.model_history:
            st.divider()
            st.subheader("🏆 Model Comparison Leaderboard")
            hist_df = pd.DataFrame(st.session_state.model_history)
            st.dataframe(hist_df, width="stretch")

            st.divider()
            col_next = st.columns([3, 1])[1]
            
            def go_next():
                st.session_state.nav_mode = "Pipeline"
                st.session_state.nav_pipeline = "🏁 Conclusion"
                
            st.button("Proceed to Conclusion ➡️", type="primary", on_click=go_next)
