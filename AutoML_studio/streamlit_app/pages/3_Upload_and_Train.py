import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import os
import traceback

# Add root directory to sys.path so we can import from src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.preprocessing.data_cleaner import DataCleaner
from src.eda.analyzer import EDAAnalyzer
from src.feature_engineering.preprocessor import FeatureEngineer
from src.models.task_detector import TaskDetector
from src.models.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.explainability.shap_explainer import SHAPExplainer
from src.persistence.manager import ModelPersistenceManager
from src.database.manager import DatabaseManager
from src.mlflow_integration.tracker import MLflowTracker
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Train AutoML", page_icon="🚀", layout="wide")

st.title("Train New Model")
st.markdown("Upload a tabular dataset (CSV or Excel) to automatically train, tune, and evaluate machine learning models.")

st.subheader("1. Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)    
        st.success(f"Successfully loaded '{uploaded_file.name}'")
        st.subheader("2. Dataset Preview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())
        st.dataframe(df.head(15), use_container_width=True)
        st.subheader("3. Configure AutoML Task")
        target_column = st.selectbox(
            "Select the Target Column (what do you want to predict?)", 
            options=df.columns.tolist(),
            index=len(df.columns)-1 
        )
        task_type = st.radio(
            "Select Task Type",
            options=["Auto-Detect", "Classification", "Regression"],
            horizontal=True
        )
        
        project_name = st.text_input("Project Name (for reporting):", value=f"Project_{uploaded_file.name.split('.')[0]}")
        
        st.markdown("---")
        if st.button("🚀 Start AutoML Pipeline", type="primary", use_container_width=True):
            st.info("Initializing AutoML Engine... This will trigger your backend modules.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Phase 1: Cleaning Data
            status_text.text("Phase 1: Cleaning Data...")
            cleaner = DataCleaner(df)
            cleaner.drop_duplicates()
            cleaner.handle_missing_values()
            cleaner.auto_handle_outliers()
            clean_df = cleaner.get_cleaned_dataFrame()
            with st.expander("Data Cleaning Report"):
                for msg in cleaner.get_cleaning_report():
                    st.write(f"- {msg}")
            progress_bar.progress(15)

            # Phase 2: EDA
            status_text.text("Phase 2: Generating EDA...")
            analyzer = EDAAnalyzer(clean_df)
            figs = {}
            with st.expander("Exploratory Data Analysis"):
                st.write("Numerical Distributions:")
                dist_figs = analyzer.plot_numerical_distributions()
                for i, fig in enumerate(dist_figs):
                    st.pyplot(fig)
                    figs[f"dist_plot_{i}"] = fig
                st.write("Correlation Heatmap:")
                heatmap = analyzer.plot_correlation_heatmap()
                st.pyplot(heatmap)
                figs["correlation_heatmap"] = heatmap
            progress_bar.progress(30)

            # Phase 3: Task Detection
            status_text.text("Phase 3: Detecting Task...")
            actual_task_type = task_type
            if task_type == "Auto-Detect":
                detector = TaskDetector(clean_df, target_column)
                actual_task_type = detector.detect_task()
            else:
                actual_task_type = task_type.lower()
            st.write(f"**Task Type Resolved As:** {actual_task_type.capitalize()}")
            progress_bar.progress(40)

            # Phase 4: Feature Engineering
            status_text.text("Phase 4: Engineering Features...")
            engineer = FeatureEngineer(clean_df, target_column)
            engineer.apply_encoding()
            engineer.apply_scaling()
            X, y = engineer.get_processed_dataframe()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            progress_bar.progress(55)

            # Phase 5: Training
            status_text.text("Phase 5: Training Models...")
            trainer = ModelTrainer(actual_task_type)
            trained_models = trainer.train_models(X_train, y_train)
            progress_bar.progress(70)

            # Phase 6: Evaluation
            status_text.text("Phase 6: Evaluating Models...")
            evaluator = ModelEvaluator(actual_task_type)
            leaderboard = evaluator.evaluate_models(trained_models, X_test, y_test)
            st.write("### Model Leaderboard")
            st.dataframe(leaderboard, use_container_width=True)
            
            best_model_name = leaderboard.iloc[0]['Model']
            best_model = trained_models[best_model_name][0]
            if actual_task_type == 'classification':
                best_metric_name = 'F1_Score'
                best_metric_val = leaderboard.iloc[0]['F1_Score']
            else:
                best_metric_name = 'R2_Score'
                best_metric_val = leaderboard.iloc[0]['R2_Score']
            
            st.success(f"Best Model Selected: **{best_model_name}** ({best_metric_name}: {best_metric_val})")
            progress_bar.progress(80)

            # Phase 7: Explainability
            status_text.text("Phase 7: Generating Explainability (SHAP)...")
            try:
                explainer = SHAPExplainer(best_model, X_train, actual_task_type)
                with st.expander("Model Explainability (SHAP)"):
                    st.write("Global Feature Importance:")
                    global_fig = explainer.plot_global_summary()
                    st.pyplot(global_fig)
                    figs["shap_global"] = global_fig
                    st.write("Local Explanation (First Test Sample):")
                    local_fig = explainer.plot_local_waterfall(0, X_test.head(1))
                    st.pyplot(local_fig)
                    figs["shap_local"] = local_fig
            except Exception as e:
                st.warning(f"Could not generate SHAP plots for {best_model_name}: {str(e)}")
            progress_bar.progress(90)

            # Phase 8: Persistence & Logging
            status_text.text("Phase 8: Saving Model & Logging Experiment...")
            
            # Expected features
            # WE MUST SAVE THE RAW FEATURES BEFORE ONE-HOT ENCODING SO INFERENCE VALIDATION WORKS!
            expected_features = list(clean_df.drop(columns=[target_column]).columns)
            
            # Class names if classification
            class_names = None
            if actual_task_type == 'classification' and target_column in engineer.label_encoders:
                class_names = list(engineer.label_encoders[target_column].classes_)
            elif actual_task_type == 'classification':
                class_names = [str(c) for c in y.unique()]

            metrics = leaderboard.iloc[0].to_dict()
            metrics.pop('Model', None)
            
            # MLflow
            try:
                import pathlib
                mlruns_dir = os.path.join(root_dir, "mlruns")
                mlruns_uri = pathlib.Path(mlruns_dir).as_uri()
                tracker = MLflowTracker(experiment_name=project_name, tracking_uri=mlruns_uri)
                tracker.log_model_run(
                    model_name=best_model_name,
                    model_object=best_model,
                    metrics=metrics,
                    artifacts=figs,
                    dataset_name=uploaded_file.name
                )
            except Exception as e:
                st.warning(f"MLflow logging failed: {str(e)}")
            
            persister = ModelPersistenceManager(base_directory=os.path.join(root_dir, "saved_models"))
            version_id = persister.save_pipeline(
                model=best_model,
                model_name=best_model_name,
                task_type=actual_task_type,
                expected_features=expected_features,
                metrics=metrics,
                preprocessor=engineer,
                class_names=class_names
            )
            
            db_manager = DatabaseManager(db_path=f"sqlite:///{os.path.join(root_dir, 'database', 'automl.db')}")
            experiment_data = {
                "project_name": project_name,
                "task_type": actual_task_type,
                "dataset_name": uploaded_file.name,
                "best_model_name": best_model_name,
                "primary_metric_name": best_metric_name,
                "primary_metric_value": float(best_metric_val),
                "version_id": version_id,
                "report_path": "" 
            }
            db_manager.log_experiment(experiment_data)

            status_text.text("Pipeline Complete!")
            progress_bar.progress(100)
            st.success("Training Complete! Go to the 'History' tab to view results and test inference in 'predict'.")

    except Exception as e:
        st.error(f"Error during pipeline execution: {str(e)}")
        st.error(traceback.format_exc())