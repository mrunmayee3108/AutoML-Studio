import streamlit as st
import requests
import json

import sys
import os

# Add root directory to sys.path so we can import from src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.persistence.manager import ModelPersistenceManager
from src.prediction.predictor import ModelPredictor

st.set_page_config(page_title="Real-time Prediction", page_icon="🔮", layout="wide")

st.title("Real-time Prediction Engine")
st.markdown("Select a deployed model and input feature values to generate predictions.")

persister = ModelPersistenceManager(base_directory=os.path.join(root_dir, "saved_models"))

def fetch_models():
    try:
        return persister.list_saved_models()
    except:
        return []

available_models = fetch_models()

if not available_models:
    st.error("No deployed models found. Please train and save a model first.")
else:
    model_options = {m["version_id"]: f"{m['model_name']} ({m['version_id']})" for m in available_models}
    selected_version = st.selectbox("Select Model Version:", options=list(model_options.keys()), format_func=lambda x: model_options[x])
    
    st.markdown("### Input Features")
    st.info("Enter the feature values exactly as they appeared in the training dataset (JSON format).")
    
    default_payload = {
        "Age": 0.5,
        "Income": -1.2,
        "Credit_Score": 0.8,
        "Loan_Amount": -0.3
    }
    
    user_input = st.text_area("Feature Payload (JSON):", value=json.dumps(default_payload, indent=4), height=200)

    if st.button("Generate Prediction", type="primary"):
        try:
            features_dict = json.loads(user_input)
            
            with st.spinner("Loading Model & Predicting..."):
                try:
                    loaded_model, loaded_prep, metadata = persister.load_pipeline(selected_version)
                    predictor = ModelPredictor(
                        model=loaded_model,
                        task_type=metadata["task_type"],
                        expected_features=metadata["expected_features"],
                        class_names=metadata.get("class_names"),
                        preprocessor=loaded_prep
                    )
                    
                    results = predictor.predict(features_dict)
                    result_data = results[0]
                    
                    st.success("Inference Completed Successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prediction", result_data.get("prediction"))
                    col2.metric("Confidence", result_data.get("confidence_score"))
                    col3.metric("Task Type", result_data.get("task_type"))
                    
                    st.json(result_data)
                except ValueError as ve:
                    st.error(f"Input Error: {str(ve)}")
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")
                    
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your syntax.")