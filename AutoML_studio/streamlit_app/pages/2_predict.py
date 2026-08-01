import streamlit as st
import requests
import json

st.set_page_config(page_title="Real-time Prediction", page_icon="🔮", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.title("Real-time Prediction Engine")
st.markdown("Select a deployed model and input feature values to generate predictions.")

# 1. Fetch available models
@st.cache_data(ttl=30)
def fetch_models():
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
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
            
            with st.spinner("Querying Inference API..."):
                payload = {
                    "version_id": selected_version,
                    "features": features_dict
                }
                
                res = requests.post(f"{API_URL}/predict", json=payload)
                
                if res.status_code == 200:
                    result_data = res.json()[0]
                    st.success("Inference Completed Successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prediction", result_data.get("prediction"))
                    col2.metric("Confidence", result_data.get("confidence_score"))
                    col3.metric("Task Type", result_data.get("task_type"))
                    
                    st.json(result_data)
                else:
                    st.error(f"API Error: {res.json().get('detail')}")
                    
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your syntax.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI server. Ensure it is running on port 8000.")