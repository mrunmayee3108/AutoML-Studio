import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Experiment History", page_icon="📊", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.title("Experiment History")
st.markdown("View all tracked AutoML training runs, model metadata, and artifacts.")

@st.cache_data(ttl=10) 
def fetch_experiments():
    try:
        response = requests.get(f"{API_URL}/experiments", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to connect to backend API: {str(e)}")
        return []

experiments = fetch_experiments()

if not experiments:
    st.warning("No experiments found in the database or backend is unreachable.")
else:
    df = pd.DataFrame(experiments)
    
    df = df[["id", "created_at", "project_name", "task_type", "dataset_name", "best_model", "metric", "version_id"]]
    df.columns = ["ID", "Date", "Project", "Task", "Dataset", "Winning Model", "Primary Metric", "Version ID"]
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    st.success(f"Total tracked experiments: {len(df)}")