import streamlit as st
import requests
import pandas as pd

import sys
import os

# Add root directory to sys.path so we can import from src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.database.manager import DatabaseManager

st.set_page_config(page_title="Experiment History", page_icon="📊", layout="wide")

st.title("Experiment History")
st.markdown("View all tracked AutoML training runs, model metadata, and artifacts.")

def fetch_experiments():
    try:
        db_path = f"sqlite:///{os.path.join(root_dir, 'database', 'automl.db')}"
        db = DatabaseManager(db_path=db_path)
        return db.get_all_experiments()
    except Exception as e:
        st.error(f"Failed to fetch from Database: {str(e)}")
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