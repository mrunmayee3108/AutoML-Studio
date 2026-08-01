import streamlit as st

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("⚡AutoML Studio")
    st.subheader("Explainable Machine Learning Platform for Automated Model Selection")
    
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the MLOps Dashboard
    This platform automatically analyzes datasets, trains multiple machine learning models, 
    tracks experiments, and serves the best models via a highly scalable REST API.

    **Navigation:**
    * 📊 **History:** View past experiments, performance metrics, and model leaderboards.
    * 🔮 **Predict:** Test deployed models in real-time using the inference engine.
    
    *Developed as a production-grade demonstration of MLOps, Explainable AI, and Software Engineering.*
    """)
    
    st.info("Backend Systems Online: FastAPI connected to SQLite & Joblib Artifact Store.")

if __name__ == "__main__":
    main()