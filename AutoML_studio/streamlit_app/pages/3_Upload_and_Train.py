import streamlit as st
import pandas as pd
import io

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
            
            status_text.text("Phase 3: Cleaning Data...")
            progress_bar.progress(20)
            status_text.text("Phase 5: Engineering Features...")
            progress_bar.progress(40)
            status_text.text("Phase 7 & 8: Training & Evaluating Models...")
            progress_bar.progress(60)
            status_text.text("Phase 15 & 16: Saving Models & Generating Reports...")
            progress_bar.progress(90)
            status_text.text("Pipeline Complete!")
            progress_bar.progress(100)
            st.success("Training Complete! Go to the 'History' tab to view results and download the PDF report.")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")