# 🚀 AutoML Studio

AutoML Studio is a state-of-the-art, **Streamlit Cloud-ready** machine learning platform designed to fully automate the data science lifecycle. From data cleaning and exploratory data analysis (EDA) to model training, evaluation, and SHAP-based explainability, AutoML Studio acts as an AI data scientist right in your browser.

## ✨ Features

- **Automated Data Cleaning:** Handles missing values, removes duplicates, and mitigates outliers.
- **Exploratory Data Analysis (EDA):** Automatically generates interactive correlation heatmaps and numerical distributions.
- **Intelligent Preprocessing:** Applies One-Hot Encoding and Standard Scaling efficiently.
- **Multi-Model Training:** Trains and tunes models across both Classification and Regression tasks (XGBoost, LightGBM, CatBoost, Random Forest, etc.).
- **SHAP Explainability:** Generates Global and Local explainability plots to demystify "black-box" models.
- **Real-Time Inference:** Perform real-time predictions with deployed `.joblib` model artifacts directly from the UI.
- **Experiment Tracking:** Logs MLflow metrics and saves historical runs in a local SQLite database.

## 🚀 Quick Start (Local & Streamlit Cloud)

This project is fully containerized as a monolithic Python application and is optimized to run effortlessly on **Streamlit Cloud**.

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/mrunmayee3108/AutoML-Studio.git
cd AutoML-Studio
pip install -r requirements.txt
```

### 2. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run streamlit_app/app.py
```
The application will be accessible at `http://localhost:8501`.

## 📂 Architecture

- `src/`: Core backend logic (EDA, Feature Engineering, Model Training, Persistence).
- `streamlit_app/`: Streamlit UI pages (`app.py`, `History`, `Predict`, `Upload and Train`).
- `database/`: Stores the SQLite DB for tracking experiment history.
- `saved_models/`: Stores versioned `.joblib` model pipelines (includes preprocessors).
- `mlruns/`: Stores MLflow tracking metadata.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

## 📜 License
This project is licensed under the MIT License.
