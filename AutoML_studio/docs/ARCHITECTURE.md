# AutoML Studio – System Architecture

AutoML Studio is a modular, end-to-end AutoML platform for tabular data, designed to demonstrate production-ready Machine Learning, Explainable AI, MLOps, and web application development.

---

## High-Level Overview

At a high level, the system consists of:

- A **core ML engine** (Python package under `src/`) that handles data ingestion, preprocessing, EDA, feature engineering, task detection, model training, evaluation, explainability, experiment tracking, and reporting.
- A **Streamlit web application** (`streamlit_app/`) that provides an interactive UI for dataset upload, EDA, training, comparison, explainability, prediction, and experiment history.
- A **FastAPI backend** (`src/api/`) that exposes REST endpoints for training and prediction.
- **MLOps components**, including MLflow for experiment tracking and a SQLite database for experiment metadata.
- **Deployment and CI/CD** using Docker and GitHub Actions.

---

## End-to-End Data Flow

The end-to-end flow for a typical user session is:

1. **Dataset Upload (Streamlit UI)**  
   User uploads a CSV or Excel file via the Streamlit interface. The file is stored under `data/uploaded/` with a timestamped name.

2. **Data Loading & Profiling**  
   The core engine loads the dataset (`src/preprocessing/data_loader.py`), validates the file type, and generates basic metadata such as shape, data types, missing values, duplicates, and memory usage.

3. **Data Cleaning & Preprocessing**  
   Cleaning components (`src/preprocessing/`) handle missing values, duplicates, outliers, and type conversions. Preprocessing pipelines transform raw data into a clean feature matrix and target vector.

4. **Automatic EDA**  
   The EDA module (`src/eda/`) computes summary statistics and generates visualizations (histograms, boxplots, correlation heatmaps, target distributions). High-level insights are surfaced back to the UI.

5. **Feature Engineering**  
   Feature engineering components (`src/feature_engineering/`) apply encoding, scaling, feature selection, and optional dimensionality reduction based on the dataset characteristics and task type.

6. **Automatic Task Detection**  
   The task detector (`src/models/task_detector.py`, later) infers whether the problem is classification or regression using the target variable’s properties and simple heuristics.

7. **Model Training & Hyperparameter Tuning**  
   The training module (`src/models/trainer.py`) orchestrates model training for multiple algorithms (scikit-learn, XGBoost, LightGBM, CatBoost), with hyperparameter tuning implemented in `src/tuning/search.py`.

8. **Model Evaluation & Comparison**  
   Evaluation utilities (`src/evaluation/metrics.py`) compute task-specific metrics and cross-validation scores. A comparison component builds a leaderboard, considering accuracy, training time, inference time, and model complexity.

9. **Explainable AI (SHAP)**  
   The explainability module (`src/explainability/shap_explainer.py`) generates global and local explanations using SHAP, including feature importance and summary plots.

10. **Experiment Tracking (MLflow) & Persistence**  
    The MLflow integration (`src/mlflow_integration/tracker.py`) logs parameters, metrics, artifacts, SHAP plots, and models. Models are also persisted using joblib under `saved_models/` with basic versioning.

11. **Reporting & History**  
    The reporting module (`src/reports/pdf_generator.py`) generates PDF reports summarizing the entire pipeline. Experiment metadata is stored in SQLite (`src/database/sqlite_client.py`), enabling an experiment history view.

12. **Serving & Deployment**  
    - **FastAPI** (`src/api/fastapi_app.py`) exposes REST endpoints for `/train`, `/predict`, `/models`, and `/experiments`.  
    - **Streamlit** (`streamlit_app/app.py`) consumes both the core engine and FastAPI APIs.  
    - **Docker** and **GitHub Actions** orchestrate containerization and basic CI/CD for deployment.

---

## Logical Component Layers

### 1. Presentation Layer

- **Streamlit App (`streamlit_app/`)**  
  Handles user interaction: dataset upload, input forms, visualizations, tables, and PDF download links.

- **FastAPI App (`src/api/fastapi_app.py`)**  
  Exposes REST endpoints for programmatic access (e.g., `/predict`) and decouples the UI from the ML engine.

### 2. Core ML Engine (`src/`)

- **Preprocessing (`src/preprocessing/`)**  
  Data loading, validation, cleaning, and transformation into model-ready features.

- **EDA (`src/eda/`)**  
  Statistical summaries and visual exploratory analysis.

- **Feature Engineering (`src/feature_engineering/`)**  
  Encoders, scalers, feature selection, and PCA.

- **Models & Training (`src/models/`)**  
  Task detection, model configuration, training pipelines, and model persistence.

- **Evaluation (`src/evaluation/`)**  
  Metric computation, cross-validation, and leaderboard generation.

- **Hyperparameter Tuning (`src/tuning/`)**  
  Randomized and grid search utilities, with best-practice defaults.

- **Explainability (`src/explainability/`)**  
  SHAP-based global and local explanations.

### 3. MLOps & Infrastructure

- **MLflow Integration (`src/mlflow_integration/`)**  
  Experiment tracking, model registry integration (later), and artifact logging.

- **Database (`src/database/`)**  
  SQLite client for storing experiment metadata and best model records.

- **Reports (`src/reports/`)**  
  PDF report generation combining dataset summary, EDA, modeling, metrics, and explanations.

- **Utilities (`src/utils/`)**  
  Shared helpers for logging, configuration loading, file paths, and general utilities.

---

## Physical Folder Structure (Summary)

The repository uses a modular, package-based layout aligned with common ML engineering best practices.[web:13]

```bash
AutoML-studio/
├── config/                 # YAML configuration files (project, logging, MLflow)
├── data/                   # Sample and uploaded datasets (small samples only)
├── docs/                   # Project documentation (including this architecture file)
├── notebooks/              # Exploratory experiments and initial prototypes
├── src/                    # Core ML engine and backend
│   ├── preprocessing/
│   ├── eda/
│   ├── feature_engineering/
│   ├── models/
│   ├── evaluation/
│   ├── tuning/
│   ├── explainability/
│   ├── mlflow_integration/
│   ├── api/
│   ├── reports/
│   ├── database/
│   └── utils/
├── streamlit_app/          # Streamlit UI code
├── tests/                  # Unit and integration tests
├── saved_models/           # Persisted model artifacts (git-ignored)
├── mlruns/                 # Local MLflow tracking directory (git-ignored)
├── requirements.txt
├── README.md
└── docs/ARCHITECTURE.md
```

---

## Design Principles

- **Separation of concerns**: Business logic (ML pipeline) is isolated from UI (Streamlit) and serving (FastAPI).  
- **Config-driven**: Paths, experiment names, and environment-specific settings live in `config/`, not hard-coded in code.[web:13]  
- **Reusability**: All core functions are written as reusable, testable Python modules callable from notebooks, APIs, or UIs.  
- **MLOps readiness**: MLflow, SQLite, and Docker are integrated from the beginning, making the project production-oriented, not just a notebook.
