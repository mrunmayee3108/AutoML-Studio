import os
import shutil
import logging
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MLflowTracker:
    # handles experiment tracking, artifact logging and model archiving
    def __init__(self, experiment_name: str = "AutoML_studio_Experiments", tracking_uri: str = "./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: '{self.experiment_name}' (id: {self.experiment_id})")
            else:
                self.experiment_id = self.experiment_id
                logger.info(f"Using existing mlflow experiment: '{self.experiment_name}' (id: {self.experiment_id})")
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracking: {str(e)}")
            raise e 

    def log_model_run(self, model_name: str, 
                      model_object: Any, metrics: Dict[str, Any], 
                      params: Optional[Dict[str, Any]] = None, 
                      artifacts: Optional[Dict[str, plt.figure]] = None, 
                      dataset_name: str = "Uploaded_Dataset") -> Optional[str]:
        # executes a complete mlflow run: logs params, metrics, plots, and the trained model.
        # returns the generated run id for tracking.
        logger.info(f"Initiating MLflow run for model: {model_name}...")
        
        try:
            with mlflow.start_run(run_name=f"{model_name}_Run") as run:
                run_id = run.info.run_id
                
                # 1. Log Metadata & Hyperparameters
                mlflow.log_param("dataset_name", dataset_name)
                mlflow.log_param("model_algorithm", model_name)
                
                if params:
                    for key, val in params.items():
                        # MLflow requires param values to be strings or numbers
                        mlflow.log_param(f"param_{key}", str(val))

                # 2. Log Evaluation & Hardware Profiling Metrics
                for metric_name, metric_val in metrics.items():
                    # Check if value is numeric before logging (skips string columns like 'Complexity')
                    if isinstance(metric_val, (int, float)) and not pd.isna(metric_val):
                        # Clean metric name for MLflow compatibility
                        clean_name = metric_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                        mlflow.log_metric(clean_name, float(metric_val))

                # 3. Log Visual Artifacts (e.g., SHAP plots, EDA charts)
                if artifacts:
                    # Create a temporary directory to save Matplotlib figures before pushing to MLflow
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for plot_name, fig in artifacts.items():
                            file_path = os.path.join(temp_dir, f"{plot_name}.png")
                            fig.savefig(file_path, bbox_inches='tight', dpi=150)
                            mlflow.log_artifact(file_path, artifact_path="visualizations")
                        logger.info(f"Successfully logged {len(artifacts)} visual artifacts.")

                # 4. Log the Trained Model Object
                # we use mlflow.sklearn for standard models; it automatically builds a conda/pip environment file!
                mlflow.sklearn.log_model(
                    sk_model=model_object,
                    artifact_path="model",
                    registered_model_name=None # Keep local for Phase 12; optional remote registry later
                )
                
                logger.info(f"MLflow run completed successfully! Run ID: {run_id}")
                return run_id

        except Exception as e:
            logger.error(f"Error during MLflow run for {model_name}: {str(e)}")
            return None