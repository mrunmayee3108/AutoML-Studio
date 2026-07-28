import os 
import json
import joblib
import platform 
import sklearn
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

logging.basicConfig(level = logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)

class ModelPersistenceManager:
    # manager for serializing, versioning, and deserializing ml pipelines and their accompanying metadata.
    def __init__(self, base_directory: str = "saved_models"):
        self.base_directory = base_directory
        os.makedirs(self.base_directory, exist_ok=True)
        logger.info(f"ModelPersistenceManager initialized. Artifacts directory: '{self.base_directory}'")

    def save_pipeline(self, model: Any, model_name: str, task_type: str, expected_features: List[str], metrics: Dict[str, Any], preprocessor: Optional[Any] = None, class_names: Optional[List[str]]=None) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = model_name.replace(" ", "_").lower()
            version_id = f"{clean_name}_v_{timestamp}"
            artifact_bundle = {
                "model": model,
                "preprocessor": preprocessor,
                "expected_features": expected_features,
                "class_names": class_names,
                "task_type": task_type.lower()
            }
            joblib_filename = f"{version_id}.joblib"
            joblib_path = os.path.join(self.base_directory, joblib_filename)
            joblib.dump(artifact_bundle, joblib_path)
            logger.info(f"Pipeline artifact successfully saved to: {joblib_path}")

            metadata = {
                "version_id": version_id,
                "model_name": model_name,
                "task_type": task_type.lower(),
                "timestamp": timestamp,
                "expected_features": expected_features,
                "class_names": class_names,
                "metrics": metrics,
                "artifact_file": joblib_filename,
                "environment_info": {
                    "python_version": platform.python_version(),
                    "sklearn_version": sklearn.__version__,
                    "joblib_version": joblib.__version__
                }
            }

            json_filename = f"{version_id}_metadata.json"
            json_path = os.path.join(self.base_directory, json_filename)
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata summary successfully saved to: {json_path}")

            return version_id

        except Exception as e:
            logger.error(f"Failed to save pipeline artifact: {str(e)}")
            raise e
        
    def load_pipeline(self, version_id: str) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
        joblib_path = os.path.join(self.base_directory, f"{version_id}.joblib")
        json_path = os.path.join(self.base_directory, f"{version_id}_metadata.json")

        if not os.path.exists(joblib_path) or not os.path.exists(json_path):
            error_msg = f"Artifacts for version '{version_id}' not found in directory '{self.base_directory}'."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading pipeline artifact: {joblib_path} ...")
            bundle = joblib.load(joblib_path)

            logger.info(f"Loading companion metadata: {json_path} ...")
            with open(json_path, "r") as f:
                metadata = json.load(f)
            saved_sklearn = metadata.get("environment_info", {}).get("sklearn_version")
            if saved_sklearn and saved_sklearn != sklearn.__version__:
                logger.warning(
                    f"Version mismatch! Model was trained with sklearn {saved_sklearn}, "
                    f"but current environment is running sklearn {sklearn.__version__}."
                )

            logger.info(f"Successfully loaded version '{version_id}' ({metadata.get('model_name')}).")
            return bundle["model"], bundle.get("preprocessor"), metadata

        except Exception as e:
            logger.error(f"Failed to load pipeline version '{version_id}': {str(e)}")
            raise e

    def list_saved_models(self) -> List[Dict[str, Any]]:
        saved_models = []
        if not os.path.exists(self.base_directory):
            return saved_models

        for file in os.listdir(self.base_directory):
            if file.endswith("_metadata.json"):
                try:
                    with open(os.path.join(self.base_directory, file), "r") as f:
                        data = json.load(f)
                        saved_models.append({
                            "version_id": data.get("version_id"),
                            "model_name": data.get("model_name"),
                            "task_type": data.get("task_type"),
                            "timestamp": data.get("timestamp"),
                            "metrics": data.get("metrics")
                        })
                except Exception as e:
                    logger.warning(f"Could not read metadata file '{file}': {str(e)}")
        saved_models.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
        return saved_models

    def get_latest_version_id(self) -> Optional[str]:
        models = self.list_saved_models()
        if not models:
            logger.warning(f"No saved models found in '{self.base_directory}'.")
            return None
        return str(models[0]["version_id"])