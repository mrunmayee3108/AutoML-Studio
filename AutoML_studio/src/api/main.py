import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from src.api.schemas import PredictionRequest, PredictionResponse, ExperimentResponse
from src.persistence.manager import ModelPersistenceManager
from src.prediction.predictor import ModelPredictor
from src.database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AutoML Studio API",
    description="Production REST API for AutoML Model Inference and Management",
    version="1.0.0"
)

persister = ModelPersistenceManager(base_directory="saved_models")
db = DatabaseManager(db_path="sqlite:///automl.db")

model_cache: Dict[str, ModelPredictor] = {}


@app.get("/health", tags=["System"])
def health_check():
    """Returns the health status of the API."""
    return {"status": "online", "message": "AutoML Studio API is running smoothly."}


@app.get("/experiments", response_model=List[ExperimentResponse], tags=["Metadata"])
def get_experiments():
    """Fetches the history of all completed AutoML experiments from the database."""
    try:
        experiments = db.get_all_experiments()
        return experiments
    except Exception as e:
        logger.error(f"Failed to fetch experiments: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal database error.")


@app.get("/models", tags=["Metadata"])
def get_saved_models():
    """Lists all available trained model versions saved on disk."""
    try:
        models = persister.list_saved_models()
        return {"total_models": len(models), "models": models}
    except Exception as e:
        logger.error(f"Failed to fetch models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal file system error.")


@app.post("/predict", response_model=List[PredictionResponse], tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Executes a real-time prediction by loading the specified model version 
    and passing the feature payload to the Inference Engine.
    """
    version_id = request.version_id
    features = request.features

    try:
        if version_id not in model_cache:
            logger.info(f"Model '{version_id}' not in cache. Loading from disk...")
            try:
                loaded_model, loaded_prep, metadata = persister.load_pipeline(version_id)
                predictor = ModelPredictor(
                    model=loaded_model,
                    task_type=metadata["task_type"],
                    expected_features=metadata["expected_features"],
                    class_names=metadata.get("class_names"),
                    preprocessor=loaded_prep
                )
                model_cache[version_id] = predictor
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Model version '{version_id}' not found.")
        predictor_engine = model_cache[version_id]
        results = predictor_engine.predict(features)
        
        return results

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error.")