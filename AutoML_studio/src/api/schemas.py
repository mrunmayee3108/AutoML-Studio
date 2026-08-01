from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PredictionRequest(BaseModel):
    """Schema for incoming prediction requests."""
    version_id: str
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    """Schema for outgoing prediction results."""
    row_index: int
    prediction: str
    raw_prediction_value: float
    task_type: str
    confidence_score: str
    confidence_value: float

class ExperimentResponse(BaseModel):
    """Schema for returning database experiment records."""
    id: int
    project_name: str
    task_type: str
    dataset_name: str
    best_model: str
    metric: str
    version_id: str
    report_path: Optional[str]
    created_at: str