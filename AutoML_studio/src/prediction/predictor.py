import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# validates input schema, applies preprocessing, predicts outcomes, and formats results as API-ready JSON with confidence scores.

class ModelPredictor:

    def __init__(
        self,
        model: Any,
        task_type: str,
        expected_features: List[str],
        class_names: Optional[List[str]] = None,
        preprocessor: Optional[Any] = None):

        self.model = model
        self.task_type = task_type.lower()
        self.expected_features = expected_features
        self.class_names = class_names
        self.preprocessor = preprocessor

        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be either 'classification' or 'regression'.")

        logger.info(f"ModelPredictor initialized for {self.task_type.upper()} task with {len(self.expected_features)} expected features.")

    def _validate_and_prepare_input(self, input_data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
        # converting dictionaries or lists of dictionaries to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("Input data must be a pandas DataFrame, a dictionary, or a list of dictionaries.")

        missing_cols = set(self.expected_features) - set(df.columns)
        if missing_cols:
            error_msg = f"Input data is missing required features: {sorted(list(missing_cols))}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        df = df[self.expected_features]

        if self.preprocessor is not None:
            try:
                logger.info("Applying preprocessing pipeline to inference data...")
                transformed_data = self.preprocessor.transform(df)
                if isinstance(transformed_data, np.ndarray):
                    df = pd.DataFrame(transformed_data, columns=self.expected_features, index=df.index)
                else:
                    df = transformed_data
            except Exception as e:
                logger.error(f"Preprocessing failed during inference: {str(e)}")
                raise e

        return df

    def predict(self, input_data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Starting inference validation and preprocessing...")
        df = self._validate_and_prepare_input(input_data)
        num_rows = len(df)
        logger.info(f"Executing prediction for {num_rows} instance(s)...")

        results = []

        try:
            raw_predictions = self.model.predict(df)

            if self.task_type == "classification":
                has_proba = hasattr(self.model, "predict_proba")
                probabilities = self.model.predict_proba(df) if has_proba else None

                for i in range(num_rows):
                    pred_val = raw_predictions[i]
                    
                    if self.class_names and isinstance(pred_val, (int, np.integer)) and pred_val < len(self.class_names):
                        pred_label = self.class_names[pred_val]
                    else:
                        pred_label = str(pred_val)

                    row_result = {
                        "row_index": i,
                        "prediction": pred_label,
                        "raw_prediction_value": int(pred_val) if isinstance(pred_val, (int, np.integer)) else pred_val,
                        "task_type": "Classification"
                    }

                    if has_proba and probabilities is not None:
                        row_probs = probabilities[i]
                        max_prob = float(np.max(row_probs))
                        
                        if self.class_names and len(self.class_names) == len(row_probs):
                            prob_dict = {f"prob_{self.class_names[idx]}": round(float(p), 4) for idx, p in enumerate(row_probs)}
                        else:
                            prob_dict = {f"prob_class_{idx}": round(float(p), 4) for idx, p in enumerate(row_probs)}

                        row_result["confidence_score"] = f"{round(max_prob * 100, 2)}%"
                        row_result["confidence_value"] = round(max_prob, 4)
                        row_result.update(prob_dict)
                    else:
                        row_result["confidence_score"] = "N/A (No proba support)"
                        row_result["confidence_value"] = 1.0

                    results.append(row_result)

            else:
                for i in range(num_rows):
                    pred_val = float(raw_predictions[i])
                    results.append({
                        "row_index": i,
                        "prediction": round(pred_val, 4),
                        "raw_prediction_value": pred_val,
                        "confidence_score": "N/A (Continuous Target)",
                        "confidence_value": 1.0,
                        "task_type": "Regression"
                    })

            logger.info("Inference completed successfully.")
            return results

        except Exception as e:
            logger.error(f"Inference execution failed: {str(e)}")
            raise e