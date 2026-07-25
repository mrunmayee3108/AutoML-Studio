import time 
import sys
import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

class ModelComparator:
    # considers latency, memory, complexity, and stability 
    
    def __init__(self, task_type: str):
        self.task_type = task_type.lower()
        logger.info(f"ModelComparator initialized for task: {self.task_type}")

    def _get_inference_latency(self, model: Any, X_test: pd.DataFrame, runs: int = 5) -> float:
        # measures avg time in millisec req to make a single pred.
        latencies = []
        for _ in range(runs):
            start_time = time.perf_counter()
            _ = model.predict(X_test)
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            latency_per_sample = total_time_ms/len(X_test)
            latencies.append(latency_per_sample)
        return round(float(np.mean(latencies)), 4)

    def _get_model_size_kb(self, model: Any) -> float:
        # ram/disk footprint
        try:
            serialized_model = pickle.dumps(model)
            size_kb = sys.getsizeof(serialized_model)/1024
            return round(size_kb, 2)
        except Exception as e:
            logger.warning(f"Could not calculate size for model: {str(e)}")
            return 0.0

    def _get_cv_stability(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[float, float]:
        # performs k fold cross validation to check model stability
        scoring_metric = 'f1_weighted' if self.task_type == 'Classification' else 'r2'
        try:
            scores = cross_val_score(model, X_train, y_train, cv = cv, scoring = scoring_metric, n_jobs = -1)
            return round(float(np.mean(scores)), 4), round(float(np.std(scores)), 4)
        except Exception as e:
            logger.error(f"cross validation failed: {str(e)}")
            return 0.0, 0.0

    def _get_model_complexity(self, model: Any) -> str:
        model_name = model.__class__.__name__
        if any(name in model_name for name in ['Linear', 'Logistic', 'Ridge', 'Lasso', 'Naive']):
            return "Low (Linear/Parametric)"
        elif any(name in model_name for name in ['Tree', 'KNeighbors']):
            return "Medium (Non-Parametric)"
        else:
            return "High (Ensemble/Boosting)"

    def generate_comparison_report(self, 
                                    trained_models: Dict[str, Tuple[Any, float]],
                                    leaderboard_df: pd.DataFrame,
                                    X_train: pd.DataFrame,
                                    y_train: pd.Series,
                                    X_test: pd.DataFrame) -> pd.DataFrame:

        # Combines accuracy metrics with hardware profiling to build the final production leaderboard.
        
        logger.info("Generating comprehensive model comparison report...")
        profiling_data = []

        for name, (model, _) in trained_models.items():
            logger.info(f"Profiling hardware and stability metrics for {name}...")
            latency = self._get_inference_latency(model, X_test)
            size_kb = self._get_model_size_kb(model)
            cv_mean, cv_std = self._get_cv_stability(model, X_train, y_train)
            complexity = self._get_model_complexity(model)

            profiling_data.append({
                'Model': name,
                'Inference_Latency (ms/sample)': latency,
                'Model_Size (KB)': size_kb,
                'CV_Mean_Score': cv_mean,
                'CV_Std_Dev (Stability)': cv_std,
                'Complexity': complexity
            })

        profiling_df = pd.DataFrame(profiling_data)
        
        full_report = pd.merge(leaderboard_df, profiling_df, on='Model', how='left')
        
        # identify the best Model (Highest CV Mean with acceptable stability)
        full_report.sort_values(by=['CV_Mean_Score', 'Inference_Latency (ms/sample)'], ascending=[False, True], inplace=True)
        full_report.reset_index(drop=True, inplace=True)
        
        logger.info("Comparison report generated successfully.")
        return full_report
