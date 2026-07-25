import pandas as pd 
import numpy as np 
import logging 
from typing import Dict, Any, Tuple 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    mean_absolute_error, mean_squared_error, r2_score
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    # evaluates trained models and generates a sortable leaderboard. 
    def __init__(self, task_type: str):
        self.task_type = task_type.lower()
        logger.info(f"ModelEvaluator initialized for task: {self.task_type}")

    def evaluate_models(self, trained_models: Dict[str, Tuple[Any, float]], X_test, y_test) -> pd.DataFrame:
        """
        Generates a leaderboard dataframe with metrics for all trained models.
        """
        metrics_list = []

        for name, (model, train_time) in trained_models.items():
            try:
                y_pred = model.predict(X_test)
                if self.task_type == 'classification':
                    is_binary = len(np.unique(y_test)) == 2
                    avg_method = 'binary' if is_binary else 'weighted'

                    f1 = f1_score(y_test, y_pred, average=avg_method)
                    metrics = {
                        'Model': name,
                        'F1_Score': round(f1, 4),
                        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                        'Precision': round(precision_score(y_test, y_pred, average=avg_method, zero_division=0), 4),
                        'Recall': round(recall_score(y_test, y_pred, average=avg_method, zero_division=0), 4),
                        'Training_Time (s)': train_time
                    }
                    
                    if hasattr(model, "predict_proba") and is_binary:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        metrics['ROC_AUC'] = round(roc_auc_score(y_test, y_prob), 4)
                    else:
                        metrics['ROC_AUC'] = None

                elif self.task_type == 'regression':
                    r2 = r2_score(y_test, y_pred)
                    metrics = {
                        'Model': name,
                        'R2_Score': round(r2, 4),
                        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                        'MAE': round(mean_absolute_error(y_test, y_pred), 4),
                        'Training_Time (s)': train_time
                    }

                metrics_list.append(metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {str(e)}")
        leaderboard = pd.DataFrame(metrics_list)
        if self.task_type == 'classification':
            leaderboard.sort_values(by=['F1_Score', 'Training_Time (s)'], ascending=[False, True], inplace=True)
        else:
            leaderboard.sort_values(by=['R2_Score', 'Training_Time (s)'], ascending=[False, True], inplace=True)
            
        leaderboard.reset_index(drop=True, inplace=True)
        return leaderboard