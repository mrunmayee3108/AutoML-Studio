import logging
import pandas as pd
from typing import Any, Dict
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, task_type: str):
        self.task_type = task_type.lower()
        self.param_grids = self._get_param_grids()

    def _get_param_grids(self) -> Dict[str, Dict[str, Any]]:
        return {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear']
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy'] 
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'CatBoost': {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8]
            }
        }

    def tune_model(self, model_name: str, base_model: Any, X_train, y_train) -> Any:
        if model_name not in self.param_grids:
            logger.warning(f"No tuning grid defined for {model_name}. Returning base model.")
            return base_model

        grid = self.param_grids[model_name]
        logger.info(f"Starting Hyperparameter Tuning for {model_name}...")

        # I started with RandomizedSearchCV because of the combinatorial explosion associated with GridSearch. Since this is a web platform, blocking the main thread for an hour to exhaustively search a parameter grid provides a terrible user experience. By using RandomizedSearchCV with a fixed n_iter, I bounded the compute time while still capturing a model that is heavily optimized. In a true enterprise environment, the best practice is actually to chain them: use RandomizedSearch to find the general 'neighborhood' of good parameters, and then run a tiny GridSearch right around those specific values to pinpoint the absolute maximum.

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=grid,
            n_iter=10, 
            cv=3,      
            scoring='f1' if self.task_type == 'classification' else 'r2',
            n_jobs=-1, # Use all available CPU cores
            random_state=42
        )

        try:
            search.fit(X_train, y_train)
            logger.info(f"Tuning complete. Best params: {search.best_params_}")
            return search.best_estimator_
        except Exception as e:
            logger.error(f"Tuning failed: {str(e)}")
            return base_model