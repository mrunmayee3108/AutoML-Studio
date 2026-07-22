import logging 
import time
from typing import Dict, Any, Tuple

# classification--
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# regression--
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, task_type: str):
        self.task_type = task_type.lower()
        self.models = self._initialize_models()
        self.trained_models: Dict[str, Any] = {}
        logger.info(f"Initialized ModelTrainer for task type: {self.task_type}")

    def _initialize_models(self) -> Dict[str, Any]:
        # it will return a dictionary of un-trained models.
        if self.task_type == 'classification':
            return{
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
                'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
            }
        elif self.task_type == 'regression':
            return {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
                'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
            }
        else:
            raise ValueError("Invalid task type. Must be 'classification' or 'regression'.")

    def train_models(self, X_train, y_train) -> Dict[str, Tuple[Any, float]]:
        logger.info(f"Starting training loop for {len(self.models)} models...")
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                end_time = time.time()
                training_time = round(end_time-start_time, 4)
                self.trained_models[name] = model
                results[name] = (model, training_time)
                logger.info(f"successfully trained {name} in {training_time}s")
            except Exception as e:
                logger.error(f"Failed to train {name}. Error: {str(e)}")
        return results