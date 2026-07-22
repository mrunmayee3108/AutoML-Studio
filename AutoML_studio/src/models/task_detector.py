import pandas as pd
import logging 
logger = logging.getLogger(__name__)

class TaskDetector:
    # automatically detects whether the ml task is classification or regression
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        logger.info(f"TaskDetector initialized for target: {self.target_column}")

    def detect_task(self) -> str:
        """
        Infers the machine learning task based on dtype, unique values, and dataset proportions.
        Returns: 'classification' or 'regression'
        """
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        target_series = self.df[self.target_column]
        dtype = target_series.dtype

        # 1. Objects, Categories, and Booleans are always Classification
        if dtype == 'object' or dtype.name == 'category' or dtype == 'bool':
            logger.info("Task detected: Classification (Categorical/Boolean target).")
            return 'classification'
        
        # 2. Floats are almost always Regression
        if pd.api.types.is_float_dtype(target_series):
            logger.info("Task detected: Regression (Float target).")
            return 'regression'
        
        # 3. Integers require advanced proportional logic
        if pd.api.types.is_integer_dtype(target_series):
            unique_values = target_series.nunique()
            total_rows = len(target_series)
            unique_ratio = unique_values / total_rows
            
            # # Condition A: 5 or fewer unique values is definitively Classification
            # if unique_values <= 5:
            #     logger.info(f"Task detected: Classification ({unique_values} unique values).")
            #     return 'classification'
                
            # Condition B: Up to 20 unique values, BUT they make up less than 10% of the dataset
            # Example: 15 unique values in 500 rows (15/500 = 0.03). It's likely a rating system (Classification)
            # Example: 15 unique values in 20 rows (15/20 = 0.75). It's likely continuous data (Regression)
            if unique_values <= 20 and unique_ratio < 0.10:
                logger.info(f"Task detected: Classification (Ratio {unique_ratio:.2f} < 0.10).")
                return 'classification'
                
            # Condition C: Everything else is Regression
            else:
                logger.info(f"Task detected: Regression ({unique_values} unique values, Ratio: {unique_ratio:.2f}).")
                return 'regression'
        
        # Fallback
        logger.warning("Could not definitively detect task. Defaulting to Regression.")
        return 'regression'