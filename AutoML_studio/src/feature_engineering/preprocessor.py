import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

logger = logging.getLogger(__name__)

class FeatureEngineer:
    # handles encoding, scaling and feature transformation
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        logger.info("FeatureEngineer initialized")
    
    def apply_encoding(self) -> None:
        # label encoding to target variable if categorical
        # and one hot encoding to feature variables
        if self.df[self.target_column].dtype == 'object' or self.df[self.target_column].dtype.name == 'category':
            le = LabelEncoder()
            self.df[self.target_column] = le.fit_transform(self.df[self.target_column].astype(str))
            self.label_encoders[self.target_column] = le
            logger.info(f"Applied Label Encoding to target: {self.target_column}")
        feature_cols = [col for col in self.df.columns if col!=self.target_column]
        categorical_features = self.df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            # pd.get_dummies is excellent here as it preserves DataFrame structure and column names
            self.df = pd.get_dummies(self.df, columns=categorical_features, drop_first=True)
            logger.info(f"Applied One-Hot Encoding to: {categorical_features}")

    def apply_scaling(self) -> None:
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        numeric_features = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        if numeric_features:
            scaler = StandardScaler()
            # We scale the data, but maintain the Pandas DataFrame format (sklearn returns numpy arrays by default)
            self.df[numeric_features] = scaler.fit_transform(self.df[numeric_features])
            self.scalers['standard'] = scaler
            self.scaled_features = numeric_features
            logger.info(f"Applied Standard Scaling to {len(numeric_features)} numerical features.")

    def transform(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted encodings and scalings to new data for inference."""
        df_inf = df_new.copy()
        # Handle one-hot encoding
        categorical_features = df_inf.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            df_inf = pd.get_dummies(df_inf, columns=categorical_features, drop_first=True)
            
        # Align columns to what was seen during training
        # We need the columns from self.df except target
        expected_cols = [col for col in self.df.columns if col != self.target_column]
        for col in expected_cols:
            if col not in df_inf.columns:
                df_inf[col] = 0
        df_inf = df_inf[expected_cols]

        # Apply scaling
        if 'standard' in self.scalers and hasattr(self, 'scaled_features'):
            df_inf[self.scaled_features] = self.scalers['standard'].transform(df_inf[self.scaled_features])
        
        return df_inf

    def get_processed_dataframe(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Separates features (X) and target (y) for model training."""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y