import pandas as pd 
import numpy as np 
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

class DataCleaner:
    # handles missing vals, duplicates, and outliers
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_report: List[str] = []
        logger.info("DataCleaner initialized.")
    
    def drop_duplicates(self) -> None:
        initial_rows = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace = True)
        dropped = initial_rows - self.df.shape[0]
        msg = f"Dropped {dropped} duplicate rows."
        self.cleaning_report.append(msg)
        logger.info(msg)
    
    def drop_columns(self, columns: List[str]) -> None:
        valid_cols = [col for col in columns if col in self.df.columns]
        self.df.drop(columns = valid_cols, inplace=True)
        msg = f"Dropped columns: {valid_cols}"
        self.cleaning_report.append(msg)
        logger.info(msg)
    
    def handle_missing_values(self, strategy: str = 'mean', columns: List[str] = None) -> None:
        cols_to_fill = columns if columns else self.df.columns 
        initial_missing = self.df[cols_to_fill].isnull().sum().sum()
        if initial_missing == 0: 
            return 
        if strategy == 'drop':
            self.df.dropna(subset=cols_to_fill, inplace=True)
        else:
            for col in cols_to_fill:
                if self.df[col].isnull().sum()>0:
                    if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                        fill_val = self.df[col].mean()
                    elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                        fill_val = self.df[col].median()
                    elif strategy == 'mode':
                        fill_val = self.df[col].mode()[0]
                    else:
                        fill_val = self.df[col].mode()[0]
                    
                    self.df[col].fillna(fill_val, inplace = True)

        msg = f"Handled missing values using '{strategy}' strategy for columns: {cols_to_fill}"
        self.cleaning_report.append(msg)
        logger.info(msg)

    # Since this is an AutoML platform, I didn't want to force the user to make statistical decisions.
    # I engineered the pipeline to automatically evaluate feature skewness. 
    # If the skewness exceeded 1.0, the pipeline dynamically applied IQR Winsorization(capping) to protect linear models; 
    # otherwise, it preserved the natural variance for tree-based algorithms.
    # When skewness is below 1.0, the data distribution is stable enough that linear models won't suffer catastrophic errors from outliers,
    # allowing us to preserve the raw variance for tree-based models

    def auto_handle_outliers(self, factor: float = 1.5) -> None:
        """
        AutoML Logic: Evaluates the skewness of numeric columns.
        If a column is highly skewed (absolute skewness > 1.0), it applies IQR capping.
        Otherwise, it leaves the outliers alone to preserve natural variance.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        columns_capped = []
        total_outliers_capped = 0
        
        for col in numeric_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 1.0:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - (factor * IQR)
                upper_bound = Q3 + (factor * IQR)
                outliers_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                    self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                    
                    total_outliers_capped += outliers_count
                    columns_capped.append(col)

        if columns_capped:
            msg = f"AutoML applied capping to {total_outliers_capped} outliers across highly skewed columns: {columns_capped}."
        else:
            msg = "AutoML detected no severe skewness; no outlier capping was required."
            
        self.cleaning_report.append(msg)
        logger.info(msg)
    
    def get_cleaned_dataFrame(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> List[str]:
        return self.cleaning_report

