import logging 
import shap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

class SHAPExplainer:
    # generates global and local explainability visualizations using SHAP.
    """
    Global Explanations (Summary Plot): Shows which features are most important across the entire dataset and whether high values of that feature drive predictions up or down.
    Local Explanations (Waterfall Plot): Explains a single specific row. For example: "The model denied Loan #104 primarily because Credit_Score pushed the probability down by -30%, even though Income tried to push it up by +10%."
    """
    def __init__(self, model: Any, X_train: pd.DataFrame, task_type: str):
        self.model = model 
        self.X_train = X_train
        self.task_type = task_type.lower()
        self.explainer = self._initialize_explainer()
        logger.info("SHAPExplainer initialized successfully.")

    def _initialize_explainer(self) -> Any:
        model_name = self.model.__class__.__name__
        logger.info(f"selecting shap explainer for model architecture: {model_name}")
        try:
            if any(name in model_name for name in ['Tree', 'Forest', 'XGB', 'LGBM', 'CatBoost', 'GradientBoosting']):
                return shap.TreeExplainer(self.model)
            elif any(name in model_name for name in ['Linear', 'Logistic', 'Ridge', 'Lasso']):
                return shap.LinearExplainer(self.model, self.X_train)
            else:
                # We use a background sample of 100 rows to prevent KernelExplainer from taking hours
                background_sample = shap.sample(self.X_train, 100)
                return shap.Explainer(self.model.predict, background_sample)
        except Exception as e:
            logger.warning(f"Optimized explainer failed ({str(e)}). Falling back to generic Explainer")
            background_sample = shap.sample(self.X_train, 100)
            return shap.Explainer(self.model, background_sample)

    def _get_shap_values(self, X_sample: pd.DataFrame) -> shap.Explanation:
        # safely calculates shap values and normalized array shapes across diff model types
        shap_values = self.explainer(X_sample)
        # 1. Regression vs. Classification Outputs:
        #    - Regression predicts 1 number (e.g., House Price). Its SHAP output 
        #      is a flat 2D spreadsheet: (Rows, Features). Shape length is 2.
        #    - Binary Classification predicts probabilities for BOTH Class 0 (No) 
        #      and Class 1 (Yes). Because there are 2 outputs per row, SHAP 
        #      stacks them into a 3D Cube: (Rows, Features, Classes). Shape length is 3!
        #
        # 2. The Problem:
        #    - Matplotlib plotting tools (like shap.summary_plot) will CRASH if 
        #      you hand them a 3D cube. They only know how to draw 2D spreadsheets.
        #
        # 3. The Solution ([:, :, 1]):
        #    - We slice the cube to extract ONLY Class 1 (the "Yes" prediction).
        #    - We ignore Class 0 because it is just the exact mathematical mirror 
        #      image of Class 1 anyway (+20% chance of Yes = -20% chance of No).
        #    - This slice flattens the 3D Cube (100, 5, 2) back into a safe 
        #      2D spreadsheet (100, 5) so our web dashboard never crashes!
        if self.task_type == 'classification' and len(shap_values.shape) == 3:
            # [All Rows, All Features, Only Class 1]
            shap_values = shap_values[:, :, 1]
        return shap_values

    def plot_global_summary(self, X_sample: Optional[pd.DataFrame] = None) -> plt.Figure:
        # generates the shap summary plot showing overall feature impact. 
        # returns a matplotlib figure object safely formatted for web display.
        if X_sample is None:
            X_sample = self.X_train.head(300)
        logger.info("Calculating SHAP values for global summary plot..")
        shap_values = self._get_shap_values(X_sample)
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Global Feature Importance & Impact (SHAP summary)", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_local_waterfall(self, row_index: int=0, X_sample: Optional[pd.DataFrame] = None) -> plt.Figure:
        # generates a shap waterfall plot explaining the exact decision for one single datapoint
        if X_sample is None: 
            X_sample = self.X_train.head(300)
        if row_index>=len(X_sample):
            raise IndexError(f"Row index {row_index} is out of bounds for sample size {len(X_sample)}")
        logger.info(f"Calculating local shap explanation for data row #{row_index}.")
        shap_values = self._get_shap_values(X_sample)
        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[row_index], show=False)
        plt.title(f"Local explanation for prediction (Row #{row_index})", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.close(fig)
        return fig

    def get_feature_importance_dataframe(self) -> pd.DataFrame:
        # returns a clean dataframe of features ranked by their absolute average shap impact.
        X_sample = self.X_train.head(300)
        shap_values = self._get_shap_values(X_sample)
        if isinstance(shap_values, shap.Explanation):
            vals = np.abs(shap_values.values).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'Mean_Absolute_SHAP': np.round(vals, 4)
        })
        importance_df.sort_values(by='Mean_Absolute_SHAP', ascending=False, inplace=True)
        importance_df.reset_index(drop=True, inplace=True)
        return importance_df