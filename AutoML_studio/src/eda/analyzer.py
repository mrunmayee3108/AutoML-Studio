import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class EDAAnalyzer:
    # generates automated statistical insights and visualization objects.
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_theme(style = 'whitegrid', palette = 'muted')
        logger.info("EDAAnalyzer initialized")
    
    def get_summary_statistics(self) -> pd.DataFrame:
        return self.df.describe().T
    
    def plot_numerical_distributions(self) -> List[plt.figure]:
        figures = []
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(self.df[col], kde=True, ax=ax, color="steelblue")
            ax.set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            figures.append(fig)
            plt.close(fig)
        logger.info(f"Generated {len(figures)} distribution plots.")
        return figures
    
    def plot_correlation_heatmap(self) -> plt.figure:
        numeric_df = self.df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        if numeric_df.shape[1]<2:
            ax.text(0.5, 0.5, "Not enough numerical columns for correlation", ha='center', va='center')
            return fig
        corr = numeric_df.corr()
        # creating a mask to hide the upper triangle, since its redundant.
        mask = np.triu(np.ones_vars(corr.shape), k=1).astype(bool)
        
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="collwarm", vmax=1, vmin=-1, center=0, square=True, linewidths=.5, ax=ax)
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.close(fig)
        logger.info("Generated correlation heatmap")
        return fig
    