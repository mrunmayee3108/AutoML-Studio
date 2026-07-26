import logging 
import pandas as pd
import numpy as np 
from typing import Dict, Any
logger = logging.getLogger(__name__)

class ModelRecommender:
    # analyzes model comparision metrics to recommend the optimal model with justification
    def __init__(self, task_type: str):
        self.task_type = task_type.lower()
        logger.info(f"ModelRecommender initialized for task: {self.task_type}")

    def _normalize_series(self, series: pd.Series) -> pd.Series:
        # min max normalization to scale metrics betn 0.0 and 1.0 for fair waiting.
        if series.max() == series.min():
            return pd.Series(1.0, index = series.index)
        return (series-series.min())/(series.max()-series.min())

    def get_recommendation(self, comparision_df: pd.DataFrame) -> Dict[str, Any]:
        if comparision_df.empty:
            raise ValueError("Comparision dataframe is empty. Cannot generate recommendation.")
        df = comparision_df.copy()
        primary_metric = 'F1_Score' if self.task_type == 'classification' else 'R2_Score'
        norm_accuracy = self._normalize_series(df[primary_metric])
        norm_cv = self._normalize_series(df['CV_Mean_Score'])
        norm_latency = 1.0-self._normalize_series(df['Inference_Latency (ms/sample)'])
        norm_instability = 1.0-self._normalize_series(df['CV_Std_Dev (Stability)'])
        # Industry Weighting: 45% CV Accuracy, 25% Test Accuracy, 15% Speed, 15% Stability
        df['Composite_Score'] = (
            (0.45*norm_cv)+(0.25*norm_accuracy)+(0.15*norm_latency)+(0.15*norm_instability)
        )
        df.sort_values(by='Composite_Score', ascending=False, inplace=True)
        winner = df.iloc[0]
        runner_up = df.iloc[1] if len(df)>1 else None
        justification = self._generate_justification_text(winner, runner_up, primary_metric)
        logger.info(f"Recommendation generated: {winner['Model']} selected as winner")
        return {
            "winner_model": winner['Model'],
            "primary_metric_name": primary_metric,
            "primary_metric_val": winner[primary_metric],
            "cv_mean_val": winner['CV_Mean_Score'],
            "latency_val": winner['Inference_Latency (ms/sample)'],
            "complexity": winner['Complexity'],
            "composite_score": round(winner['Composite_Score'], 4),
            "justification_text": justification,
            "ranked_leaderboard": df
        }

    def _generate_justification_text(self, win: pd.Series, runner: pd.Series, metric_name: str) -> str:
        text = (
            f"**Executive summary & model recommendation**\n\n"
            f"AutoML Studio recommends deploying **{win['Model']}** as the product winner. "
            f"During testing, this model achieved an outstanding **{metric_name} of {win[metric_name]:.4f}** paired with a "
            f"robust **Cross-Validation Mean of {win['CV_Mean_Score']:.4f}** ($\pm${win['CV_Std_Dev (Stability)']:.4f}), "
            f"demonstrating exceptional generalization without severe overfitting.\n\n"
            f"**Why this model won:**\n"
            f"• **Hardware Efficiency:** It processes predictions with an inference latency of just **{win['Inference_Latency (ms/sample)']:.4f} ms/sample**, "
            f"making it suitable for high-throughput, real-time API endpoints.\n"
            f"• **Architectural Profile:** Categorized as a *{win['Complexity']}* algorithm, it balances predictive firepower with manageable memory consumption ({win['Model_Size (KB)']:.2f} KB).\n"
        )

        if runner is not None:
            text += (
                f"• **Competitive Comparison:** While **{runner['Model']}** performed competitively (Rank #2 with a {metric_name} of {runner[metric_name]:.4f}), "
                f"**{win['Model']}** secured the winning position due to superior overall balance across stability, execution speed, and cross-validated consistency."
            )

        return text
