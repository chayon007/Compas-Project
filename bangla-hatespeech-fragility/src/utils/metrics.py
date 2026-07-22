"""Metrics calculation and reporting utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from typing import Dict, Tuple
import json


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'accuracy': (y_pred == y_true).mean(),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['per_class'] = report
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            if y_pred_proba.ndim > 1:
                # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            else:
                # Binary
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
    
    return metrics


def save_metrics_to_csv(metrics_dict: Dict, output_path: str, experiment_name: str = ""):
    """
    Save metrics to CSV format for easy paper table generation.
    
    Args:
        metrics_dict: Dictionary of metrics (can be nested for different conditions)
        output_path: Path to save CSV
        experiment_name: Name of experiment for tracking
    """
    rows = []
    
    if isinstance(metrics_dict, dict):
        for key, value in metrics_dict.items():
            if isinstance(value, dict) and 'macro_f1' in value:
                # This is a metrics dict from calculate_metrics
                row = {
                    'experiment': experiment_name,
                    'condition': key,
                    'macro_f1': value.get('macro_f1', np.nan),
                    'weighted_f1': value.get('weighted_f1', np.nan),
                    'accuracy': value.get('accuracy', np.nan),
                    'precision': value.get('precision', np.nan),
                    'recall': value.get('recall', np.nan),
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")
    return df


def format_metrics_for_paper(metrics_dict: Dict) -> str:
    """Format metrics dictionary as LaTeX table row."""
    return (
        f"{metrics_dict.get('macro_f1', 0):.3f} & "
        f"{metrics_dict.get('weighted_f1', 0):.3f} & "
        f"{metrics_dict.get('accuracy', 0):.3f}"
    )
