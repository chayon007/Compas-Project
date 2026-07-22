"""Dialectal fairness audit utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix
)


class DialectalFairnessAudit:
    """Audit model fairness across Bangla dialects."""
    
    @staticmethod
    def calculate_per_dialect_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dialect_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics broken down by dialect group.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            dialect_labels: Dialect group labels
        
        Returns:
            Dictionary mapping dialect to metrics
        """
        metrics = {}
        dialects = np.unique(dialect_labels)
        
        for dialect in dialects:
            mask = dialect_labels == dialect
            y_true_d = y_true[mask]
            y_pred_d = y_pred[mask]
            
            if len(y_true_d) == 0:
                continue
            
            metrics[dialect] = {
                'f1': f1_score(y_true_d, y_pred_d, average='macro', zero_division=0),
                'precision': precision_score(y_true_d, y_pred_d, average='macro', zero_division=0),
                'recall': recall_score(y_true_d, y_pred_d, average='macro', zero_division=0),
                'support': len(y_true_d),
                'accuracy': (y_pred_d == y_true_d).mean(),
            }
            
            # False positive/negative rates (for binary classification)
            if len(np.unique(y_true)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_true_d, y_pred_d).ravel()
                metrics[dialect]['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics[dialect]['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_fairness_gaps(
        per_dialect_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate fairness gaps across dialects.
        
        Measures:
        - Max F1 gap
        - Max FPR (False Positive Rate) gap
        - Max FNR (False Negative Rate) gap
        
        Args:
            per_dialect_metrics: Dict from calculate_per_dialect_metrics
        
        Returns:
            Dictionary of fairness gap metrics
        """
        gaps = {}
        
        # F1 gap
        f1_scores = [m['f1'] for m in per_dialect_metrics.values()]
        if f1_scores:
            gaps['max_f1_gap'] = max(f1_scores) - min(f1_scores)
            gaps['mean_f1'] = np.mean(f1_scores)
            gaps['std_f1'] = np.std(f1_scores)
        
        # FPR gap
        fprs = [m.get('fpr', 0) for m in per_dialect_metrics.values() if 'fpr' in m]
        if fprs:
            gaps['max_fpr_gap'] = max(fprs) - min(fprs)
            gaps['mean_fpr'] = np.mean(fprs)
        
        # FNR gap
        fnrs = [m.get('fnr', 0) for m in per_dialect_metrics.values() if 'fnr' in m]
        if fnrs:
            gaps['max_fnr_gap'] = max(fnrs) - min(fnrs)
            gaps['mean_fnr'] = np.mean(fnrs)
        
        # Equalized Odds Difference (max FPR or FNR gap)
        if fprs and fnrs:
            gaps['equalized_odds_diff'] = max(
                max(fprs) - min(fprs),
                max(fnrs) - min(fnrs)
            )
        
        return gaps
    
    @staticmethod
    def format_fairness_report(
        per_dialect_metrics: Dict[str, Dict[str, float]],
        fairness_gaps: Dict[str, float]
    ) -> str:
        """
        Format fairness audit results as readable report.
        
        Args:
            per_dialect_metrics: Per-dialect metrics
            fairness_gaps: Fairness gap metrics
        
        Returns:
            Formatted report string
        """
        report = "=" * 70 + "\n"
        report += "DIALECTAL FAIRNESS AUDIT REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Per-dialect breakdown
        report += "PER-DIALECT METRICS:\n"
        report += "-" * 70 + "\n"
        report += f"{'Dialect':<20} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Support':<8}\n"
        report += "-" * 70 + "\n"
        
        for dialect, metrics in sorted(per_dialect_metrics.items()):
            report += (f"{dialect:<20} {metrics['f1']:<10.3f} "
                      f"{metrics['precision']:<12.3f} {metrics['recall']:<10.3f} "
                      f"{metrics['support']:<8d}\n")
        
        # Fairness gaps
        report += "\n" + "=" * 70 + "\n"
        report += "FAIRNESS GAPS:\n"
        report += "-" * 70 + "\n"
        
        for gap_name, gap_value in sorted(fairness_gaps.items()):
            report += f"{gap_name:<25}: {gap_value:.4f}\n"
        
        report += "=" * 70 + "\n"
        
        return report
    
    @staticmethod
    def save_fairness_report_csv(
        per_dialect_metrics: Dict[str, Dict[str, float]],
        output_path: str
    ) -> pd.DataFrame:
        """
        Save per-dialect metrics to CSV for paper tables.
        
        Args:
            per_dialect_metrics: Per-dialect metrics
            output_path: Path to save CSV
        
        Returns:
            DataFrame
        """
        rows = []
        for dialect, metrics in per_dialect_metrics.items():
            rows.append({
                'dialect': dialect,
                'f1': metrics.get('f1', np.nan),
                'precision': metrics.get('precision', np.nan),
                'recall': metrics.get('recall', np.nan),
                'fpr': metrics.get('fpr', np.nan),
                'fnr': metrics.get('fnr', np.nan),
                'accuracy': metrics.get('accuracy', np.nan),
                'support': metrics.get('support', 0),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Fairness metrics saved to {output_path}")
        
        return df
