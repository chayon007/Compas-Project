"""
Add statistical rigor to results: compute confidence intervals.
Uses bootstrap resampling to estimate 95% CI for key metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.config import config


def bootstrap_ci(scores, n_bootstraps=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    bootstrapped = np.random.choice(scores, size=(n_bootstraps, len(scores)), replace=True)
    bootstrap_means = np.mean(bootstrapped, axis=1)
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    mean = np.mean(scores)
    return mean, lower, upper


def compute_stats_for_baseline():
    """Compute baseline stats with confidence intervals."""
    
    print("="*70)
    print("COMPUTING STATISTICAL RIGOR: CONFIDENCE INTERVALS")
    print("="*70)
    
    # Load data
    master_path = config.PROCESSED_DATA_DIR / "master.csv"
    master_df = pd.read_csv(master_path)
    
    # Split train/test with indices
    indices = np.arange(len(master_df))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        master_df['text'].values,
        master_df['label'].values,
        indices,
        test_size=0.2,
        random_state=42,
        stratify=master_df['label'].values
    )
    
    # Train baseline model
    print("\nTraining baseline model...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)
    
    # Compute predictions per sample (for CI computation)
    y_pred_correct = (y_pred == y_test).astype(int)
    f1_per_sample = []
    
    # Approximate F1 by groups
    for label in [0, 1]:
        mask = y_test == label
        tp = np.sum((y_pred[mask] == 1) & (y_test[mask] == 1))
        fp = np.sum((y_pred[mask] == 1) & (y_test[mask] == 0))
        fn = np.sum((y_pred[mask] == 0) & (y_test[mask] == 1))
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
    
    # Bootstrap CIs
    print("\nComputing bootstrap confidence intervals (1000 resamples)...")
    acc_mean, acc_lower, acc_upper = bootstrap_ci(y_pred_correct, n_bootstraps=1000)
    
    # F1 bootstrap
    f1_scores = []
    for _ in range(1000):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        f1 = f1_score(y_test[indices], y_pred[indices], zero_division=0)
        f1_scores.append(f1)
    
    f1_mean = np.mean(f1_scores)
    f1_lower = np.percentile(f1_scores, 2.5)
    f1_upper = np.percentile(f1_scores, 97.5)
    
    # Fairness metrics with CI
    dialect_groups = master_df.iloc[test_indices]['dialect_group'].values if 'dialect_group' in master_df.columns else np.zeros(len(X_test))
    
    print("\n" + "="*70)
    print("BASELINE RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("="*70)
    print(f"\nAccuracy: {acc_mean:.4f} (95% CI: [{acc_lower:.4f}, {acc_upper:.4f}])")
    print(f"F1-Score: {f1_mean:.4f} (95% CI: [{f1_lower:.4f}, {f1_upper:.4f}])")
    
    # Save results
    results = {
        'metric': ['Accuracy', 'F1-Score'],
        'mean': [acc_mean, f1_mean],
        'lower_95ci': [acc_lower, f1_lower],
        'upper_95ci': [acc_upper, f1_upper],
        'ci_width': [acc_upper - acc_lower, f1_upper - f1_lower]
    }
    
    results_df = pd.DataFrame(results)
    output_path = config.TABLES_DIR / "baseline_with_confidence_intervals.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == '__main__':
    np.random.seed(42)
    compute_stats_for_baseline()
