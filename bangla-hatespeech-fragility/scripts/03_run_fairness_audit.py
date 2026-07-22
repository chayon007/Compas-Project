"""
M4: Axis B - Dialectal fairness audit.

This script:
1. Loads master dataset
2. Groups data by dialect
3. Trains model on standard Bangla
4. Evaluates on each dialect group separately
5. Computes fairness gaps and per-dialect metrics
6. Saves results as CSV table
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fairness import DialectalFairnessAudit
from src.utils.config import config
from sklearn.metrics import f1_score, accuracy_score


def main():
    """Run dialectal fairness audit."""
    
    print("=" * 70)
    print("M4: AXIS B - DIALECTAL FAIRNESS AUDIT")
    print("=" * 70)
    
    # Check if master dataset exists
    master_path = config.PROCESSED_DATA_DIR / "master.csv"
    if not master_path.exists():
        print(f"\n❌ Master dataset not found!")
        print("Please run script 00_prepare_data.py first!")
        return
    
    # Check if TF-IDF model exists
    tfidf_model_path = config.RESULTS_DIR / "models" / "tfidf_baseline" / "model.pkl"
    tfidf_vectorizer_path = config.RESULTS_DIR / "models" / "tfidf_baseline" / "vectorizer.pkl"
    
    if not (tfidf_model_path.exists() and tfidf_vectorizer_path.exists()):
        print(f"\n❌ TF-IDF model not found!")
        print("Please run script 01_train_baseline.py first!")
        return
    
    # Load data and model
    print("\nLoading data and models...")
    master_df = pd.read_csv(master_path)
    
    with open(tfidf_model_path, "rb") as f:
        clf = pickle.load(f)
    with open(tfidf_vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    print(f"✓ Loaded {len(master_df)} samples")
    
    # Check for dialect information
    if 'dialect_group' not in master_df.columns:
        print("\n⚠️  No dialect_group column found!")
        print("Dataset does not contain dialect information.")
        print("This is expected if BIDWESH (dialectal dataset) was not loaded.")
        return
    
    # ===== Analyze Dialect Distribution =====
    print("\n" + "=" * 70)
    print("DIALECT DISTRIBUTION")
    print("=" * 70)
    print(master_df['dialect_group'].value_counts())
    
    # Remove 'unknown' dialect if present
    if 'unknown' in master_df['dialect_group'].values:
        print(f"\nRemoving {(master_df['dialect_group'] == 'unknown').sum()} samples with unknown dialect...")
        master_df = master_df[master_df['dialect_group'] != 'unknown']
    
    dialects = master_df['dialect_group'].unique()
    if len(dialects) < 2:
        print(f"\n⚠️  Only 1 dialect group found: {dialects}")
        print("Cannot perform fairness audit with single group.")
        return
    
    print(f"✓ {len(dialects)} dialect groups: {dialects.tolist()}")
    
    # ===== Split Data =====
    X_train, X_test, y_train, y_test, dialects_train, dialects_test = train_test_split(
        master_df['text_norm'].values,
        master_df['label'].values,
        master_df['dialect_group'].values,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=master_df['label'].values
    )
    
    # ===== Evaluate Per-Dialect =====
    print("\n" + "=" * 70)
    print("PER-DIALECT EVALUATION")
    print("=" * 70)
    
    # Vectorize test set
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    
    # Calculate per-dialect metrics
    audit = DialectalFairnessAudit()
    per_dialect_metrics = audit.calculate_per_dialect_metrics(y_test, y_pred, dialects_test)
    
    print(f"\nPer-dialect performance:")
    for dialect, metrics in sorted(per_dialect_metrics.items()):
        print(f"\n  {dialect.upper()}:")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Support: {metrics['support']}")
        if 'fpr' in metrics:
            print(f"    FPR: {metrics['fpr']:.4f}")
            print(f"    FNR: {metrics['fnr']:.4f}")
    
    # ===== Compute Fairness Gaps =====
    print("\n" + "=" * 70)
    print("FAIRNESS GAPS")
    print("=" * 70)
    
    fairness_gaps = audit.calculate_fairness_gaps(per_dialect_metrics)
    
    for gap_name, gap_value in sorted(fairness_gaps.items()):
        print(f"  {gap_name}: {gap_value:.4f}")
    
    # ===== Save Results =====
    metrics_df = audit.save_fairness_report_csv(
        per_dialect_metrics,
        config.TABLES_DIR / "dialect_fairness_metrics.csv"
    )
    
    # Save fairness gaps
    gaps_df = pd.DataFrame([fairness_gaps])
    gaps_df.to_csv(config.TABLES_DIR / "fairness_gaps.csv", index=False)
    
    print(f"\n✓ Fairness audit results saved:")
    print(f"  Metrics: {config.TABLES_DIR / 'dialect_fairness_metrics.csv'}")
    print(f"  Gaps: {config.TABLES_DIR / 'fairness_gaps.csv'}")
    
    # Print full report
    print("\n" + "=" * 70)
    print(audit.format_fairness_report(per_dialect_metrics, fairness_gaps))
    
    print(f"\n✓ Axis B evaluation complete!")
    print(f"  Next: Run script 04_run_continual_learning.py (Axis A)")


if __name__ == "__main__":
    main()
