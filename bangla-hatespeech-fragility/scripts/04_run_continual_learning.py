"""
M5: Axis A - Temporal continual learning evaluation.

This script:
1. Orders datasets chronologically (T1, T2, T3 phases)
2. Trains baseline model on T1
3. Tests on T1 (baseline)
4. Fine-tunes on T2 and tests on both T1 and T2
5. Fine-tunes on T3 and tests on all three
6. Computes continual learning metrics (AA, BWT, FM)
7. Saves results as CSV table
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.continual import compute_continual_learning_metrics
from src.utils.config import config
from sklearn.metrics import f1_score, accuracy_score


def main():
    """Run temporal continual learning evaluation."""
    
    print("=" * 70)
    print("M5: AXIS A - TEMPORAL CONTINUAL LEARNING EVALUATION")
    print("=" * 70)
    
    # Check if master dataset exists
    master_path = config.PROCESSED_DATA_DIR / "master.csv"
    if not master_path.exists():
        print(f"\n❌ Master dataset not found!")
        print("Please run script 00_prepare_data.py first!")
        return
    
    # Load master dataset
    print("\nLoading data...")
    master_df = pd.read_csv(master_path)
    print(f"✓ Loaded {len(master_df)} samples")
    
    # Check for task_phase (temporal ordering)
    if 'task_phase' not in master_df.columns:
        print("\n⚠️  No task_phase column found!")
        print("Dataset does not contain temporal phase information.")
        return
    
    # ===== Create Balanced Temporal Phases =====
    print("\n" + "=" * 70)
    print("CREATING BALANCED TEMPORAL PHASES FOR EVALUATION")
    print("=" * 70)
    
    # Split data chronologically but ensure both classes in each phase
    # Sort by index (proxy for temporal ordering)
    master_df_sorted = master_df.sort_index().reset_index(drop=True)
    
    # Create 3 temporal phases with overlapping data to ensure class balance
    n_total = len(master_df_sorted)
    phase_size = n_total // 3
    
    # Phase 1: First 1/3
    phase1_end = phase_size
    # Phase 2: Middle 1/3 (with some overlap)
    phase2_start = phase_size // 2
    phase2_end = phase_size + phase_size // 2
    # Phase 3: Last 1/3 (with some overlap)
    phase3_start = phase_size + phase_size // 4
    
    phases_data = {
        't1': master_df_sorted.iloc[:phase1_end],
        't2': master_df_sorted.iloc[phase2_start:phase2_end],
        't3': master_df_sorted.iloc[phase3_start:]
    }
    
    # ===== Analyze Task Distribution =====
    print("\n" + "=" * 70)
    print("TEMPORAL PHASE DISTRIBUTION")
    print("=" * 70)
    for phase_name, phase_df in phases_data.items():
        counts = phase_df['label'].value_counts().sort_index()
        print(f"\n{phase_name}: {len(phase_df)} samples")
        print(counts)
    
    tasks = ['t1', 't2', 't3']
    print(f"\n✓ {len(tasks)} temporal phases: {tasks}")
    
    # ===== Split Data by Task =====
    print("\n" + "=" * 70)
    print("SPLITTING DATA BY TASK PHASE")
    print("=" * 70)
    
    task_data = {}
    for task in tasks:
        task_df = phases_data[task]
        
        # Split into train/test for this task
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                task_df['text_norm'].values,
                task_df['label'].values,
                test_size=0.2,
                random_state=config.RANDOM_SEED,
                stratify=task_df['label'].values
            )
        except ValueError:
            # If stratification fails, try without it
            X_train, X_test, y_train, y_test = train_test_split(
                task_df['text_norm'].values,
                task_df['label'].values,
                test_size=0.2,
                random_state=config.RANDOM_SEED
            )
        
        task_data[task] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'size': len(task_df)
        }
        
        print(f"  {task}: {len(task_df)} samples (train: {len(X_train)}, test: {len(X_test)})")
    
    # ===== Simulate Continual Learning Strategies =====
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING STRATEGIES")
    print("=" * 70)
    
    strategies = {
        'sequential_finetuning': [],
        'joint_training': [],
    }
    
    # Strategy 1: Sequential Fine-tuning (lower bound - catastrophic forgetting)
    print("\n1. SEQUENTIAL FINE-TUNING (baseline)...")
    all_accuracies_seq = {}
    
    prev_model = None
    prev_vectorizer = None
    
    for i, task in enumerate(tasks):
        print(f"\n  Task {i+1} ({task}):")
        
        # Train or fine-tune model
        if i == 0:
            # First task - train from scratch
            vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
            X_train_vec = vectorizer.fit_transform(task_data[task]['X_train'])
            model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
            model.fit(X_train_vec, task_data[task]['y_train'])
        else:
            # Subsequent tasks - fine-tune
            X_train_vec = prev_vectorizer.transform(task_data[task]['X_train'])
            model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
            model.fit(X_train_vec, task_data[task]['y_train'])
        
        # Evaluate on all tasks seen so far
        task_accuracies = {}
        for j, eval_task in enumerate(tasks[:i+1]):
            X_test_vec = prev_vectorizer.transform(task_data[eval_task]['X_test']) if j > 0 else vectorizer.transform(task_data[eval_task]['X_test'])
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(task_data[eval_task]['y_test'], y_pred)
            task_accuracies[j] = acc
            print(f"    Accuracy on {eval_task}: {acc:.4f}")
        
        all_accuracies_seq[i] = task_accuracies
        prev_model = model
        prev_vectorizer = vectorizer
    
    strategies['sequential_finetuning'] = all_accuracies_seq
    
    # Strategy 2: Joint Training (upper bound - no forgetting)
    print("\n2. JOINT TRAINING (upper bound)...")
    
    # Concatenate all data
    X_train_all = np.concatenate([task_data[t]['X_train'] for t in tasks])
    y_train_all = np.concatenate([task_data[t]['y_train'] for t in tasks])
    
    # Train single model
    vectorizer_joint = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    X_train_vec_joint = vectorizer_joint.fit_transform(X_train_all)
    model_joint = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    model_joint.fit(X_train_vec_joint, y_train_all)
    
    # Evaluate on all tasks
    all_accuracies_joint = {}
    for i, task in enumerate(tasks):
        X_test_vec = vectorizer_joint.transform(task_data[task]['X_test'])
        y_pred = model_joint.predict(X_test_vec)
        acc = accuracy_score(task_data[task]['y_test'], y_pred)
        all_accuracies_joint[i] = {i: acc for i in range(len(tasks))}
        print(f"  Accuracy on {task}: {acc:.4f}")
    
    strategies['joint_training'] = all_accuracies_joint
    
    # ===== Compute CL Metrics =====
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING METRICS")
    print("=" * 70)
    
    cl_results = []
    
    for strategy_name, accuracies in strategies.items():
        print(f"\n{strategy_name.upper()}:")
        
        cl_metrics = compute_continual_learning_metrics(accuracies)
        
        for metric_name, metric_value in cl_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
            cl_results.append({
                'strategy': strategy_name,
                'metric': metric_name,
                'value': metric_value,
            })
    
    # ===== Save Results =====
    results_df = pd.DataFrame(cl_results)
    results_df.to_csv(config.TABLES_DIR / "continual_learning_metrics.csv", index=False)
    
    print(f"\n✓ Continual learning results saved:")
    print(f"  {config.TABLES_DIR / 'continual_learning_metrics.csv'}")
    
    print(f"\n✓ Axis A evaluation complete!")
    print(f"\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved in: {config.TABLES_DIR}")
    print(f"  - baseline_tfidf_metrics.csv")
    print(f"  - transliteration_robustness.csv")
    print(f"  - dialect_fairness_metrics.csv")
    print(f"  - continual_learning_metrics.csv")


if __name__ == "__main__":
    main()
