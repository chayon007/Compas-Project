"""
M3: Axis C - Evaluate transliteration robustness.

This script:
1. Loads master dataset
2. Generates transliteration attacks (L1, L2, L3)
3. Evaluates baseline model on clean vs attacked data
4. Computes robustness metrics and accuracy drops
5. Saves results as CSV table
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

from src.attacks import TransliterationAttack
from src.utils.config import config
from src.utils.metrics import calculate_metrics
from sklearn.metrics import f1_score, accuracy_score


def main():
    """Run transliteration robustness evaluation."""
    
    print("=" * 70)
    print("M3: AXIS C - TRANSLITERATION ROBUSTNESS EVALUATION")
    print("=" * 70)
    
    # Check if master dataset exists
    master_path = config.PROCESSED_DATA_DIR / "master.csv"
    if not master_path.exists():
        print(f"\n❌ Master dataset not found!")
        print("Please run scripts 00_prepare_data.py first!")
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
    
    print(f"✓ Loaded {len(master_df)} samples and models")
    
    # Split test data
    X_train, X_test, y_train, y_test = train_test_split(
        master_df['text_norm'].values,
        master_df['label'].values,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=master_df['label'].values
    )
    
    # ===== Generate Attacks =====
    print("\n" + "=" * 70)
    print("GENERATING ADVERSARIAL ATTACKS")
    print("=" * 70)
    
    attack_gen = TransliterationAttack(random_seed=config.RANDOM_SEED)
    
    # Generate attacked texts for test set
    print("\nGenerating L1, L2, L3 attacks on test set...")
    attacked_dataset = attack_gen.generate_attacked_dataset(
        texts=X_test.tolist(),
        labels=y_test.tolist(),
        attack_levels=['l1', 'l2', 'l3'],
        perturbation_ratio=0.2  # L1/L3: replace 20% of tokens
    )
    
    print(f"✓ Generated {len(attacked_dataset)} attack variants")
    
    # ===== Evaluate on All Attack Levels =====
    print("\n" + "=" * 70)
    print("EVALUATING ROBUSTNESS")
    print("=" * 70)
    
    results = []
    
    for attack_type, (texts, labels) in attacked_dataset.items():
        print(f"\nEvaluating on {attack_type.upper()} attack...")
        
        # Vectorize
        if attack_type == 'clean':
            X_test_vec = vectorizer.transform(texts)
        else:
            X_test_vec = vectorizer.transform(texts)
        
        # Predict
        y_pred = clf.predict(X_test_vec)
        
        # Compute metrics
        acc = accuracy_score(labels, y_pred)
        f1 = f1_score(labels, y_pred, average='weighted', zero_division=0)
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        
        results.append({
            'attack_type': attack_type,
            'accuracy': acc,
            'f1_weighted': f1,
            'num_samples': len(texts),
        })
    
    # ===== Compute Robustness Metrics =====
    print("\n" + "=" * 70)
    print("ROBUSTNESS METRICS")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    clean_acc = results_df[results_df['attack_type'] == 'clean']['accuracy'].values[0]
    
    robustness_df = results_df.copy()
    robustness_df['accuracy_drop'] = clean_acc - robustness_df['accuracy']
    robustness_df['accuracy_drop_pct'] = (robustness_df['accuracy_drop'] / clean_acc * 100).round(2)
    
    print(f"\n{robustness_df[['attack_type', 'accuracy', 'accuracy_drop', 'accuracy_drop_pct']].to_string(index=False)}")
    
    # ===== Save Results =====
    results_output = config.TABLES_DIR / "transliteration_robustness.csv"
    robustness_df.to_csv(results_output, index=False)
    
    print(f"\n✓ Transliteration robustness results saved:")
    print(f"  {results_output}")
    
    # ===== Summary Statistics =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Clean accuracy: {clean_acc:.4f}")
    print(f"L1 accuracy drop: {robustness_df[robustness_df['attack_type'] == 'l1']['accuracy_drop'].values[0]:.4f} ({robustness_df[robustness_df['attack_type'] == 'l1']['accuracy_drop_pct'].values[0]:.2f}%)")
    print(f"L2 accuracy drop: {robustness_df[robustness_df['attack_type'] == 'l2']['accuracy_drop'].values[0]:.4f} ({robustness_df[robustness_df['attack_type'] == 'l2']['accuracy_drop_pct'].values[0]:.2f}%)")
    print(f"L3 accuracy drop: {robustness_df[robustness_df['attack_type'] == 'l3']['accuracy_drop'].values[0]:.4f} ({robustness_df[robustness_df['attack_type'] == 'l3']['accuracy_drop_pct'].values[0]:.2f}%)")
    
    print(f"\n✓ Axis C evaluation complete!")
    print(f"  Next: Run script 03_run_fairness_audit.py (Axis B)")


if __name__ == "__main__":
    main()
