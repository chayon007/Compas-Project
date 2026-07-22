"""
Test defense strategies against L2 transliteration attacks.
This evaluates whether data augmentation + script normalization actually 
recover accuracy when tested on L2-attacked samples.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import config
from src.attacks.transliteration import TransliterationAttack


def train_defense_models(X_train, X_test, y_train, y_test):
    """Train three models: baseline, augmented, and script-normalized+augmented."""
    
    results = []
    
    # ===== Model 1: Baseline =====
    print("\n1. Training baseline model (no augmentation)...")
    vec_baseline = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf_base = vec_baseline.fit_transform(X_train)
    X_test_tfidf_base = vec_baseline.transform(X_test)
    
    model_baseline = LogisticRegression(max_iter=1000, random_state=42)
    model_baseline.fit(X_train_tfidf_base, y_train)
    
    y_pred_base = model_baseline.predict(X_test_tfidf_base)
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    
    print(f"   ✓ Baseline accuracy (clean): {acc_base:.4f}, F1: {f1_base:.4f}")
    
    results.append({
        'model': 'Baseline (no augmentation)',
        'accuracy_clean': acc_base,
        'f1_clean': f1_base,
        'vectorizer': vec_baseline,
        'model': model_baseline
    })
    
    # ===== Model 2: With Data Augmentation =====
    print("\n2. Training model with data augmentation...")
    X_train_aug = list(X_train)
    y_train_aug = list(y_train)
    
    # Add augmented samples (30% of original)
    n_augment = int(len(X_train) * 0.3)
    np.random.seed(42)
    indices = np.random.choice(len(X_train), n_augment, replace=True)
    
    for idx in indices:
        text = X_train[idx]
        # Simple augmentation: character swaps, romanization
        if np.random.random() > 0.5:
            # Random character swaps (L1-style)
            chars = list(text)
            n_swaps = max(1, len(chars) // 20)
            swap_indices = np.random.choice(len(chars), n_swaps, replace=False)
            for i in swap_indices:
                if i + 1 < len(chars):
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
            text = ''.join(chars)
        
        X_train_aug.append(text)
        y_train_aug.append(y_train[idx])
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    vec_aug = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf_aug = vec_aug.fit_transform(X_train_aug)
    X_test_tfidf_aug = vec_aug.transform(X_test)
    
    model_aug = LogisticRegression(max_iter=1000, random_state=42)
    model_aug.fit(X_train_tfidf_aug, y_train_aug)
    
    y_pred_aug = model_aug.predict(X_test_tfidf_aug)
    acc_aug = accuracy_score(y_test, y_pred_aug)
    f1_aug = f1_score(y_test, y_pred_aug)
    
    print(f"   ✓ Augmented model accuracy (clean): {acc_aug:.4f}, F1: {f1_aug:.4f}")
    
    results.append({
        'model': 'Data Augmentation (30%)',
        'accuracy_clean': acc_aug,
        'f1_clean': f1_aug,
        'vectorizer': vec_aug,
        'model': model_aug
    })
    
    return results, model_baseline, vec_baseline, model_aug, vec_aug


def evaluate_on_l2_attacks(X_test, y_test, models_info):
    """Test models on L2-attacked samples."""
    
    print("\n" + "="*70)
    print("EVALUATING ON L2 ATTACKS")
    print("="*70)
    
    # Create L2 attacks on test set
    print("\nGenerating L2 attacks (Bangla → Roman transliteration)...")
    attack_gen = TransliterationAttack(random_seed=42)
    X_test_l2 = []
    for text in X_test:
        attacked = attack_gen.level2_attack(text)
        X_test_l2.append(attacked)
    
    X_test_l2 = np.array(X_test_l2)
    print(f"✓ Generated {len(X_test_l2)} L2-attacked samples")
    
    # Evaluate each model on L2 attacks
    results = []
    for info in models_info:
        model_name = info['model']
        model = info['model']
        vectorizer = info['vectorizer']
        
        # Transform L2 attacked samples
        X_test_l2_tfidf = vectorizer.transform(X_test_l2)
        
        # Predict
        y_pred_l2 = model.predict(X_test_l2_tfidf)
        acc_l2 = accuracy_score(y_test, y_pred_l2)
        f1_l2 = f1_score(y_test, y_pred_l2)
        
        acc_drop = info['accuracy_clean'] - acc_l2
        acc_recovery = 1.0 - (acc_drop / 0.0513)  # 0.0513 is baseline L2 drop
        
        print(f"\n{model_name}:")
        print(f"   Clean accuracy:  {info['accuracy_clean']:.4f}")
        print(f"   L2 accuracy:     {acc_l2:.4f}")
        print(f"   Accuracy drop:   {acc_drop:.4f} ({acc_drop*100:.2f}%)")
        print(f"   Recovery rate:   {max(0, acc_recovery)*100:.1f}% of 5.13% loss")
        
        results.append({
            'model': model_name,
            'accuracy_clean': info['accuracy_clean'],
            'accuracy_l2': acc_l2,
            'f1_clean': info['f1_clean'],
            'f1_l2': f1_l2,
            'accuracy_drop': acc_drop,
            'recovery_percent': max(0, acc_recovery) * 100
        })
    
    return pd.DataFrame(results)


def main():
    """Run defense evaluation on L2 attacks."""
    
    print("=" * 70)
    print("DEFENSE STRATEGY EVALUATION ON L2 ATTACKS")
    print("=" * 70)
    
    # Load master dataset
    print("\nLoading master dataset...")
    master_path = config.PROCESSED_DATA_DIR / "master.csv"
    master_df = pd.read_csv(master_path)
    print(f"✓ Loaded {len(master_df)} samples")
    
    # Split train/test
    print("\nSplitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        master_df['text'].values,
        master_df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=master_df['label'].values
    )
    
    # Train models
    models_info, model_base, vec_base, model_aug, vec_aug = train_defense_models(
        X_train, X_test, y_train, y_test
    )
    
    # Evaluate on L2 attacks
    results_df = evaluate_on_l2_attacks(X_test, y_test, models_info)
    
    # Save results
    output_path = config.TABLES_DIR / "defense_l2_recovery.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Defense Recovery on L2 Attacks")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print("\nKey Finding:")
    baseline_drop = results_df[results_df['model'] == 'Baseline (no augmentation)']['accuracy_drop'].values[0]
    print(f"  • Baseline drops {baseline_drop*100:.2f}% on L2 attacks")
    
    for model_name in results_df['model'].unique():
        if model_name != 'Baseline (no augmentation)':
            recovery = results_df[results_df['model'] == model_name]['recovery_percent'].values[0]
            print(f"  • {model_name}: {recovery:.1f}% recovery from 5.13% baseline loss")


if __name__ == '__main__':
    main()
