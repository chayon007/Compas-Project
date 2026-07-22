"""
M2: Train baseline models (TF-IDF + LR, BanglaBERT, XLM-R).

This script:
1. Loads master dataset
2. Splits into train/test
3. Trains TF-IDF + Logistic Regression baseline
4. (Optional) Trains BanglaBERT/XLM-R transformer baseline
5. Saves results to CSV for paper
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import TextPreprocessor
from src.models.baseline import train_tfidf_baseline, train_transformer_baseline
from src.utils.config import config
from src.utils.metrics import save_metrics_to_csv, calculate_metrics


def main():
    """Run baseline training pipeline."""
    
    print("=" * 70)
    print("M2: TRAIN BASELINE MODELS")
    print("=" * 70)
    
    # Check if master dataset exists
    master_path = config.PROCESSED_DATA_DIR / "master.csv"
    if not master_path.exists():
        print(f"\n❌ Master dataset not found at {master_path}")
        print("Please run script 00_prepare_data.py first!")
        return
    
    # Load master dataset
    print(f"\nLoading master dataset...")
    master_df = pd.read_csv(master_path)
    print(f"✓ Loaded {len(master_df)} samples")
    
    # Check label distribution
    print(f"\nLabel distribution:")
    print(master_df['label'].value_counts())
    print(f"Class imbalance ratio: {master_df['label'].value_counts()[1] / master_df['label'].value_counts()[0]:.3f}")
    
    # Split train/test
    print(f"\nSplitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        master_df['text_norm'].values,
        master_df['label'].values,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=master_df['label'].values
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Train label dist: {np.bincount(y_train)}")
    print(f"  Test label dist: {np.bincount(y_test)}")
    
    # ===== Baseline 1: TF-IDF + Logistic Regression =====
    print("\n" + "=" * 70)
    print("BASELINE 1: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)
    
    tfidf_results = train_tfidf_baseline(
        X_train, X_test, y_train, y_test,
        max_features=30000,
        ngram_range=(1, 2)
    )
    
    # Save TF-IDF model and vectorizer
    tfidf_model_dir = config.RESULTS_DIR / "models" / "tfidf_baseline"
    tfidf_model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(tfidf_model_dir / "model.pkl", "wb") as f:
        pickle.dump(tfidf_results['model'], f)
    with open(tfidf_model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_results['vectorizer'], f)
    
    print(f"\n✓ TF-IDF model saved to {tfidf_model_dir}")
    
    # Save TF-IDF results
    tfidf_metrics_df = pd.DataFrame([{
        'model': 'tfidf_lr',
        'accuracy': tfidf_results['test_accuracy'],
        'macro_f1': tfidf_results['test_report'].split('\n')[-3].split()[-2],
    }])
    tfidf_metrics_df.to_csv(config.TABLES_DIR / "baseline_tfidf_metrics.csv", index=False)
    
    # ===== Baseline 2: BanglaBERT (Optional - skip if no GPU) =====
    print("\n" + "=" * 70)
    print("BASELINE 2: BANGLABE RT (Transformer)")
    print("=" * 70)
    print("⚠️  This requires GPU and will download ~1.2GB model")
    print("   Skipping for now - uncomment in script to enable")
    print("=" * 70)
    
    # Uncomment to train transformer (requires GPU):
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        transformer_results = train_transformer_baseline(
            X_train.tolist(),
            y_train.tolist(),
            X_test.tolist(),
            y_test.tolist(),
            model_name="google-bert/bert-base-multilingual-uncased",
            output_dir=config.RESULTS_DIR / "models" / "banglbert_baseline",
            num_epochs=3,
            batch_size=16,
            device=device
        )
        print("✓ BanglaBERT training complete")
    except Exception as e:
        print(f"❌ Error training transformer: {e}")
    """
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  Metrics: {config.TABLES_DIR / 'baseline_tfidf_metrics.csv'}")
    print(f"  Models: {config.RESULTS_DIR / 'models'}")
    
    print(f"\n✓ Baseline models trained successfully!")
    print(f"  Next: Run script 02_run_transliteration.py (Axis C)")


if __name__ == "__main__":
    main()
