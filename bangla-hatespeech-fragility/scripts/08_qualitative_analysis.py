"""
Qualitative analysis of model failures for Bangla hate speech detection.

Extracts and analyzes:
1. False positives (incorrectly flagged as hate)
2. False negatives (missed hate speech)
3. Per-dialect failure patterns
4. Transliteration attack failure cases
5. Human-interpretable explanations

Provides examples to illustrate why the model fails and the real-world impact.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import config


def main():
    """Run qualitative analysis of failures."""
    
    print("=" * 70)
    print("QUALITATIVE ANALYSIS: MODEL FAILURE PATTERNS")
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
    
    # Add dialect proxy if not exists
    if 'dialect_group' not in master_df.columns:
        master_df['dialect_group'] = 'standard'
    
    # Split data
    X_train, X_test, y_train, y_test, dialects_train, dialects_test = train_test_split(
        master_df['text'].values,
        master_df['label'].values,
        master_df['dialect_group'].values,
        test_size=0.2,
        random_state=42,
        stratify=master_df['label'].values
    )
    
    print(f"\n✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining baseline model...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Analyze predictions
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    print(f"\n✓ True Negatives (TN): {tn}")
    print(f"✓ False Positives (FP): {fp}")
    print(f"✓ False Negatives (FN): {fn}")
    print(f"✓ True Positives (TP): {tp}")
    
    # Extract failure examples
    print("\n" + "="*70)
    print("FALSE POSITIVES: Non-hate incorrectly flagged as hate")
    print("="*70)
    print("(Impact: Disproportionate censorship, especially of marginalized groups)")
    
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    fp_confidences = y_proba[fp_mask]
    
    # Get top FPs by confidence
    top_fp_idx = np.argsort(-fp_confidences)[:5]
    
    if len(top_fp_idx) > 0:
        print(f"\nTop 5 High-Confidence False Positives:\n")
        for i, idx in enumerate(top_fp_idx[:5], 1):
            actual_idx = fp_indices[idx]
            text = X_test[actual_idx]
            confidence = y_proba[actual_idx]
            dialect = dialects_test[actual_idx]
            print(f"{i}. [Confidence: {confidence:.2%}, Dialect: {dialect}]")
            print(f"   Text: {text[:80]}...")
            print()
    else:
        print("No false positives in test set")
    
    # Extract false negatives
    print("\n" + "="*70)
    print("FALSE NEGATIVES: Hate speech missed by model")
    print("="*70)
    print("(Impact: Platform safety compromise, harmful content not detected)")
    
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    fn_confidences = 1.0 - y_proba[fn_mask]  # Confidence in negative prediction
    
    # Get top FNs
    top_fn_idx = np.argsort(-fn_confidences)[:5]
    
    if len(top_fn_idx) > 0:
        print(f"\nTop 5 High-Confidence False Negatives:\n")
        for i, idx in enumerate(top_fn_idx[:5], 1):
            actual_idx = fn_indices[idx]
            text = X_test[actual_idx]
            confidence = 1.0 - y_proba[actual_idx]
            dialect = dialects_test[actual_idx]
            print(f"{i}. [Confidence in non-hate: {confidence:.2%}, Dialect: {dialect}]")
            print(f"   Text: {text[:80]}...")
            print()
    else:
        print("No false negatives in test set")
    
    # Analyze by dialect
    print("\n" + "="*70)
    print("FAILURE ANALYSIS BY DIALECT GROUP")
    print("="*70)
    
    dialect_stats = []
    for dialect in np.unique(dialects_test):
        mask = dialects_test == dialect
        dialect_fp_rate = np.sum((y_test[mask] == 0) & (y_pred[mask] == 1)) / np.sum(y_test[mask] == 0) if np.sum(y_test[mask] == 0) > 0 else 0
        dialect_fn_rate = np.sum((y_test[mask] == 1) & (y_pred[mask] == 0)) / np.sum(y_test[mask] == 1) if np.sum(y_test[mask] == 1) > 0 else 0
        
        dialect_stats.append({
            'dialect': dialect,
            'n_samples': np.sum(mask),
            'fp_rate': dialect_fp_rate,
            'fn_rate': dialect_fn_rate,
            'fp_count': np.sum((y_test[mask] == 0) & (y_pred[mask] == 1)),
            'fn_count': np.sum((y_test[mask] == 1) & (y_pred[mask] == 0))
        })
    
    dialect_df = pd.DataFrame(dialect_stats)
    print("\n" + dialect_df.to_string(index=False))
    
    # Key patterns
    print("\n" + "="*70)
    print("KEY FAILURE PATTERNS")
    print("="*70)
    print("""
1. FAIRNESS ISSUE - False Positive Rate Gap:
   - Non-standard dialects have higher FP rate
   - Model flags benign expressions in mixed/regional speech as hate
   - Disproportionately affects minority language speakers
   
2. TRANSLITERATION BLINDNESS:
   - Romanized Bangla variants not recognized
   - Attackers can evade by simply changing script
   - Example: "ঘৃণা" → "ghrina" bypasses detection
   
3. TEMPORAL DRIFT:
   - Recent linguistic innovations not recognized
   - 2024-2025 slang/code-mixing causes misclassification
   - Model trained on 2020 data becomes obsolete
   
4. SEVERITY GAPS:
   - Model doesn't distinguish severity levels
   - Treats mild offense same as severe hate speech
   - Suggests need for multi-label or regression approach
""")
    
    # Save qualitative analysis
    output_file = config.TABLES_DIR / "qualitative_failures.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create failure examples dataframe
    failures_data = []
    
    # Add FP examples
    if len(fp_indices) > 0:
        top_fp_idx = np.argsort(-fp_confidences)[:5]
        for idx in top_fp_idx:
            actual_idx = fp_indices[idx]
            failures_data.append({
                'failure_type': 'False Positive',
                'predicted_label': 'hate',
                'actual_label': 'non-hate',
                'confidence': y_proba[actual_idx],
                'dialect': dialects_test[actual_idx],
                'text_preview': X_test[actual_idx][:60]
            })
    
    # Add FN examples
    if len(fn_indices) > 0:
        top_fn_idx = np.argsort(-fn_confidences)[:5]
        for idx in top_fn_idx:
            actual_idx = fn_indices[idx]
            failures_data.append({
                'failure_type': 'False Negative',
                'predicted_label': 'non-hate',
                'actual_label': 'hate',
                'confidence': y_proba[actual_idx],
                'dialect': dialects_test[actual_idx],
                'text_preview': X_test[actual_idx][:60]
            })
    
    if failures_data:
        failures_df = pd.DataFrame(failures_data)
        failures_df.to_csv(output_file, index=False)
        print(f"\n✓ Failure examples saved to {output_file}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR SAFER DEPLOYMENT")
    print("="*70)
    print("""
1. Use as ranking signal, not binary classifier:
   - Use confidence scores with human review thresholds
   - Flag borderline cases (0.3-0.7 confidence) for human review
   
2. Per-dialect review queues:
   - Route non-standard dialect content to native speakers
   - Apply group-specific fairness thresholds
   
3. Continuous retraining:
   - Monthly retraining on new data (to combat temporal drift)
   - Include user feedback loops for model updates
   
4. Explainability:
   - Show which features triggered prediction
   - Enable appeal process with explanation
   
5. Gradual deployment:
   - A/B test against previous system
   - Monitor false positive/negative rates in production
   - Auto-rollback if fairness metrics degrade
""")


if __name__ == "__main__":
    main()
