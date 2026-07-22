#!/usr/bin/env python3
"""
Generate publication-quality visualizations from result CSVs
Run: python generate_visualizations_fixed.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
project_root = Path(__file__).parent
results_dir = project_root / "results"
tables_dir = results_dir / "tables"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Style settings
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

print("="*70)
print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
print("="*70)

# Figure 1: Baseline Performance
print("\n[1/4] Baseline Performance...")
try:
    df_baseline = pd.read_csv(tables_dir / "baseline_tfidf_metrics.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    models = df_baseline['model'].tolist()
    accuracy = df_baseline['accuracy'].tolist()
    f1 = df_baseline['macro_f1'].tolist()
    
    x = range(len(models))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], accuracy, width, label='Accuracy', 
                   alpha=0.8, color='#1f77b4')
    bars2 = ax.bar([i + width/2 for i in x], f1, width, label='Macro F1', 
                   alpha=0.8, color='#ff7f0e')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Model Performance\n(TF-IDF + Logistic Regression)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0.70, 0.85])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "01_baseline_performance.png", dpi=300, bbox_inches='tight')
    print("   [OK] Saved: 01_baseline_performance.png")
    plt.close()
except Exception as e:
    print(f"   [ERROR] {e}")

# Figure 2: Transliteration Robustness
print("\n[2/4] Transliteration Robustness...")
try:
    df_trans = pd.read_csv(tables_dir / "transliteration_robustness.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    attacks = df_trans['attack_type'].tolist()
    accuracy = df_trans['accuracy'].tolist()
    drop_pct = df_trans['accuracy_drop_pct'].tolist()
    
    # Color coding
    colors = ['#2ca02c' if a == 'clean' else '#ff7f0e' if 'l1' in str(a) 
              else '#d62728' if 'l2' in str(a) else '#9467bd' for a in attacks]
    
    # Left: Accuracy by attack
    bars1 = ax1.bar(attacks, accuracy, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy Under Transliteration Attacks', fontsize=12, fontweight='bold')
    ax1.set_ylim([0.74, 0.82])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right: Accuracy drop
    bars2 = ax2.bar(attacks, drop_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy Drop (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy Drop vs Clean Baseline', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels and highlight critical
    for i, (bar, drop) in enumerate(zip(bars2, drop_pct)):
        height = bar.get_height()
        label = f'{drop:.2f}%'
        if drop > 3:  # Highlight critical
            label += ' [CRITICAL]'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "02_transliteration_robustness.png", dpi=300, bbox_inches='tight')
    print("   [OK] Saved: 02_transliteration_robustness.png")
    plt.close()
except Exception as e:
    print(f"   [ERROR] {e}")

# Figure 3: Dialect Fairness
print("\n[3/4] Dialect Fairness...")
try:
    df_dialect = pd.read_csv(tables_dir / "dialect_fairness_metrics.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    dialects = df_dialect['dialect'].tolist()
    colors_dialect = ['#2ca02c', '#ff7f0e']  # Standard: green, Mixed: orange
    
    # F1 Score by dialect
    ax = axes[0, 0]
    f1_scores = df_dialect['f1'].tolist()
    bars = ax.bar(dialects, f1_scores, color=colors_dialect, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('F1 Score by Dialect', fontweight='bold', fontsize=11)
    ax.set_ylim([0.6, 0.8])
    ax.grid(axis='y', alpha=0.3)
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Precision vs Recall
    ax = axes[0, 1]
    precision = df_dialect['precision'].tolist()
    recall = df_dialect['recall'].tolist()
    x = range(len(dialects))
    width = 0.35
    ax.bar([i - width/2 for i in x], precision, width, label='Precision', 
           alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=1)
    ax.bar([i + width/2 for i in x], recall, width, label='Recall', 
           alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=1)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Precision vs Recall by Dialect', fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(dialects)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # FPR by dialect (FAIRNESS ISSUE)
    ax = axes[1, 0]
    fpr = df_dialect['fpr'].tolist()
    bars = ax.bar(dialects, fpr, color=colors_dialect, alpha=0.7, edgecolor='red', linewidth=2)
    ax.set_ylabel('False Positive Rate', fontweight='bold')
    ax.set_title('FPR by Dialect\n(Red Border = Fairness Issue)', fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, fp in zip(bars, fpr):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{fp:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # FNR by dialect
    ax = axes[1, 1]
    fnr = df_dialect['fnr'].tolist()
    bars = ax.bar(dialects, fnr, color=colors_dialect, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('False Negative Rate', fontweight='bold')
    ax.set_title('FNR by Dialect', fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, fn in zip(bars, fnr):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{fn:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "03_dialect_fairness.png", dpi=300, bbox_inches='tight')
    print("   [OK] Saved: 03_dialect_fairness.png")
    plt.close()
except Exception as e:
    print(f"   [ERROR] {e}")

# Figure 4: Fairness Gaps Summary
print("\n[4/4] Fairness Gaps Summary...")
try:
    df_gaps = pd.read_csv(tables_dir / "fairness_gaps.csv")
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    metrics = ['Max F1 Gap', 'Max FPR Gap\n(CRITICAL)', 'Max FNR Gap', 'Equalized\nOdds Diff']
    values = [
        df_gaps['max_f1_gap'].values[0],
        df_gaps['max_fpr_gap'].values[0],
        df_gaps['max_fnr_gap'].values[0],
        df_gaps['equalized_odds_diff'].values[0]
    ]
    
    # Color coding: red for violations (>0.1), yellow for warnings (0.05-0.1), green for acceptable
    colors_bars = []
    for v in values:
        if v > 0.15:
            colors_bars.append('#d62728')  # Red - CRITICAL
        elif v > 0.1:
            colors_bars.append('#ff7f0e')  # Orange - WARNING
        else:
            colors_bars.append('#2ca02c')  # Green - OK
    
    bars = ax.barh(metrics, values, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Gap Value', fontweight='bold', fontsize=12)
    ax.set_title('Fairness Violation Metrics\n(RED = Critical, ORANGE = Warning, GREEN = Acceptable)', 
                fontsize=13, fontweight='bold')
    ax.set_xlim([0, max(values) * 1.2])
    ax.grid(axis='x', alpha=0.3)
    
    # Add threshold line
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Fairness Threshold (0.1)')
    ax.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "04_fairness_gaps.png", dpi=300, bbox_inches='tight')
    print("   [OK] Saved: 04_fairness_gaps.png")
    plt.close()
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "="*70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nOutput location: {figures_dir}/")
print("\nGenerated files:")
print("  1. 01_baseline_performance.png - Bar chart of baseline metrics")
print("  2. 02_transliteration_robustness.png - Attack robustness analysis")
print("  3. 03_dialect_fairness.png - Per-dialect performance matrix")
print("  4. 04_fairness_gaps.png - Fairness violations summary")
print("\nReady for insertion into IEEE paper!")
