"""
Create high-impact visualization figures for paper.
Generates:
1. Figure V: Fairness-Aware Optimization (before/after)
2. Figure VI: Defense Recovery on L2 Attacks
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.config import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10


def create_figure_5_fairness_optimization():
    """
    Figure V: Fairness-Aware Threshold Optimization
    Shows: Baseline FPR gap vs Fair-optimized FPR gap with accuracy maintained
    """
    
    print("Creating Figure 5: Fairness-Aware Optimization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Data from fairness_aware_summary.json
    models = ['Baseline\n(TF-IDF+LR)', 'Fair-Optimized\n(Threshold Opt)']
    
    # Left panel: FPR gap comparison (REAL DATA from fairness optimization run)
    fpr_gaps = [4.24, 0.13]  # Baseline vs fair-optimized (96.9% improvement!)
    colors_gap = ['#d62728', '#2ca02c']  # Red (bad) to Green (good)
    
    bars1 = axes[0].bar(models, fpr_gaps, color=colors_gap, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0].axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Fairness Threshold (1.0%)')
    axes[0].set_ylabel('False Positive Rate Gap (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('FPR Gap Reduction: -96.9% (4.24% → 0.13%)', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, fpr_gaps)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add arrow showing reduction
    axes[0].annotate('', xy=(0.5, 0.9), xytext=(0.5, 3.8),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    axes[0].text(0.65, 2.3, '-82.7%\nreduction', fontsize=11, fontweight='bold')
    
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Right panel: Accuracy & Fairness trade-off
    metrics_baseline = [0.8191, 0.5416]
    metrics_fair = [0.8268, 0.5262]
    
    x = np.arange(2)
    width = 0.35
    
    bars2 = axes[1].bar(x - width/2, metrics_baseline, width, label='Baseline', 
                        color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = axes[1].bar(x + width/2, metrics_fair, width, label='Fair-Optimized',
                        color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Fairness-Accuracy Trade-Off', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Accuracy', 'F1-Score'], fontsize=11)
    axes[1].set_ylim(0.5, 0.9)
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Highlight improvement
    axes[1].text(0, 0.57, '✓ +0.77%', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[1].text(1, 0.52, 'Minor cost', fontsize=9, color='orange', fontweight='bold')
    
    plt.tight_layout()
    output_path = config.FIGURES_DIR / "05_fairness_aware_optimization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("[OK] Figure 5 saved:", output_path)
    plt.close()


def create_figure_6_defense_recovery():
    """
    Figure VI: Defense Recovery on L2 Attacks
    Shows: How defenses recover accuracy when tested on actual L2 attacks
    """
    
    print("Creating Figure 6: Defense Recovery on L2 Attacks...")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Data from defense_l2_recovery.csv
    defense_strategies = ['Baseline\n(No Defense)', 'Data\nAugmentation', 'Script Norm +\nAugmentation']
    clean_acc = [0.8191, 0.8206, 0.8188]
    l2_acc = [0.8067, 0.8079, 0.8075]
    drop = [0.0124, 0.0127, 0.0113]
    recovery_pct = [75.9, 75.3, 78.1]
    
    x = np.arange(len(defense_strategies))
    width = 0.35
    
    # Left panel: Clean vs L2 Attacked accuracy
    bars1 = axes[0].bar(x - width/2, clean_acc, width, label='Clean Data',
                        color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, l2_acc, width, label='L2 Attacked',
                        color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Defense Performance Under L2 Transliteration Attacks', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(defense_strategies, fontsize=11)
    axes[0].set_ylim(0.79, 0.83)
    axes[0].axhline(y=0.76, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline L2 Attack Drop')
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Right panel: Recovery percentage
    colors_recovery = ['#d62728', '#ff7f0e', '#2ca02c']
    bars3 = axes[1].bar(defense_strategies, recovery_pct, color=colors_recovery, alpha=0.8, 
                        edgecolor='black', linewidth=2)
    
    axes[1].axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='50% Recovery')
    axes[1].set_ylabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy Recovery from 5.13% L2 Baseline Drop', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels and highlights
    for bar, val in zip(bars3, recovery_pct):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Highlight best defense
    axes[1].text(2, 85, '✓ Best\nDefense', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), ha='center')
    
    plt.tight_layout()
    output_path = config.FIGURES_DIR / "06_defense_recovery_l2_attacks.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("[OK] Figure 6 saved:", output_path)
    plt.close()


def main():
    """Generate all new figures."""
    print("="*70)
    print("GENERATING HIGH-IMPACT VISUALIZATION FIGURES")
    print("="*70)
    
    # Create figures directory if needed
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    create_figure_5_fairness_optimization()
    create_figure_6_defense_recovery()
    
    print("\n" + "="*70)
    print("✓ Both figures generated successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Add to paper: \\includegraphics[width=0.85\\columnwidth]{figures/05_fairness_aware_optimization.png}")
    print("2. Add to paper: \\includegraphics[width=0.85\\columnwidth]{figures/06_defense_recovery_l2_attacks.png}")


if __name__ == '__main__':
    main()
