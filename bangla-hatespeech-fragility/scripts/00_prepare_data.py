"""
M1: Prepare unified master dataset from all raw datasets.

This script:
1. Defines expected dataset locations
2. Loads all datasets using DatasetLoader
3. Unifies labels to binary format
4. Detects script types (Bangla/Romanized/Mixed)
5. Saves master CSV for downstream experiments
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DatasetLoader
from src.data.preprocessor import TextPreprocessor
from src.utils.config import config


def main():
    """Run data preparation pipeline."""
    
    print("=" * 70)
    print("M1: PREPARE UNIFIED MASTER DATASET")
    print("=" * 70)
    
    # Define where to find raw datasets
    # Updated to handle directory structures with train/test splits
    dataset_paths = {
        'karim_2020': config.RAW_DATA_DIR / 'Bengali_hate_speech_dataset',        # Train/test split
        'bidwesh': config.RAW_DATA_DIR / 'BIDWESH A Bangla Regional Based Hate Speech Detect',  # Nested dirs
        'boishommo': config.RAW_DATA_DIR / 'BOISHOMMO A Standardized Multi-Label Bangla Hate S',  # May not exist
        'banth': config.RAW_DATA_DIR / 'BanTH',                                    # Has full_with_stats.csv
    }
    
    print(f"\nExpected dataset locations:")
    print("-" * 70)
    for name, path in dataset_paths.items():
        print(f"  {name}: {path}")
        print(f"    exists: {path.exists()}")
    
    print("\n" + "=" * 70)
    print("INSTRUCTIONS: Download Datasets First")
    print("=" * 70)
    print("""
To proceed, please download these datasets and place them in:
  {}

Datasets:
1. BIDWESH (2025) - Bangla Regional Hate Speech
   - Mendeley Data: https://data.mendeley.com/
   - ~9,183 samples, Bangla script, dialectal variations
   - Save as: bidwesh.csv

2. BOISHOMMO (2025) - Multi-label Bangla Hate Speech
   - Mendeley Data or ACL Anthology
   - ~2,499 samples, multi-label, newest data
   - Save as: boishommo.csv

3. BanTH (2024) - Transliterated Bangla Hate Speech
   - HuggingFace or ACL Anthology
   - ~37,300 samples, Romanized script
   - Save as: banth.csv

4. Karim et al. (2020) - BD-SHS Dataset
   - UCI ML Repository
   - ~10,000 samples, oldest temporal phase
   - Save as: karim_2020.csv

Once downloaded, uncomment the dataset_paths lines above and re-run this script.
    """.format(config.RAW_DATA_DIR))
    
    # Check if any datasets are available
    available_datasets = [name for name, path in dataset_paths.items() if path.exists()]
    
    if not available_datasets:
        print("\n❌ No datasets found! Please download them first.")
        return
    
    print(f"\n✓ Found {len(available_datasets)} dataset(s): {available_datasets}")
    
    # Create unified dataset
    loader = DatasetLoader(config.RAW_DATA_DIR)
    master_df = loader.create_master_dataset(
        dataset_paths=dataset_paths,
        output_path=config.PROCESSED_DATA_DIR / "master.csv"
    )
    
    print("\n" + "=" * 70)
    print("MASTER DATASET SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(master_df)}")
    print(f"\nLabel distribution:")
    print(master_df['label'].value_counts())
    print(f"\nDatasets included:")
    print(master_df['dataset'].value_counts())
    print(f"\nScript types:")
    print(master_df['script_type'].value_counts())
    print(f"\nDialect groups:")
    print(master_df['dialect_group'].value_counts())
    print(f"\nTask phases (temporal):")
    print(master_df['task_phase'].value_counts())
    
    print("\n✓ Master dataset created successfully!")
    print(f"  Saved to: {config.PROCESSED_DATA_DIR / 'master.csv'}")
    
    # Save some basic statistics
    stats = {
        'total_samples': len(master_df),
        'hate_samples': (master_df['label'] == 1).sum(),
        'non_hate_samples': (master_df['label'] == 0).sum(),
        'unique_dialects': master_df['dialect_group'].nunique(),
        'script_types': master_df['script_type'].unique().tolist(),
    }
    
    print("\n" + "=" * 70)
    print("DATASET METADATA")
    print("=" * 70)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
