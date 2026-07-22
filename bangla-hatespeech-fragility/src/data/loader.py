"""Data loading utilities for all Bangla hate speech datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from glob import glob


class DatasetLoader:
    """Load and unify Bangla hate speech datasets."""
    
    def __init__(self, raw_data_dir: Path):
        """
        Initialize loader.
        
        Args:
            raw_data_dir: Path to raw data directory
        """
        self.raw_data_dir = Path(raw_data_dir)
    
    def load_karim_2020(self, filepath: Path) -> pd.DataFrame:
        """Load Karim et al. 2020 (BD-SHS) dataset from train/test/validate split files."""
        dfs = []
        parent_dir = filepath.parent if filepath.is_file() else filepath
        
        # Load train, test, validate files if they exist
        for split_file in ['train.csv', 'test.csv', 'validate.csv']:
            split_path = parent_dir / split_file
            if split_path.exists():
                df = pd.read_csv(split_path)
                dfs.append(df)
        
        # If no split files, try loading single file
        if not dfs and filepath.exists():
            df = pd.read_csv(filepath)
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Rename columns to standard format (handle various possible names)
        df = df.rename(columns={
            'comment': 'text',
            'annotation': 'label',
            'text': 'text',
            'label': 'label',
            'sentence': 'text',
            'target': 'label'
        })
        
        df['dataset'] = 'karim_2020'
        df['year'] = 2020
        df['dialect_group'] = 'standard'
        return df[['text', 'label', 'dataset', 'year', 'dialect_group']]
    
    def load_bidwesh(self, filepath: Path) -> pd.DataFrame:
        """Load BIDWESH dataset from nested directory structure."""
        dfs = []
        parent_dir = filepath if filepath.is_dir() else filepath.parent
        
        # Search for BIDWESH Dataset.csv recursively
        import glob
        pattern = str(parent_dir / "**" / "BIDWESH Dataset.csv")
        found_files = glob.glob(pattern, recursive=True)
        
        if found_files:
            df = pd.read_csv(found_files[0])
            dfs.append(df)
        
        # If no Dataset.csv, look for any CSV file in the directory
        if not dfs:
            pattern = str(parent_dir / "**" / "*.csv")
            found_files = glob.glob(pattern, recursive=True)
            if found_files:
                df = pd.read_csv(found_files[0])
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        
        # BIDWESH has regional dialect columns: Chittagong, Noakhali, Barishal
        regional_cols = ['Chittagong', 'Noakhali', 'Barishal']
        available_cols = [col for col in regional_cols if col in df.columns]
        
        if available_cols:
            # Use first non-null regional text as primary text
            df['text'] = df[available_cols].bfill(axis=1).iloc[:, 0]
        else:
            # Fallback: try generic column mapping
            column_map = {}
            for col in df.columns:
                if col.lower() in ['text', 'sentence', 'comment', 'content']:
                    column_map[col] = 'text'
                elif col.lower() in ['label', 'annotation', 'class']:
                    column_map[col] = 'label'
            df = df.rename(columns=column_map)
        
        # Map label column - BIDWESH uses 'hate speech' column (0/1)
        if 'hate speech' in df.columns:
            df['label'] = df['hate speech']
        elif 'label' not in df.columns and 'target' in df.columns:
            # Convert target to binary (ind/grp/other -> need mapping)
            df['label'] = (df['target'] != 'other').astype(int)
        
        if 'text' not in df.columns or 'label' not in df.columns:
            return pd.DataFrame()
        
        df['dataset'] = 'bidwesh'
        df['year'] = 2025
        df['dialect_group'] = 'mixed'
        return df[['text', 'label', 'dataset', 'year', 'dialect_group']].dropna()
    
    def load_boishommo(self, filepath: Path) -> pd.DataFrame:
        """Load BOISHOMMO dataset from Excel files or train/test/validate splits."""
        dfs = []
        parent_dir = filepath if filepath.is_dir() else filepath.parent
        
        # First try to find Excel files recursively
        excel_files = list(parent_dir.rglob('*.xlsx'))
        if excel_files:
            for excel_file in excel_files:
                try:
                    # Skip .zip files
                    if excel_file.suffix != '.xlsx':
                        continue
                    # Try to read Excel file (may have multiple sheets)
                    xl_file = pd.ExcelFile(excel_file)
                    for sheet_name in xl_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        if len(df) > 0:  # Only add non-empty sheets
                            dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to read {excel_file}: {e}")
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                # Fallback: Try to find CSV files
                csv_files = list(parent_dir.rglob('*.csv'))
                if csv_files:
                    for csv_file in csv_files:
                        df = pd.read_csv(csv_file)
                        dfs.append(df)
                    df = pd.concat(dfs, ignore_index=True)
                else:
                    return pd.DataFrame()
        else:
            # Load train, test, validate files if they exist
            for split_file in ['train.csv', 'test.csv', 'validate.csv', 'val.csv']:
                split_path = parent_dir / split_file
                if split_path.exists():
                    df = pd.read_csv(split_path)
                    dfs.append(df)
            
            # If no split files, try loading single file
            if not dfs and filepath.exists() and filepath.is_file():
                df = pd.read_csv(filepath)
                dfs.append(df)
            
            if not dfs:
                return pd.DataFrame()
            
            df = pd.concat(dfs, ignore_index=True)
        
        # Rename columns to standard format
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['text', 'sentence', 'comment', 'content', 'tweet', 'post']:
                column_map[col] = 'text'
            elif col_lower in ['label', 'annotation', 'class', 'target', 'labels', 'hate', 'hate_speech']:
                column_map[col] = 'label'
        
        df = df.rename(columns=column_map)
        
        # Keep only text and label columns
        if 'text' in df.columns and 'label' in df.columns:
            df = df[['text', 'label']].dropna()
            df['dataset'] = 'boishommo'
            df['year'] = 2025
            df['dialect_group'] = 'standard'
            return df[['text', 'label', 'dataset', 'year', 'dialect_group']]
        else:
            self.logger.warning(f"BOISHOMMO: Could not find 'text' and 'label' columns. Found: {df.columns.tolist()}")
            return pd.DataFrame()
    
    def load_banth(self, filepath: Path) -> pd.DataFrame:
        """Load BanTH dataset (transliterated) from train/test/val or consolidated file."""
        parent_dir = filepath if filepath.is_dir() else filepath.parent
        dfs = []
        
        # Try to load consolidated file first
        full_file = parent_dir / 'full_with_stats.csv'
        if full_file.exists():
            df = pd.read_csv(full_file)
            dfs.append(df)
        else:
            # Load train, test, val files if they exist
            for split_file in ['train.csv', 'test.csv', 'val.csv', 'validate.csv']:
                split_path = parent_dir / split_file
                if split_path.exists():
                    df = pd.read_csv(split_path)
                    dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Rename columns to standard format
        column_map = {}
        for col in df.columns:
            if col.lower() in ['text', 'sentence', 'comment', 'content']:
                column_map[col] = 'text'
            elif col.lower() in ['label', 'annotation', 'class', 'target']:
                column_map[col] = 'label'
        
        df = df.rename(columns=column_map)
        
        df['dataset'] = 'banth'
        df['year'] = 2024
        df['dialect_group'] = 'standard'  # Transliteration is orthogonal to dialect
        return df[['text', 'label', 'dataset', 'year', 'dialect_group']]
    
    def binarize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert labels to binary format: 0 = non-hate, 1 = hate.
        
        Maps various label schemes to binary.
        """
        def binarize(x):
            if isinstance(x, str):
                x_lower = x.lower().strip()
                if x_lower in ['hate', 'abusive', 'offensive', 'hateful', '1']:
                    return 1
                else:
                    return 0
            else:
                return 1 if x == 1 else 0
        
        df['label'] = df['label'].apply(binarize)
        return df
    
    def create_master_dataset(self, dataset_paths: Dict[str, Path], 
                             output_path: Path) -> pd.DataFrame:
        """
        Load all datasets and create unified master CSV.
        
        Args:
            dataset_paths: Dict mapping dataset names to file paths
            output_path: Path to save master CSV
        
        Returns:
            Unified DataFrame
        """
        dfs = []
        
        print("Loading datasets...")
        for dataset_name, filepath in dataset_paths.items():
            if not filepath.exists():
                print(f"⚠️  {dataset_name} not found at {filepath}, skipping...")
                continue
            
            print(f"Loading {dataset_name}...")
            try:
                if dataset_name == 'karim_2020':
                    df = self.load_karim_2020(filepath)
                elif dataset_name == 'bidwesh':
                    df = self.load_bidwesh(filepath)
                elif dataset_name == 'boishommo':
                    df = self.load_boishommo(filepath)
                elif dataset_name == 'banth':
                    df = self.load_banth(filepath)
                else:
                    print(f"Unknown dataset: {dataset_name}")
                    continue
                
                # Binarize labels
                df = self.binarize_labels(df)
                
                # Add processed text column
                df['text_norm'] = df['text'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                
                dfs.append(df)
                print(f"  ✓ {dataset_name}: {len(df)} samples loaded")
            except Exception as e:
                print(f"  ✗ Error loading {dataset_name}: {e}")
        
        # Combine all datasets
        master_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates based on normalized text
        master_df = master_df.drop_duplicates(subset=['text_norm'])
        
        # Create task phase for continual learning (temporal order)
        master_df['task_phase'] = master_df['year'].apply(
            lambda x: 't1' if x < 2023 else ('t2' if x < 2024 else 't3')
        )
        
        # Add script type detection
        from .preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        master_df['script_type'] = master_df['text'].apply(preprocessor.get_script_type)
        
        # Reorder columns for clarity
        columns_order = ['text', 'text_norm', 'label', 'dataset', 'year', 
                        'dialect_group', 'script_type', 'task_phase']
        master_df = master_df[columns_order]
        
        # Save master dataset
        master_df.to_csv(output_path, index=False)
        print(f"\n✓ Master dataset saved to {output_path}")
        print(f"  Total samples: {len(master_df)}")
        print(f"  Label distribution:\n{master_df['label'].value_counts()}")
        print(f"  Datasets: {master_df['dataset'].unique().tolist()}")
        print(f"  Script types: {master_df['script_type'].unique().tolist()}")
        
        return master_df
