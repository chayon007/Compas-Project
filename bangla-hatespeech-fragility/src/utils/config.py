"""Global configuration management."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Global configuration for experiments."""
    
    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.ROOT_DIR / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.INTERIM_DATA_DIR = self.DATA_DIR / "interim"
        self.RESULTS_DIR = self.ROOT_DIR / "results"
        self.TABLES_DIR = self.RESULTS_DIR / "tables"
        self.FIGURES_DIR = self.RESULTS_DIR / "figures"
        self.LOGS_DIR = self.RESULTS_DIR / "logs"
        self.CONFIGS_DIR = self.ROOT_DIR / "configs"
        
        # Create directories if they don't exist
        for d in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.INTERIM_DATA_DIR,
                  self.TABLES_DIR, self.FIGURES_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Global settings
        self.RANDOM_SEED = 42
        self.DEVICE = "cuda" if self._check_cuda() else "cpu"
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML config file."""
        yaml_path = self.CONFIGS_DIR / filename
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def __repr__(self) -> str:
        return (
            f"Config(ROOT_DIR={self.ROOT_DIR}, DEVICE={self.DEVICE}, "
            f"RANDOM_SEED={self.RANDOM_SEED})"
        )


# Global instance
config = Config()
