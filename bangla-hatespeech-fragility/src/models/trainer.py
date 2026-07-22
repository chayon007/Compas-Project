"""Trainer utility classes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainerConfig:
    """Configuration for model training."""
    
    model_name: str = "google-bert/bert-base-multilingual-uncased"
    output_dir: Path = Path("models/checkpoints")
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cuda"
    max_length: int = 512
    
    def __post_init__(self):
        """Ensure output_dir is a Path."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


class TransformerTrainer:
    """Simple wrapper for transformer training."""
    
    def __init__(self, config: TrainerConfig):
        """Initialize trainer with config."""
        self.config = config
