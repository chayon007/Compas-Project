"""Model training and baseline implementations."""

from .baseline import train_tfidf_baseline, train_transformer_baseline
from .trainer import TrainerConfig, TransformerTrainer

__all__ = ["train_tfidf_baseline", "train_transformer_baseline", "TrainerConfig", "TransformerTrainer"]
