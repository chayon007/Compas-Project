"""Continual learning strategies for temporal robustness."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import torch


class ContinualLearningStrategy(ABC):
    """Abstract base class for continual learning strategies."""
    
    @abstractmethod
    def learn(self, task_id: int, X_train: np.ndarray, y_train: np.ndarray):
        """Learn on a new task."""
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions on test data."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get continual learning metrics."""
        pass


class SequentialFineTuning(ContinualLearningStrategy):
    """Sequential fine-tuning strategy (lower bound - maximum forgetting)."""
    
    def __init__(self):
        """Initialize strategy."""
        self.model = None
        self.task_accuracies = []
    
    def learn(self, task_id: int, X_train: np.ndarray, y_train: np.ndarray):
        """Learn on new task by fine-tuning on that task only."""
        # This is a placeholder - in practice, would fine-tune transformer
        pass
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """Return metrics."""
        return {
            'strategy': 'sequential_finetuning',
            'average_accuracy': np.mean(self.task_accuracies) if self.task_accuracies else 0,
        }


class ExperienceReplay(ContinualLearningStrategy):
    """Experience replay strategy (store samples from previous tasks)."""
    
    def __init__(self, buffer_size: int = 50, buffer_per_task: int = None):
        """
        Initialize replay strategy.
        
        Args:
            buffer_size: Max samples to store per task
            buffer_per_task: Samples per task (overrides buffer_size if set)
        """
        self.buffer_size = buffer_size
        self.buffer_per_task = buffer_per_task or buffer_size
        self.memory_buffer = []
        self.task_accuracies = []
    
    def learn(self, task_id: int, X_train: np.ndarray, y_train: np.ndarray):
        """Learn on new task + replay samples from previous tasks."""
        # Combine current task with replayed samples from memory
        pass
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """Return metrics."""
        return {
            'strategy': 'experience_replay',
            'average_accuracy': np.mean(self.task_accuracies) if self.task_accuracies else 0,
            'buffer_size': len(self.memory_buffer),
        }


class ElasticWeightConsolidation(ContinualLearningStrategy):
    """Elastic Weight Consolidation (EWC) strategy."""
    
    def __init__(self, ewc_lambda: float = 0.4):
        """
        Initialize EWC strategy.
        
        Args:
            ewc_lambda: Regularization weight for EWC penalty
        """
        self.ewc_lambda = ewc_lambda
        self.fisher_information = {}
        self.optimal_weights = {}
        self.task_accuracies = []
    
    def learn(self, task_id: int, X_train: np.ndarray, y_train: np.ndarray):
        """Learn on new task with EWC regularization."""
        # Compute Fisher information matrix for this task
        # Apply EWC penalty during training
        pass
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """Return metrics."""
        return {
            'strategy': 'ewc',
            'ewc_lambda': self.ewc_lambda,
            'average_accuracy': np.mean(self.task_accuracies) if self.task_accuracies else 0,
        }


def compute_continual_learning_metrics(
    all_accuracies: Dict[int, Dict[int, float]]
) -> Dict[str, float]:
    """
    Compute continual learning metrics from task-wise accuracies.
    
    Args:
        all_accuracies: Dict mapping task_id -> {eval_task_id: accuracy}
    
    Returns:
        Dictionary with CL metrics:
        - Average Accuracy (AA): Mean accuracy across all tasks
        - Backward Transfer (BWT): How much learning new tasks helps old tasks
        - Forward Transfer (FWT): How much old knowledge helps new tasks
        - Forgetting Measure (FM): How much old task performance drops
    """
    num_tasks = len(all_accuracies)
    
    # Average Accuracy
    all_acc = []
    for task_accs in all_accuracies.values():
        all_acc.extend(task_accs.values())
    aa = np.mean(all_acc) if all_acc else 0
    
    # Backward Transfer (BWT)
    # How much does performance on task i improve after learning tasks i+1, ..., T
    bwt = 0
    for i in range(num_tasks - 1):
        # Accuracy on task i after learning all subsequent tasks
        acc_after = all_accuracies[num_tasks - 1].get(i, 0)
        # Accuracy on task i right after learning it
        acc_right_after = all_accuracies[i].get(i, 0)
        bwt += (acc_after - acc_right_after)
    bwt /= max(1, num_tasks - 1)
    
    # Forward Transfer (FWT)
    # Performance on task i before learning it (using model from task i-1)
    fwt = 0
    for i in range(1, num_tasks):
        # Random baseline accuracy on task i (typically 0.5 for binary, 1/num_classes)
        acc_before = 0.5
        # Accuracy on task i after learning task i-1
        acc_after_prev = all_accuracies[i - 1].get(i, 0)
        fwt += (acc_after_prev - acc_before)
    fwt /= max(1, num_tasks - 1)
    
    # Forgetting Measure (FM)
    # Max performance on task i minus performance after learning all tasks
    fm = 0
    for i in range(num_tasks - 1):
        max_acc = max(all_accuracies[j].get(i, 0) for j in range(i, num_tasks))
        final_acc = all_accuracies[num_tasks - 1].get(i, 0)
        fm += (max_acc - final_acc)
    fm /= max(1, num_tasks - 1)
    
    return {
        'average_accuracy': aa,
        'backward_transfer': bwt,
        'forward_transfer': fwt,
        'forgetting_measure': fm,
    }
