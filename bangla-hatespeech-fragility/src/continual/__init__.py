"""Continual learning modules."""

from .strategies import (
    ContinualLearningStrategy,
    SequentialFineTuning,
    ExperienceReplay,
    ElasticWeightConsolidation
)
import numpy as np


def compute_continual_learning_metrics(accuracies):
    """
    Compute continual learning metrics from accuracy matrix.
    
    Args:
        accuracies: Dict mapping task_index -> {eval_task_index: accuracy}
    
    Returns:
        Dict with metrics: AA (Average Accuracy), BWT (Backward Transfer), FM (Forward Transfer)
    """
    # Convert to matrix format: rows = task, cols = eval_task
    n_tasks = len(accuracies)
    acc_matrix = np.zeros((n_tasks, n_tasks))
    
    for task_idx, task_accs in accuracies.items():
        for eval_idx, acc in task_accs.items():
            acc_matrix[task_idx, eval_idx] = acc
    
    # Average Accuracy: mean of diagonal (final accuracy on each task)
    AA = np.mean(np.diag(acc_matrix))
    
    # Backward Transfer: how much did learning new tasks hurt old task performance
    # BWT = (performance_on_old_after_new - performance_on_old_initially)
    BWT_values = []
    for i in range(n_tasks - 1):  # For each task except the last
        final_perf = acc_matrix[n_tasks - 1, i]  # Performance after learning all tasks
        initial_perf = acc_matrix[i, i]  # Initial performance
        BWT_values.append(final_perf - initial_perf)
    
    BWT = np.mean(BWT_values) if BWT_values else 0.0
    
    # Forward Transfer: improvement on new task due to learning previous tasks
    # FM = (performance_on_new_with_prior_learning - random_baseline)
    # Using performance on each task vs assuming 0.5 baseline for hate speech detection
    FM_values = []
    for i in range(1, n_tasks):  # For each task after the first
        perf_with_prior = acc_matrix[i, i]  # Performance when learning this task
        # Approximation: benefit over baseline
        FM_values.append(perf_with_prior - 0.5)  # 0.5 is random baseline
    
    FM = np.mean(FM_values) if FM_values else 0.0
    
    return {
        'AA': AA,  # Average Accuracy
        'BWT': BWT,  # Backward Transfer (negative is forgetting)
        'FM': FM   # Forward Transfer
    }


__all__ = [
    "ContinualLearningStrategy",
    "SequentialFineTuning",
    "ExperienceReplay",
    "ElasticWeightConsolidation",
    "compute_continual_learning_metrics"
]
