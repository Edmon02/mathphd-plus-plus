"""
Curriculum Learning Scheduler
Orders training data from easy to hard based on difficulty estimates.
"""

import numpy as np
from typing import List, Dict, Optional
from torch.utils.data import Sampler


class CurriculumSampler(Sampler):
    """Sampler that presents data in difficulty-increasing order.

    Supports:
    - Linear curriculum: easy→hard across the epoch
    - Anti-curriculum: hard→easy (for comparison)
    - Mixed: curriculum for first 75%, random for last 25%
    """

    def __init__(
        self,
        difficulties: List[float],
        strategy: str = "mixed",  # "linear", "anti", "mixed", "random"
        seed: int = 42,
    ):
        self.difficulties = np.array(difficulties)
        self.strategy = strategy
        self.rng = np.random.RandomState(seed)
        self.n = len(difficulties)

    def __iter__(self):
        if self.strategy == "random":
            indices = self.rng.permutation(self.n).tolist()

        elif self.strategy == "linear":
            indices = np.argsort(self.difficulties).tolist()

        elif self.strategy == "anti":
            indices = np.argsort(-self.difficulties).tolist()

        elif self.strategy == "mixed":
            sorted_indices = np.argsort(self.difficulties)
            # First 75%: curriculum order
            curriculum_end = int(0.75 * self.n)
            curriculum_part = sorted_indices[:curriculum_end].tolist()
            # Last 25%: random
            random_part = sorted_indices[curriculum_end:]
            self.rng.shuffle(random_part)
            indices = curriculum_part + random_part.tolist()

        else:
            raise ValueError(f"Unknown curriculum strategy: {self.strategy}")

        return iter(indices)

    def __len__(self):
        return self.n


def compute_curriculum_weights(
    difficulties: List[float],
    epoch: int,
    total_epochs: int,
    min_weight: float = 0.1,
) -> List[float]:
    """Compute per-sample loss weights based on curriculum progress.

    Early epochs: upweight easy samples
    Later epochs: uniform weights (all samples contribute equally)
    """
    difficulties = np.array(difficulties)
    progress = epoch / total_epochs  # 0 to 1

    if progress < 0.5:
        # Early: downweight hard samples
        weights = 1.0 - (1.0 - min_weight) * difficulties * (1.0 - 2 * progress)
    else:
        # Late: uniform
        weights = np.ones_like(difficulties)

    return weights.tolist()
