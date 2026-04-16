"""
Evaluation Metrics
Accuracy, pass@k, and specialized math metrics.
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


def accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute exact match accuracy after normalization."""
    from ..rewards.sympy_verifier import verify_answer

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        score, _ = verify_answer(pred, gt)
        if score > 0.5:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def pass_at_k(results: List[List[bool]], k: int) -> float:
    """Compute pass@k metric.

    Args:
        results: List of [bool, ...] per problem (True = correct)
        k: Number of attempts

    Returns:
        Probability of getting at least one correct in k attempts.
    """
    scores = []
    for per_problem in results:
        n = len(per_problem)
        c = sum(per_problem)
        if n - c < k:
            score = 1.0
        else:
            score = 1.0 - np.prod([(n - c - i) / (n - i) for i in range(k)])
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def accuracy_by_difficulty(
    predictions: List[str],
    ground_truths: List[str],
    difficulties: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute accuracy broken down by difficulty level."""
    from ..rewards.sympy_verifier import verify_answer

    by_level = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, gt, diff in zip(predictions, ground_truths, difficulties):
        score, _ = verify_answer(pred, gt)
        by_level[diff]["total"] += 1
        if score > 0.5:
            by_level[diff]["correct"] += 1

    result = {}
    for level in sorted(by_level.keys()):
        data = by_level[level]
        result[level] = {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"],
        }

    return result


def accuracy_by_subject(
    predictions: List[str],
    ground_truths: List[str],
    subjects: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute accuracy broken down by math subject."""
    from ..rewards.sympy_verifier import verify_answer

    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, gt, subject in zip(predictions, ground_truths, subjects):
        score, _ = verify_answer(pred, gt)
        by_subject[subject]["total"] += 1
        if score > 0.5:
            by_subject[subject]["correct"] += 1

    result = {}
    for subject in sorted(by_subject.keys()):
        data = by_subject[subject]
        result[subject] = {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"],
        }

    return result
