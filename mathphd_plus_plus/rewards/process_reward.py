"""
Process Reward Module
Wrapper for using the PRM at inference time for step-level scoring.
"""

import torch
from typing import List, Tuple, Optional
from transformers import AutoTokenizer


class ProcessRewardScorer:
    """Scores reasoning steps using a trained Process Reward Model."""

    def __init__(
        self,
        model,  # ProcessRewardModel instance
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        step_delimiter: str = "\n",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.step_delimiter = step_delimiter
        self.model.eval()

    @torch.no_grad()
    def score_solution(self, solution: str) -> Tuple[float, List[Tuple[str, float]]]:
        """Score a complete solution by scoring each step prefix.

        Args:
            solution: Full solution text

        Returns:
            (mean_score, [(step_text, step_score), ...])
        """
        steps = [s.strip() for s in solution.split(self.step_delimiter) if s.strip()]
        if not steps:
            return 0.0, []

        step_scores = []
        prefix = ""

        for step in steps:
            prefix += step + self.step_delimiter
            encoding = self.tokenizer(
                prefix,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            ).to(self.device)

            result = self.model(**encoding)
            score = result["rewards"].item()
            step_scores.append((step, score))

        mean_score = sum(s for _, s in step_scores) / len(step_scores)
        return mean_score, step_scores

    @torch.no_grad()
    def score_step(self, prefix: str) -> float:
        """Score a single reasoning prefix.

        Args:
            prefix: All text up to and including the current step

        Returns:
            float score in [0, 1]
        """
        encoding = self.tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        ).to(self.device)

        result = self.model(**encoding)
        return result["rewards"].item()

    @torch.no_grad()
    def batch_score_prefixes(self, prefixes: List[str]) -> List[float]:
        """Score multiple prefixes in a batch.

        Args:
            prefixes: List of text prefixes to score

        Returns:
            List of float scores
        """
        if not prefixes:
            return []

        encoding = self.tokenizer(
            prefixes,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(self.device)

        result = self.model(**encoding)
        return result["rewards"].tolist()
