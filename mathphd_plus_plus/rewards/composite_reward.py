"""
Composite Reward Function
Combines correctness, process, and format rewards for GRPO training.

R_total = w1 * R_correctness + w2 * R_process + w3 * R_format
"""

import math
import re
from typing import Dict, Optional

from .sympy_verifier import verify_answer, extract_answer_from_response


class CompositeReward:
    """Combined reward function for GRPO training.

    Components:
    - R_correctness (w=0.6): SymPy-verified answer correctness
    - R_process (w=0.3): Step-level quality from PRM
    - R_format (w=0.1): Structural compliance bonus
    """

    def __init__(
        self,
        process_scorer=None,  # ProcessRewardScorer instance
        correctness_weight: float = 0.6,
        process_weight: float = 0.3,
        format_weight: float = 0.1,
    ):
        self.process_scorer = process_scorer
        self.w_correct = correctness_weight
        self.w_process = process_weight
        self.w_format = format_weight

    def compute_correctness_reward(
        self,
        response: str,
        ground_truth: str,
    ) -> float:
        """Compute correctness reward via SymPy verification.

        Returns 1.0 if correct, 0.0 if incorrect.
        """
        predicted = extract_answer_from_response(response)
        score, method = verify_answer(predicted, ground_truth)
        return score

    def compute_process_reward(self, response: str) -> float:
        """Compute process reward using PRM.

        Returns mean step score in [0, 1].
        If no PRM available, returns 0.5 (neutral).
        """
        if self.process_scorer is None:
            return 0.5

        try:
            mean_score, _ = self.process_scorer.score_solution(response)
            if math.isnan(mean_score) or math.isinf(mean_score):
                return 0.5
            return max(0.0, min(1.0, mean_score))
        except Exception:
            return 0.5

    def compute_format_reward(self, response: str) -> float:
        """Compute format compliance reward.

        Checks for structured output tags: <thinking>, <answer>, <verification>.
        Returns bonus score in [0, 1].
        """
        score = 0.0
        checks = {
            "<thinking>": 0.3,
            "</thinking>": 0.1,
            "<answer>": 0.3,
            "</answer>": 0.1,
            "<verification>": 0.15,
            "</verification>": 0.05,
        }

        for tag, weight in checks.items():
            if tag in response:
                score += weight

        return min(score, 1.0)

    def compute_reward(
        self,
        response: str,
        ground_truth: str,
    ) -> Dict[str, float]:
        """Compute composite reward.

        Args:
            response: Model-generated solution
            ground_truth: Correct answer string

        Returns:
            dict with 'total', 'correctness', 'process', 'format' scores
        """
        r_correct = self.compute_correctness_reward(response, ground_truth)
        r_process = self.compute_process_reward(response)
        r_format = self.compute_format_reward(response)

        total = (
            self.w_correct * r_correct +
            self.w_process * r_process +
            self.w_format * r_format
        )

        if math.isnan(total) or math.isinf(total):
            total = 0.0
        total = max(0.0, min(1.0, total))

        return {
            "total": total,
            "correctness": r_correct,
            "process": r_process,
            "format": r_format,
        }

    def compute_batch_rewards(
        self,
        responses: list,
        ground_truth: str,
    ) -> list:
        """Compute rewards for a group of responses (for GRPO).

        Args:
            responses: List of G generated solutions
            ground_truth: Correct answer string

        Returns:
            List of reward dicts, one per response
        """
        return [self.compute_reward(r, ground_truth) for r in responses]
