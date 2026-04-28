"""
Composite Reward Function
Combines correctness, process, and format rewards for GRPO training.

R_total = w1 * R_correctness + w2 * R_process + w3 * R_format
"""

import math
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
    ) -> Dict[str, object]:
        """Compute correctness reward via SymPy verification.

        Returns 1.0 if correct, 0.0 if incorrect.
        """
        predicted = extract_answer_from_response(response)
        score, method = verify_answer(predicted, ground_truth)
        return {
            "score": score,
            "method": method,
            "predicted": predicted,
            "has_prediction": bool(predicted.strip()),
        }

    def compute_process_reward(self, response: str) -> float:
        """Compute process reward using PRM.

        Returns mean step score in [0, 1].
        If no PRM is available or scoring fails, returns None so the caller can
        ignore this term instead of silently injecting a neutral reward.
        """
        if self.process_scorer is None:
            return None

        try:
            mean_score, _ = self.process_scorer.score_solution(response)
            if math.isnan(mean_score) or math.isinf(mean_score):
                return None
            return max(0.0, min(1.0, mean_score))
        except Exception:
            return None

    def compute_format_reward(self, response: str) -> float:
        """Compute format compliance reward.

        Checks for structured output tags: <thinking>, <answer>, <verification>.
        Returns bonus score in [0, 1].
        """
        score = 0.0
        if "<thinking>" in response and "</thinking>" in response:
            score += 0.3
        if "<answer>" in response and "</answer>" in response:
            score += 0.5
        if "<verification>" in response and "</verification>" in response:
            score += 0.2
        return score

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
        correctness = self.compute_correctness_reward(response, ground_truth)
        r_process = self.compute_process_reward(response)
        r_format = self.compute_format_reward(response)

        weighted_terms = [
            (self.w_correct, correctness["score"]),
            (self.w_format, r_format),
        ]
        if r_process is not None:
            weighted_terms.append((self.w_process, r_process))

        total_weight = sum(weight for weight, _ in weighted_terms)
        total = sum(weight * score for weight, score in weighted_terms)
        if total_weight > 0:
            total = total / total_weight

        if math.isnan(total) or math.isinf(total):
            total = 0.0
        total = max(0.0, min(1.0, total))

        return {
            "total": total,
            "correctness": correctness["score"],
            "correctness_method": correctness["method"],
            "predicted_answer": correctness["predicted"],
            "has_prediction": correctness["has_prediction"],
            "process": r_process if r_process is not None else 0.0,
            "process_available": r_process is not None,
            "format": r_format,
            "active_reward_weight": total_weight,
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
