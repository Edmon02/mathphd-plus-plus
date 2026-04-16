"""
Unified Inference Pipeline
Dispatches to the appropriate inference strategy based on problem difficulty.
"""

import torch
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .self_consistency import self_consistency
from .tree_of_thoughts import TreeOfThoughts
from .mcts import MCTS
from .multi_agent_debate import debate


class InferencePipeline:
    """Unified inference pipeline that selects strategy based on difficulty.

    Strategy selection:
    - Easy (difficulty 1-2): Direct generation or self-consistency
    - Medium (difficulty 3): Tree-of-Thoughts with PRM
    - Hard (difficulty 4-5): MCTS + multi-agent debate
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        process_scorer=None,
        config=None,
        device: str = "cuda",
    ):
        from ..configs.base_config import InferenceConfig
        self.config = config or InferenceConfig()

        self.model = model
        self.tokenizer = tokenizer
        self.process_scorer = process_scorer
        self.device = device

        # Initialize strategies
        self.tot = TreeOfThoughts(
            model=model,
            tokenizer=tokenizer,
            process_scorer=process_scorer,
            beam_width=self.config.tot_beam_width,
            max_depth=self.config.tot_max_depth,
            branching_factor=self.config.tot_branching_factor,
            step_max_tokens=self.config.tot_step_tokens,
            device=device,
        )

        self.mcts = MCTS(
            model=model,
            tokenizer=tokenizer,
            c_puct=self.config.mcts_c_puct,
            num_simulations=self.config.mcts_num_simulations,
            max_depth=self.config.mcts_max_depth,
            rollout_max_tokens=self.config.mcts_rollout_max_tokens,
            device=device,
        )

    def solve(
        self,
        problem: str,
        difficulty: Optional[int] = None,
        strategy: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict:
        """Solve a problem using the appropriate strategy.

        Args:
            problem: Math problem text
            difficulty: 1-5 difficulty level (auto-detected if None)
            strategy: Force a specific strategy: "direct", "sc", "tot", "mcts", "debate"
            ground_truth: Optional answer for MCTS reward

        Returns:
            dict with 'answer', 'reasoning', 'strategy_used', 'confidence'
        """
        if strategy:
            return self._run_strategy(strategy, problem, ground_truth)

        # Auto-select based on difficulty
        if difficulty is None:
            difficulty = self._estimate_difficulty(problem)

        if difficulty <= 2:
            return self._run_strategy("sc", problem, ground_truth)
        elif difficulty <= 3:
            return self._run_strategy("tot", problem, ground_truth)
        else:
            # Try MCTS first, fall back to debate if no good solution
            result = self._run_strategy("mcts", problem, ground_truth)
            if result.get("confidence", 0) < 0.3:
                debate_result = self._run_strategy("debate", problem, ground_truth)
                if debate_result.get("confidence", 0) > result.get("confidence", 0):
                    return debate_result
            return result

    def _run_strategy(
        self,
        strategy: str,
        problem: str,
        ground_truth: Optional[str] = None,
    ) -> Dict:
        """Run a specific inference strategy."""
        if strategy == "direct":
            return self._direct_generation(problem)

        elif strategy == "sc":
            result = self_consistency(
                self.model, self.tokenizer, problem,
                n=self.config.sc_num_samples,
                temperature=self.config.sc_temperature,
                top_p=self.config.sc_top_p,
                max_new_tokens=self.config.max_new_tokens,
                device=self.device,
            )
            result["strategy_used"] = "self_consistency"
            return result

        elif strategy == "tot":
            result = self.tot.search(problem)
            result["strategy_used"] = "tree_of_thoughts"
            result["confidence"] = result.get("score", 0.5)
            return result

        elif strategy == "mcts":
            result = self.mcts.search(problem, ground_truth)
            result["strategy_used"] = "mcts"
            result["confidence"] = result.get("best_q_value", 0.5)
            return result

        elif strategy == "debate":
            result = debate(
                self.model, self.tokenizer, problem,
                rounds=self.config.debate_rounds,
                ground_truth=ground_truth,
                process_scorer=self.process_scorer,
                device=self.device,
            )
            result["strategy_used"] = "multi_agent_debate"
            result["confidence"] = 0.5  # Debate doesn't produce a confidence score
            return result

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _direct_generation(self, problem: str) -> Dict:
        """Simple direct generation (greedy decoding)."""
        prompt = (
            f"<|im_start|>system\nYou are MathPhD++. Solve step by step.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        response = self.tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)

        from ..rewards.sympy_verifier import extract_answer_from_response
        answer = extract_answer_from_response(response)

        return {
            "answer": answer,
            "reasoning": response,
            "strategy_used": "direct",
            "confidence": 0.5,
        }

    def _estimate_difficulty(self, problem: str) -> int:
        """Estimate problem difficulty from text heuristics."""
        text_lower = problem.lower()

        hard_indicators = [
            "prove that", "show that", "find all", "determine all",
            "conjecture", "topology", "cohomology", "manifold",
            "galois", "riemann", "functional equation",
        ]
        medium_indicators = [
            "integral", "derivative", "eigenvalue", "matrix",
            "polynomial", "sequence", "series", "probability",
        ]
        easy_indicators = [
            "calculate", "compute", "simplify", "evaluate",
            "what is", "how many", "find the value",
        ]

        score = 3  # Default medium

        for indicator in hard_indicators:
            if indicator in text_lower:
                score += 1

        for indicator in medium_indicators:
            if indicator in text_lower:
                score += 0.3

        for indicator in easy_indicators:
            if indicator in text_lower:
                score -= 1

        return max(1, min(5, int(score)))
