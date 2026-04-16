"""
Monte Carlo Tree Search (MCTS) for Mathematical Proof Search
Adapted from AlphaProof-style MCTS with UCT selection.
"""

import math
import random
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards.sympy_verifier import extract_answer_from_response, verify_answer


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    state: str  # Full text state
    action: str  # Action (reasoning step) that led here
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visit_count: int = 0
    total_reward: float = 0.0
    prior: float = 0.0  # Policy prior P(a|s)
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Mean reward Q(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def uct_score(self, c_puct: float = 1.5) -> float:
        """UCT score: Q(s,a) + c * P(a|s) * sqrt(N(parent)) / (1 + N(s,a))"""
        if self.parent is None:
            return 0.0
        parent_visits = max(self.parent.visit_count, 1)
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """Select child with highest UCT score."""
        return max(self.children, key=lambda c: c.uct_score(c_puct))


class MCTS:
    """Monte Carlo Tree Search for mathematical reasoning.

    Uses the language model as both policy (for expansion) and
    the SymPy verifier as the reward signal for backpropagation.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        c_puct: float = 1.5,
        num_simulations: int = 50,
        max_depth: int = 15,
        rollout_max_tokens: int = 512,
        num_expansion_actions: int = 3,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.rollout_max_tokens = rollout_max_tokens
        self.num_expansion_actions = num_expansion_actions
        self.device = device

    def _generate_actions(self, state: str, k: int = 3) -> List[Tuple[str, float]]:
        """Generate k possible next actions (reasoning steps) with priors.

        Returns list of (action_text, prior_probability) tuples.
        """
        input_ids = self.tokenizer.encode(state, return_tensors="pt").to(self.device)
        actions = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(k):
                with torch.amp.autocast('cuda'):
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=100,
                        temperature=0.9,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                new_tokens = output.sequences[0][input_ids.size(1):]
                step_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                step_text = step_text.split("\n\n")[0].strip()

                if not step_text:
                    continue

                # Compute prior from mean log-prob of generated tokens
                if output.scores:
                    log_probs = []
                    for i, scores in enumerate(output.scores):
                        if i < len(new_tokens):
                            log_prob = torch.log_softmax(scores[0], dim=-1)
                            token_log_prob = log_prob[new_tokens[i]].item()
                            log_probs.append(token_log_prob)
                    prior = math.exp(sum(log_probs) / max(len(log_probs), 1))
                else:
                    prior = 1.0 / k

                actions.append((step_text, prior))

        # Normalize priors
        total_prior = sum(p for _, p in actions)
        if total_prior > 0:
            actions = [(text, p / total_prior) for text, p in actions]

        return actions

    def _rollout(self, state: str) -> str:
        """Complete a reasoning chain from current state using greedy decoding."""
        input_ids = self.tokenizer.encode(state, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.rollout_max_tokens,
                    temperature=0.0,  # Greedy for rollout
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return full_text

    def _is_terminal(self, text: str) -> bool:
        """Check if state contains a final answer."""
        markers = ["<answer>", "\\boxed{", "the answer is", "therefore,"]
        return any(m in text.lower() for m in markers)

    def _evaluate(self, state: str, ground_truth: Optional[str] = None) -> float:
        """Evaluate terminal state.

        If ground truth is available, verify with SymPy.
        Otherwise, use heuristic scoring.
        """
        if ground_truth:
            predicted = extract_answer_from_response(state)
            score, _ = verify_answer(predicted, ground_truth)
            return score

        # Heuristic: check if solution looks complete and well-structured
        score = 0.3  # Base
        if "<answer>" in state:
            score += 0.3
        if "therefore" in state.lower() or "thus" in state.lower():
            score += 0.2
        if "\\boxed{" in state:
            score += 0.2
        return min(score, 1.0)

    def search(
        self,
        problem: str,
        ground_truth: Optional[str] = None,
    ) -> Dict:
        """Run MCTS search on a problem.

        Args:
            problem: Math problem text
            ground_truth: Optional ground truth for reward verification

        Returns:
            dict with 'answer', 'reasoning', 'simulations_used', 'tree_size'
        """
        # Create root node
        prompt = (
            f"<|im_start|>system\nYou are MathPhD++. Solve step by step.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<thinking>\n"
        )
        root = MCTSNode(state=prompt, action="root")
        tree_size = 1

        for sim in range(self.num_simulations):
            node = root

            # 1. SELECTION: traverse tree using UCT
            while node.children and not node.is_terminal:
                node = node.best_child(self.c_puct)

            # 2. EXPANSION: add new children
            if not node.is_terminal and node.visit_count > 0:
                actions = self._generate_actions(
                    node.state,
                    k=self.num_expansion_actions,
                )

                for action_text, prior in actions:
                    new_state = node.state + action_text + "\n"
                    child = MCTSNode(
                        state=new_state,
                        action=action_text,
                        parent=node,
                        prior=prior,
                        is_terminal=self._is_terminal(new_state),
                    )
                    node.children.append(child)
                    tree_size += 1

                if node.children:
                    node = node.children[0]  # Explore first new child

            # 3. SIMULATION (rollout)
            if node.is_terminal:
                reward = self._evaluate(node.state, ground_truth)
            else:
                rollout_text = self._rollout(node.state)
                reward = self._evaluate(rollout_text, ground_truth)

            # 4. BACKPROPAGATION
            current = node
            while current is not None:
                current.visit_count += 1
                current.total_reward += reward
                current = current.parent

        # Select best child of root by visit count
        if root.children:
            best_child = max(root.children, key=lambda c: c.visit_count)

            # Follow the most-visited path to get full reasoning
            reasoning_node = best_child
            while reasoning_node.children:
                reasoning_node = max(reasoning_node.children, key=lambda c: c.visit_count)

            answer = extract_answer_from_response(reasoning_node.state)

            return {
                "answer": answer,
                "reasoning": reasoning_node.state,
                "simulations_used": self.num_simulations,
                "tree_size": tree_size,
                "root_visits": root.visit_count,
                "best_child_visits": best_child.visit_count,
                "best_q_value": best_child.q_value,
            }

        # Fallback: direct generation
        rollout = self._rollout(prompt)
        return {
            "answer": extract_answer_from_response(rollout),
            "reasoning": rollout,
            "simulations_used": self.num_simulations,
            "tree_size": tree_size,
            "root_visits": root.visit_count,
            "best_child_visits": 0,
            "best_q_value": 0.0,
        }
