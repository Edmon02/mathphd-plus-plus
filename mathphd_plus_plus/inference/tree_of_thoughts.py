"""
Tree-of-Thoughts (ToT) Inference
PRM-guided beam search over reasoning steps.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards.sympy_verifier import extract_answer_from_response


@dataclass
class ToTNode:
    """A node in the Tree of Thoughts."""
    text: str  # Full text up to this point
    step_text: str  # Just this step's text
    score: float = 0.0  # PRM score
    depth: int = 0
    parent: Optional['ToTNode'] = None
    children: List['ToTNode'] = field(default_factory=list)
    is_terminal: bool = False

    def get_path(self) -> List[str]:
        """Get all step texts from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.step_text)
            node = node.parent
        return list(reversed(path))


class TreeOfThoughts:
    """Tree-of-Thoughts search with PRM-guided beam selection.

    At each depth:
    1. For each beam node, generate K continuations (next reasoning step)
    2. Score each continuation with PRM
    3. Keep top beam_width candidates across all branches
    4. Repeat until solution or max_depth
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        process_scorer=None,  # ProcessRewardScorer
        beam_width: int = 3,
        max_depth: int = 8,
        branching_factor: int = 3,
        step_max_tokens: int = 100,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = process_scorer
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.step_max_tokens = step_max_tokens
        self.device = device

    def generate_step(
        self,
        prefix: str,
        num_continuations: int = 3,
    ) -> List[str]:
        """Generate multiple possible next reasoning steps."""
        input_ids = self.tokenizer.encode(prefix, return_tensors="pt").to(self.device)
        continuations = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_continuations):
                with torch.amp.autocast('cuda'):
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=self.step_max_tokens,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                new_text = self.tokenizer.decode(
                    output[0][input_ids.size(1):],
                    skip_special_tokens=True,
                )

                # Extract first step (up to next newline or step boundary)
                step = new_text.split("\n\n")[0].split("\nStep")[0].strip()
                if step:
                    continuations.append(step)

        return continuations

    def score_node(self, node: ToTNode) -> float:
        """Score a node using PRM or heuristic."""
        if self.scorer:
            return self.scorer.score_step(node.text)

        # Heuristic fallback: prefer longer, more detailed reasoning
        text = node.text
        score = 0.5  # Base

        # Reward mathematical content
        math_indicators = ["therefore", "thus", "hence", "since", "because",
                           "=", "\\frac", "\\sum", "\\int", "proof"]
        for indicator in math_indicators:
            if indicator in text.lower():
                score += 0.05

        # Reward structured output
        if "<answer>" in text:
            score += 0.2

        return min(score, 1.0)

    def is_terminal(self, text: str) -> bool:
        """Check if text contains a final answer."""
        terminal_markers = [
            "<answer>", "</answer>",
            "\\boxed{", "the answer is",
            "therefore, the answer", "thus, the answer",
            "final answer:",
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in terminal_markers)

    def search(self, problem: str) -> Dict:
        """Run Tree-of-Thoughts search on a problem.

        Returns:
            dict with 'answer', 'reasoning', 'nodes_explored', 'best_path'
        """
        # Create root node with problem prompt
        prompt = (
            f"<|im_start|>system\nYou are MathPhD++. Solve step by step.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<thinking>\n"
        )

        root = ToTNode(text=prompt, step_text="", depth=0)
        beam = [root]
        all_terminals = []
        nodes_explored = 0

        for depth in range(self.max_depth):
            candidates = []

            for node in beam:
                if node.is_terminal:
                    all_terminals.append(node)
                    continue

                # Generate continuations
                continuations = self.generate_step(
                    node.text,
                    num_continuations=self.branching_factor,
                )

                for step_text in continuations:
                    new_text = node.text + step_text + "\n"
                    child = ToTNode(
                        text=new_text,
                        step_text=step_text,
                        depth=depth + 1,
                        parent=node,
                    )
                    child.score = self.score_node(child)
                    child.is_terminal = self.is_terminal(new_text)
                    node.children.append(child)
                    candidates.append(child)
                    nodes_explored += 1

                    if child.is_terminal:
                        all_terminals.append(child)

            if not candidates:
                break

            # Select top beam_width candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:self.beam_width]

        # Select best terminal node, or best leaf if no terminal
        if all_terminals:
            best = max(all_terminals, key=lambda x: x.score)
        elif beam:
            best = max(beam, key=lambda x: x.score)
        else:
            best = root

        # Extract answer
        answer = extract_answer_from_response(best.text)

        return {
            "answer": answer,
            "reasoning": best.text,
            "best_path": best.get_path(),
            "score": best.score,
            "nodes_explored": nodes_explored,
            "depth_reached": best.depth,
            "num_terminals": len(all_terminals),
        }
