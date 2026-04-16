"""
GRPO Data Preprocessor
Filters problems with verifiable answers for reinforcement learning.
"""

import re
from typing import Dict, List, Optional
from datasets import Dataset


def extract_verifiable_answer(text: str) -> Optional[str]:
    """Extract a verifiable (numeric or symbolic) answer from solution text."""
    # Try \\boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Try #### delimiter (GSM8K style)
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            return parts[-1].strip()

    # Try "The answer is X" pattern
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)', text)
    if match:
        return match.group(1).strip()

    return None


def classify_answer_type(answer: str) -> str:
    """Classify answer as numeric, symbolic, expression, or unknown."""
    if not answer:
        return "unknown"

    # Clean LaTeX
    clean = answer.replace("\\", "").replace("{", "").replace("}", "")
    clean = clean.replace(",", "").strip()

    # Try numeric
    try:
        float(clean)
        return "numeric"
    except ValueError:
        pass

    # Try fraction
    if "/" in clean:
        parts = clean.split("/")
        if len(parts) == 2:
            try:
                float(parts[0])
                float(parts[1])
                return "numeric"
            except ValueError:
                pass

    # Symbolic expressions
    if any(c in answer for c in ["x", "y", "z", "n", "\\frac", "\\sqrt", "\\pi"]):
        return "symbolic"

    return "expression"


def prepare_grpo_dataset(
    sft_datasets: Dict,
    max_problems: int = 2000,
    seed: int = 42,
) -> Dataset:
    """Prepare GRPO training data.

    Filters for problems with verifiable answers that SymPy can check.
    Assigns difficulty levels for curriculum ordering.
    """
    problems = []

    # Process MATH dataset (primary source for GRPO)
    if "math_train" in sft_datasets:
        difficulty_map = {
            "Level 1": 1, "Level 2": 2, "Level 3": 3, "Level 4": 4, "Level 5": 5,
        }
        for item in sft_datasets["math_train"]:
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            if not problem or not solution:
                continue

            answer = extract_verifiable_answer(solution)
            if answer is None:
                continue

            answer_type = classify_answer_type(answer)
            if answer_type == "unknown":
                continue

            level = item.get("level", "Level 3")
            problems.append({
                "problem": problem,
                "answer": answer,
                "answer_type": answer_type,
                "difficulty": difficulty_map.get(level, 3),
                "source": "math",
                "subject": item.get("type", ""),
            })

    # Process GSM8K (easier problems for warm-up)
    if "gsm8k_train" in sft_datasets:
        for item in sft_datasets["gsm8k_train"]:
            problem = item.get("question", "")
            solution = item.get("answer", "")
            if not problem or not solution:
                continue

            answer = extract_verifiable_answer(solution)
            if answer is None:
                continue

            problems.append({
                "problem": problem,
                "answer": answer,
                "answer_type": "numeric",
                "difficulty": 1,
                "source": "gsm8k",
                "subject": "arithmetic",
            })

    # Process NuminaMath (competition-level)
    if "numina" in sft_datasets:
        for item in sft_datasets["numina"]:
            problem = item.get("problem", item.get("question", ""))
            solution = item.get("solution", item.get("answer", ""))
            if not problem or not solution:
                continue

            answer = extract_verifiable_answer(solution)
            if answer is None:
                continue

            answer_type = classify_answer_type(answer)
            if answer_type == "unknown":
                continue

            problems.append({
                "problem": problem,
                "answer": answer,
                "answer_type": answer_type,
                "difficulty": 4,
                "source": "numina",
                "subject": "competition",
            })

    print(f"[GRPO] Found {len(problems)} verifiable problems")

    # Sort by difficulty (curriculum)
    problems.sort(key=lambda x: x["difficulty"])

    # Subsample if needed
    if len(problems) > max_problems:
        # Stratified sampling across difficulty levels
        by_diff = {}
        for p in problems:
            d = p["difficulty"]
            by_diff.setdefault(d, []).append(p)

        per_level = max_problems // len(by_diff)
        sampled = []
        for d in sorted(by_diff.keys()):
            level_problems = by_diff[d]
            n = min(per_level, len(level_problems))
            sampled.extend(level_problems[:n])

        # Fill remaining from hardest levels
        remaining = max_problems - len(sampled)
        if remaining > 0:
            extra = [p for p in problems if p not in sampled]
            sampled.extend(extra[:remaining])

        problems = sampled
        problems.sort(key=lambda x: x["difficulty"])

    print(f"[GRPO] Final dataset: {len(problems)} problems")
    for d in sorted(set(p["difficulty"] for p in problems)):
        n = sum(1 for p in problems if p["difficulty"] == d)
        print(f"  Difficulty {d}: {n} problems")

    return Dataset.from_list(problems)
