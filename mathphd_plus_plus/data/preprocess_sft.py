"""
SFT Data Preprocessor
Formats multiple datasets into ChatML format with structured math output.
"""

import re
import random
import hashlib
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are MathPhD++, an advanced mathematical reasoning assistant. "
    "Show your complete reasoning step-by-step. Verify your answer using "
    "an independent method when possible."
)


def format_chatml(
    problem: str,
    solution: str,
    system: str = SYSTEM_PROMPT,
) -> str:
    """Format a problem-solution pair into ChatML format with structured output."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{solution}<|im_end|>"
    )


def structure_solution(solution: str, answer: Optional[str] = None) -> str:
    """Wrap a solution in <thinking>/<answer>/<verification> tags."""
    # If solution already has structure, return as-is
    if "<thinking>" in solution:
        return solution

    # Split into reasoning and final answer
    lines = solution.strip().split("\n")

    # Try to detect final answer line
    answer_line = None
    reasoning_lines = lines
    for i, line in enumerate(lines):
        if any(marker in line.lower() for marker in [
            "the answer is", "therefore", "thus,", "hence,", "= \\boxed",
            "\\boxed{", "final answer",
        ]):
            answer_line = line
            reasoning_lines = lines[:i]
            break

    thinking = "\n".join(reasoning_lines).strip()
    if not thinking:
        thinking = solution.strip()

    if answer_line:
        final_answer = answer_line.strip()
    elif answer:
        final_answer = answer
    else:
        final_answer = lines[-1].strip() if lines else solution.strip()

    return (
        f"<thinking>\n{thinking}\n</thinking>\n"
        f"<answer>\n{final_answer}\n</answer>"
    )


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract \\boxed{...} answer from solution text."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1)
    return None


def process_metamath(dataset, max_samples: int = 40_000, seed: int = 42) -> List[Dict]:
    """Process MetaMathQA dataset."""
    samples = []
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)

    for idx in indices[:max_samples]:
        item = dataset[idx]
        problem = item.get("query", item.get("question", ""))
        solution = item.get("response", item.get("answer", ""))

        if not problem or not solution:
            continue

        structured = structure_solution(solution)
        formatted = format_chatml(problem, structured)
        samples.append({
            "text": formatted,
            "source": "metamath",
            "difficulty": 2,  # Medium
        })

    return samples


def process_math_dataset(dataset) -> List[Dict]:
    """Process MATH competition dataset."""
    difficulty_map = {
        "Level 1": 1, "Level 2": 2, "Level 3": 3, "Level 4": 4, "Level 5": 5,
    }
    samples = []

    for item in dataset:
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        level = item.get("level", "Level 3")

        if not problem or not solution:
            continue

        answer = extract_boxed_answer(solution)
        structured = structure_solution(solution, answer)
        formatted = format_chatml(problem, structured)
        samples.append({
            "text": formatted,
            "source": "math",
            "difficulty": difficulty_map.get(level, 3),
            "answer": answer or "",
            "subject": item.get("type", ""),
        })

    return samples


def process_gsm8k(dataset) -> List[Dict]:
    """Process GSM8K dataset."""
    samples = []

    for item in dataset:
        problem = item.get("question", "")
        solution = item.get("answer", "")

        if not problem or not solution:
            continue

        # GSM8K answers have #### delimiter
        parts = solution.split("####")
        reasoning = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""

        structured = (
            f"<thinking>\n{reasoning}\n</thinking>\n"
            f"<answer>\n{answer}\n</answer>"
        )
        formatted = format_chatml(problem, structured)
        samples.append({
            "text": formatted,
            "source": "gsm8k",
            "difficulty": 1,
            "answer": answer,
        })

    return samples


def process_numina(dataset, max_samples: int = 3_000, seed: int = 42) -> List[Dict]:
    """Process NuminaMath-CoT dataset."""
    samples = []
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)

    for idx in indices[:max_samples]:
        item = dataset[idx]
        problem = item.get("problem", item.get("question", ""))
        solution = item.get("solution", item.get("answer", ""))

        if not problem or not solution:
            continue

        structured = structure_solution(solution)
        formatted = format_chatml(problem, structured)
        source_type = item.get("source", "competition")

        # Estimate difficulty from source
        diff = 3
        if "aime" in str(source_type).lower():
            diff = 5
        elif "amc" in str(source_type).lower():
            diff = 4
        elif "mathd" in str(source_type).lower():
            diff = 2

        samples.append({
            "text": formatted,
            "source": "numina",
            "difficulty": diff,
        })

    return samples


def dedup_by_problem(samples: List[Dict], threshold: float = 0.95) -> List[Dict]:
    """Simple deduplication based on problem text hash."""
    seen_hashes = set()
    deduped = []

    for sample in samples:
        # Extract just the problem portion for hashing
        text = sample["text"]
        user_start = text.find("<|im_start|>user\n")
        user_end = text.find("<|im_end|>", user_start)
        if user_start >= 0 and user_end >= 0:
            problem_text = text[user_start:user_end].lower().strip()
        else:
            problem_text = text[:200].lower().strip()

        # Normalize whitespace
        problem_text = re.sub(r'\s+', ' ', problem_text)
        h = hashlib.md5(problem_text.encode()).hexdigest()

        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(sample)

    return deduped


def prepare_sft_dataset(
    sft_datasets: Dict,
    tokenizer: AutoTokenizer,
    config=None,
    max_seq_length: int = 1024,
    seed: int = 42,
) -> Dataset:
    """Full SFT data preparation pipeline.

    1. Process each source dataset
    2. Combine and deduplicate
    3. Filter by length
    4. Sort by difficulty (curriculum)
    5. Create tokenized dataset
    """
    all_samples = []

    # Process each dataset
    if "metamath" in sft_datasets:
        max_n = 40_000
        if config and hasattr(config, 'sft_data_mix'):
            max_n = config.sft_data_mix.get("meta-math/MetaMathQA", 40_000)
        samples = process_metamath(sft_datasets["metamath"], max_samples=max_n, seed=seed)
        print(f"  MetaMathQA: {len(samples)} samples")
        all_samples.extend(samples)

    if "math_train" in sft_datasets:
        samples = process_math_dataset(sft_datasets["math_train"])
        print(f"  MATH train: {len(samples)} samples")
        all_samples.extend(samples)

    if "gsm8k_train" in sft_datasets:
        samples = process_gsm8k(sft_datasets["gsm8k_train"])
        print(f"  GSM8K train: {len(samples)} samples")
        all_samples.extend(samples)

    if "numina" in sft_datasets:
        max_n = 3_000
        if config and hasattr(config, 'sft_data_mix'):
            max_n = config.sft_data_mix.get("AI-MO/NuminaMath-CoT", 3_000)
        samples = process_numina(sft_datasets["numina"], max_samples=max_n, seed=seed)
        print(f"  NuminaMath: {len(samples)} samples")
        all_samples.extend(samples)

    print(f"\n[SFT] Total before dedup: {len(all_samples)}")

    # Deduplicate
    all_samples = dedup_by_problem(all_samples)
    print(f"[SFT] After dedup: {len(all_samples)}")

    # Filter by token length
    filtered = []
    for sample in all_samples:
        token_len = len(tokenizer.encode(sample["text"], add_special_tokens=False))
        if token_len <= max_seq_length:
            sample["token_length"] = token_len
            filtered.append(sample)

    print(f"[SFT] After length filter (max {max_seq_length}): {len(filtered)}")

    # Sort by difficulty for curriculum
    filtered.sort(key=lambda x: x.get("difficulty", 3))

    return Dataset.from_list(filtered)
