"""
Self-Consistency Inference
Generate N solutions, extract answers, return majority vote.
"""

import torch
from typing import List, Dict, Tuple, Optional
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards.sympy_verifier import extract_answer_from_response, normalize_answer, try_numeric_comparison


def generate_n_solutions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    n: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> List[str]:
    """Generate N diverse solutions for a problem."""
    prompt = (
        f"<|im_start|>system\nYou are MathPhD++, an advanced mathematical reasoning assistant. "
        f"Show your complete reasoning step-by-step. Verify your answer.<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    solutions = []

    model.eval()
    with torch.no_grad():
        for _ in range(n):
            with torch.amp.autocast('cuda'):
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                output[0][input_ids.size(1):],
                skip_special_tokens=True,
            )
            solutions.append(response)

    return solutions


def cluster_answers(answers: List[str]) -> List[Tuple[str, int]]:
    """Cluster answers by equivalence, handling numeric and symbolic forms.

    Returns list of (canonical_answer, count) sorted by count descending.
    """
    clusters = {}  # canonical -> (answer, count)

    for answer in answers:
        normalized = normalize_answer(answer)
        matched = False

        for canonical in list(clusters.keys()):
            # Try numeric comparison
            result = try_numeric_comparison(normalized, canonical)
            if result is True:
                clusters[canonical] = (clusters[canonical][0], clusters[canonical][1] + 1)
                matched = True
                break

            # Exact match after normalization
            if normalized == canonical:
                clusters[canonical] = (clusters[canonical][0], clusters[canonical][1] + 1)
                matched = True
                break

        if not matched:
            clusters[normalized] = (answer, 1)

    # Sort by count
    sorted_clusters = sorted(clusters.values(), key=lambda x: x[1], reverse=True)
    return sorted_clusters


def self_consistency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    n: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> Dict:
    """Self-consistency inference: majority vote over N solutions.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        problem: Math problem text
        n: Number of solutions to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_new_tokens: Max tokens per solution
        device: Device

    Returns:
        dict with:
        - 'answer': majority vote answer
        - 'confidence': proportion of solutions agreeing
        - 'solutions': all generated solutions
        - 'clusters': answer clusters with counts
    """
    # Generate N solutions
    solutions = generate_n_solutions(
        model, tokenizer, problem,
        n=n, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens, device=device,
    )

    # Extract answers
    answers = [extract_answer_from_response(sol) for sol in solutions]

    # Cluster by equivalence
    clusters = cluster_answers(answers)

    # Majority vote
    best_answer, best_count = clusters[0] if clusters else ("", 0)
    confidence = best_count / n if n > 0 else 0

    return {
        "answer": best_answer,
        "confidence": confidence,
        "solutions": solutions,
        "answers": answers,
        "clusters": clusters,
        "n": n,
    }
