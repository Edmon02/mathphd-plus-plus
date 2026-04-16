"""
Evaluation Runner
Run benchmarks (GSM8K, MATH, AIME) and produce detailed reports.
"""

import os
import json
import torch
from typing import Dict, Optional, List
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards.sympy_verifier import extract_answer_from_response, verify_answer
from .metrics import accuracy, accuracy_by_difficulty, accuracy_by_subject


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """Generate a single answer using greedy decoding."""
    prompt = (
        f"<|im_start|>system\nYou are MathPhD++, an advanced mathematical reasoning assistant. "
        f"Show your complete reasoning step-by-step.<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=temperature > 0,
            )

    response = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
    return response


def evaluate_gsm8k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    cache_dir: str = "./data_cache",
) -> Dict:
    """Evaluate on GSM8K test set."""
    print("[EVAL] Running GSM8K evaluation...")

    dataset = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    predictions = []
    ground_truths = []
    results = []

    for item in tqdm(dataset, desc="GSM8K"):
        problem = item["question"]
        solution = item["answer"]

        # Extract ground truth
        gt = solution.split("####")[-1].strip() if "####" in solution else ""

        # Generate
        response = generate_answer(model, tokenizer, problem, device=device)
        pred = extract_answer_from_response(response)

        score, method = verify_answer(pred, gt)

        predictions.append(pred)
        ground_truths.append(gt)
        results.append({
            "problem": problem,
            "prediction": pred,
            "ground_truth": gt,
            "correct": score > 0.5,
            "method": method,
            "full_response": response,
        })

    acc = accuracy(predictions, ground_truths)
    print(f"[GSM8K] Accuracy: {acc:.2%} ({sum(1 for r in results if r['correct'])}/{len(results)})")

    return {
        "benchmark": "gsm8k",
        "accuracy": acc,
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "results": results,
    }


def evaluate_math(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    cache_dir: str = "./data_cache",
) -> Dict:
    """Evaluate on MATH test set with difficulty and subject breakdown."""
    print("[EVAL] Running MATH evaluation...")

    dataset = load_dataset("hendrycks/competition_math", split="test", cache_dir=cache_dir)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    difficulty_map = {"Level 1": 1, "Level 2": 2, "Level 3": 3, "Level 4": 4, "Level 5": 5}

    predictions = []
    ground_truths = []
    difficulties = []
    subjects = []
    results = []

    for item in tqdm(dataset, desc="MATH"):
        problem = item["problem"]
        solution = item["solution"]
        level = item.get("level", "Level 3")
        subject = item.get("type", "unknown")

        # Extract ground truth from \\boxed{}
        import re
        gt_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        gt = gt_match.group(1) if gt_match else ""

        # Generate
        response = generate_answer(model, tokenizer, problem, device=device)
        pred = extract_answer_from_response(response)

        score, method = verify_answer(pred, gt)

        predictions.append(pred)
        ground_truths.append(gt)
        difficulties.append(difficulty_map.get(level, 3))
        subjects.append(subject)
        results.append({
            "problem": problem[:200],
            "prediction": pred,
            "ground_truth": gt,
            "correct": score > 0.5,
            "level": level,
            "subject": subject,
        })

    acc = accuracy(predictions, ground_truths)
    by_diff = accuracy_by_difficulty(predictions, ground_truths, difficulties)
    by_subj = accuracy_by_subject(predictions, ground_truths, subjects)

    print(f"[MATH] Overall accuracy: {acc:.2%}")
    print(f"\nBy difficulty:")
    for level, data in by_diff.items():
        print(f"  Level {level}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    print(f"\nBy subject:")
    for subject, data in by_subj.items():
        print(f"  {subject}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    return {
        "benchmark": "math",
        "accuracy": acc,
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "by_difficulty": by_diff,
        "by_subject": by_subj,
        "results": results,
    }


def run_all_evaluations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    benchmarks: List[str] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "./eval_results",
    device: str = "cuda",
) -> Dict:
    """Run all evaluation benchmarks and save results."""
    benchmarks = benchmarks or ["gsm8k", "math"]
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    if "gsm8k" in benchmarks:
        gsm_results = evaluate_gsm8k(model, tokenizer, max_samples, device)
        all_results["gsm8k"] = gsm_results

    if "math" in benchmarks:
        math_results = evaluate_math(model, tokenizer, max_samples, device)
        all_results["math"] = math_results

    # Save results
    summary = {}
    for bench, data in all_results.items():
        summary[bench] = {
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"],
        }
        if "by_difficulty" in data:
            summary[bench]["by_difficulty"] = data["by_difficulty"]

    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    for bench, data in summary.items():
        print(f"  {bench}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    print(f"{'='*60}")

    return all_results
