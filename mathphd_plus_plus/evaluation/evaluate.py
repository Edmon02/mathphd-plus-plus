"""
Evaluation Runner
Run benchmarks (GSM8K, MATH, AIME) and produce detailed reports.
"""

import os
import json
import torch
from typing import Dict, Optional, List
from contextlib import nullcontext
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.preprocess_sft import SYSTEM_PROMPT
from ..rewards.sympy_verifier import (
    extract_answer_candidates,
    extract_answer_from_response,
    verify_answer,
)
from .metrics import accuracy, accuracy_by_difficulty, accuracy_by_subject


def load_math_eval_dataset(
    max_samples: Optional[int] = None,
    cache_dir: str = "./data_cache",
):
    """Load a standard-format MATH evaluation dataset without loading-script fallbacks."""
    dataset = None
    dataset_name = None
    load_errors = {}

    # Use MATH-500 for quick runs and the full parquet-backed MATH set otherwise.
    dataset_candidates = [
        ("HuggingFaceH4/MATH-500", None),
        ("DigitalLearningGmbH/MATH-lighteval", "default"),
    ]
    if max_samples is None or max_samples > 500:
        dataset_candidates.reverse()

    for candidate_name, candidate_config in dataset_candidates:
        try:
            dataset = load_dataset(
                candidate_name,
                candidate_config,
                split="test",
                cache_dir=cache_dir,
            )
            dataset_name = candidate_name
            break
        except Exception as exc:
            load_errors[candidate_name] = str(exc)

    return dataset, dataset_name, load_errors


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.95,
    system_prompt: str = SYSTEM_PROMPT,
    device: str = "cuda",
) -> str:
    """Generate a single answer using greedy or sampled decoding."""
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

    do_sample = temperature > 0
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
        repetition_penalty=1.2,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    model.eval()
    autocast_ctx = torch.amp.autocast('cuda') if str(device).startswith('cuda') else nullcontext()
    with torch.no_grad():
        with autocast_ctx:
            output = model.generate(**gen_kwargs)

    response = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
    return response


def evaluate_gsm8k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.95,
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
        gt = solution.rsplit("####", 1)[-1].strip() if "####" in solution else solution.strip()

        # Generate
        response = generate_answer(
            model,
            tokenizer,
            problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        candidates = extract_answer_candidates(response)
        pred = candidates[0]["answer"] if candidates else extract_answer_from_response(response)

        score, method = verify_answer(pred, gt)

        predictions.append(pred)
        ground_truths.append(gt)
        results.append({
            "problem": problem,
            "prediction": pred,
            "ground_truth": gt,
            "correct": score > 0.5,
            "verification_score": score,
            "method": method,
            "prediction_candidates": candidates,
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
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.95,
    device: str = "cuda",
    cache_dir: str = "./data_cache",
) -> Dict:
    """Evaluate on MATH test set with difficulty and subject breakdown."""
    print("[EVAL] Running MATH evaluation...")

    dataset, dataset_name, load_errors = load_math_eval_dataset(
        max_samples=max_samples,
        cache_dir=cache_dir,
    )

    if dataset is None:
        print("[EVAL] WARNING: Could not load MATH dataset, skipping...")
        return {
            "benchmark": "math",
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "results": [],
            "error": "Dataset not available",
            "load_errors": load_errors,
        }

    print(f"[EVAL] Loaded MATH from: {dataset_name}")
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
        solution = item.get("solution", item.get("answer", ""))
        level = item.get("level", item.get("difficulty", "Level 3"))
        subject = item.get("type", item.get("subject", "unknown"))

        solution_candidates = extract_answer_candidates(solution)
        gt = solution_candidates[0]["answer"] if solution_candidates else ""

        # Generate
        response = generate_answer(
            model,
            tokenizer,
            problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        candidates = extract_answer_candidates(response)
        pred = candidates[0]["answer"] if candidates else extract_answer_from_response(response)

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
            "verification_score": score,
            "verification_method": method,
            "level": level,
            "subject": subject,
            "prediction_candidates": candidates,
            "ground_truth_candidates": solution_candidates,
            "full_response": response,
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
        "dataset_name": dataset_name,
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
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.95,
    device: str = "cuda",
) -> Dict:
    """Run all evaluation benchmarks and save results."""
    benchmarks = benchmarks or ["gsm8k", "math"]
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    if "gsm8k" in benchmarks:
        gsm_results = evaluate_gsm8k(
            model,
            tokenizer,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        all_results["gsm8k"] = gsm_results

    if "math" in benchmarks:
        math_results = evaluate_math(
            model,
            tokenizer,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
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

    for bench, data in all_results.items():
        with open(os.path.join(output_dir, f"{bench}_results.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    for bench, data in summary.items():
        print(f"  {bench}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    print(f"{'='*60}")

    return all_results
