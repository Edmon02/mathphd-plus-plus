"""
Data Download Pipeline
Downloads all required datasets from HuggingFace Hub for each training stage.
"""

import os
from datasets import load_dataset, concatenate_datasets
from typing import Optional


def download_cpt_data(cache_dir: str = "./data_cache", max_samples: Optional[int] = None):
    """Download continued pre-training data: OpenWebMath + proof-pile subsets."""
    print("[CPT] Downloading OpenWebMath...")
    owm = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    print("[CPT] Downloading algebraic-stack (proof-pile math subset)...")
    proof = load_dataset(
        "EleutherAI/proof-pile-2",
        name="algebraic-stack",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    return {"open_web_math": owm, "proof_pile": proof}


def download_sft_data(cache_dir: str = "./data_cache"):
    """Download all SFT datasets."""
    datasets_dict = {}

    print("[SFT] Downloading MetaMathQA...")
    datasets_dict["metamath"] = load_dataset(
        "meta-math/MetaMathQA",
        split="train",
        cache_dir=cache_dir,
    )

    print("[SFT] Downloading MATH...")
    datasets_dict["math_train"] = load_dataset(
        "hendrycks/competition_math",
        split="train",
        cache_dir=cache_dir,
    )

    print("[SFT] Downloading GSM8K...")
    datasets_dict["gsm8k_train"] = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        cache_dir=cache_dir,
    )

    print("[SFT] Downloading NuminaMath-CoT...")
    datasets_dict["numina"] = load_dataset(
        "AI-MO/NuminaMath-CoT",
        split="train",
        cache_dir=cache_dir,
    )

    return datasets_dict


def download_prm_data(cache_dir: str = "./data_cache"):
    """Download PRM training data."""
    print("[PRM] Downloading math-shepherd (process reward data)...")
    prm_data = load_dataset(
        "peiyi9979/Math-Shepherd",
        split="train",
        cache_dir=cache_dir,
    )
    return prm_data


def download_eval_data(cache_dir: str = "./data_cache"):
    """Download evaluation benchmarks."""
    eval_datasets = {}

    print("[EVAL] Downloading MATH test...")
    eval_datasets["math_test"] = load_dataset(
        "hendrycks/competition_math",
        split="test",
        cache_dir=cache_dir,
    )

    print("[EVAL] Downloading GSM8K test...")
    eval_datasets["gsm8k_test"] = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        cache_dir=cache_dir,
    )

    return eval_datasets


def download_all(cache_dir: str = "./data_cache"):
    """Download all datasets for all stages."""
    os.makedirs(cache_dir, exist_ok=True)

    data = {
        "cpt": download_cpt_data(cache_dir),
        "sft": download_sft_data(cache_dir),
        "prm": download_prm_data(cache_dir),
        "eval": download_eval_data(cache_dir),
    }
    print("\n[DONE] All datasets downloaded.")
    return data


if __name__ == "__main__":
    download_all()
