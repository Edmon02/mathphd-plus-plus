"""
Continued Pre-Training Data Preprocessor
Handles: packing, curriculum ordering, structure annotation for multi-objective loss.
"""

import re
import torch
import numpy as np
from typing import Dict, List, Optional, Iterator
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm


# Regex patterns for math structure detection
THEOREM_PATTERNS = [
    r'(?i)(theorem|proposition|corollary)\s*[\d.]*\s*[.:]',
    r'\\begin\{theorem\}',
    r'\\begin\{proposition\}',
    r'\\begin\{corollary\}',
]

PROOF_PATTERNS = [
    r'(?i)proof\s*[.:]',
    r'\\begin\{proof\}',
]

DEFINITION_PATTERNS = [
    r'(?i)(definition|def\.)\s*[\d.]*\s*[.:]',
    r'\\begin\{definition\}',
]

LEMMA_PATTERNS = [
    r'(?i)lemma\s*[\d.]*\s*[.:]',
    r'\\begin\{lemma\}',
]


def detect_math_regions(text: str) -> List[Dict]:
    """Detect theorem/proof/definition/lemma regions in text.
    Returns list of (start_char, end_char, type) spans."""
    regions = []

    for pattern in THEOREM_PATTERNS:
        for m in re.finditer(pattern, text):
            regions.append({"start": m.start(), "end": min(m.start() + 2000, len(text)), "type": "theorem"})

    for pattern in PROOF_PATTERNS:
        for m in re.finditer(pattern, text):
            regions.append({"start": m.start(), "end": min(m.start() + 3000, len(text)), "type": "proof"})

    for pattern in DEFINITION_PATTERNS:
        for m in re.finditer(pattern, text):
            regions.append({"start": m.start(), "end": min(m.start() + 1000, len(text)), "type": "definition"})

    for pattern in LEMMA_PATTERNS:
        for m in re.finditer(pattern, text):
            regions.append({"start": m.start(), "end": min(m.start() + 2000, len(text)), "type": "lemma"})

    return regions


def create_structure_weight_mask(
    input_ids: torch.Tensor,
    char_to_token_map: List[Optional[int]],
    regions: List[Dict],
    upweight_factor: float = 2.0,
) -> torch.Tensor:
    """Create per-token loss weight mask that upweights structural regions.

    Args:
        input_ids: [seq_len] token ids
        char_to_token_map: mapping from char positions to token positions
        regions: detected math regions
        upweight_factor: weight multiplier for structural tokens

    Returns:
        [seq_len] float tensor of per-token weights (1.0 default, upweight_factor for structure)
    """
    weights = torch.ones(len(input_ids), dtype=torch.float32)

    for region in regions:
        for char_pos in range(region["start"], min(region["end"], len(char_to_token_map))):
            token_pos = char_to_token_map[char_pos]
            if token_pos is not None and token_pos < len(weights):
                weights[token_pos] = upweight_factor

    return weights


def pack_documents(
    texts: Iterator[str],
    tokenizer: AutoTokenizer,
    max_seq_length: int = 2048,
    max_chunks: Optional[int] = None,
) -> Dataset:
    """Pack tokenized documents into fixed-length chunks.

    Concatenates documents with EOS separator, then splits into chunks.
    No padding — every token contributes to training.

    Returns HuggingFace Dataset with columns: input_ids, attention_mask, labels, structure_weights
    """
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    buffer_ids = []
    buffer_weights = []
    all_chunks = []

    for text in texts:
        # Tokenize
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids = encoding["input_ids"]

        # Detect structural regions for weight mask
        regions = detect_math_regions(text)

        # Build char->token map from offset mapping
        char_to_token = [None] * (len(text) + 1)
        for token_idx, (start, end) in enumerate(encoding["offset_mapping"]):
            for c in range(start, end):
                if c < len(char_to_token):
                    char_to_token[c] = token_idx

        # Create per-token weights
        weights = create_structure_weight_mask(
            torch.tensor(ids), char_to_token, regions
        ).tolist()

        # Add to buffer with EOS separator
        buffer_ids.extend(ids + [eos_id])
        buffer_weights.extend(weights + [1.0])

        # Extract chunks when buffer is large enough
        while len(buffer_ids) >= max_seq_length:
            chunk_ids = buffer_ids[:max_seq_length]
            chunk_weights = buffer_weights[:max_seq_length]
            buffer_ids = buffer_ids[max_seq_length:]
            buffer_weights = buffer_weights[max_seq_length:]

            all_chunks.append({
                "input_ids": chunk_ids,
                "attention_mask": [1] * max_seq_length,
                "labels": chunk_ids.copy(),  # Shifted internally by model
                "structure_weights": chunk_weights,
            })

            if max_chunks and len(all_chunks) >= max_chunks:
                break

        if max_chunks and len(all_chunks) >= max_chunks:
            break

    return Dataset.from_list(all_chunks)


def estimate_difficulty(text: str) -> float:
    """Estimate document difficulty using heuristics.

    Returns float 0-1 where 0=elementary, 1=research.
    Used for curriculum ordering.
    """
    score = 0.0
    text_lower = text.lower()

    # Vocabulary complexity signals
    advanced_terms = [
        "cohomology", "sheaf", "functor", "homomorphism", "isomorphism",
        "manifold", "topology", "algebraic geometry", "spectral sequence",
        "etale", "scheme", "category theory", "homotopy", "fibration",
        "moduli", "galois", "abelian variety", "riemann surface",
        "stochastic", "martingale", "ergodic", "measure theory",
    ]
    intermediate_terms = [
        "theorem", "proof", "lemma", "corollary", "integral",
        "derivative", "eigenvalue", "vector space", "group theory",
        "ring", "field extension", "metric space", "convergence",
        "compactness", "continuity", "differentiable",
    ]
    elementary_terms = [
        "addition", "subtraction", "multiplication", "fraction",
        "equation", "solve for x", "linear equation", "quadratic",
    ]

    for term in advanced_terms:
        if term in text_lower:
            score += 0.15

    for term in intermediate_terms:
        if term in text_lower:
            score += 0.05

    for term in elementary_terms:
        if term in text_lower:
            score -= 0.1

    # Formula density
    formula_count = text.count("$") // 2 + text.count("\\begin{")
    words = len(text.split())
    if words > 0:
        formula_density = formula_count / words
        score += min(formula_density * 10, 0.3)

    # Citation density
    citation_count = len(re.findall(r'\[\d+\]', text)) + len(re.findall(r'\\cite\{', text))
    if words > 0:
        score += min(citation_count / words * 50, 0.2)

    return max(0.0, min(1.0, score))


def prepare_cpt_dataset(
    streaming_datasets: Dict,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 2048,
    target_tokens: int = 500_000_000,
    seed: int = 42,
) -> Dataset:
    """Full CPT data preparation pipeline.

    1. Sample from streaming datasets
    2. Estimate difficulty
    3. Sort by difficulty (curriculum)
    4. Pack into chunks with structure weights
    """
    max_chunks = target_tokens // max_seq_length
    print(f"[CPT] Target: {max_chunks} chunks of {max_seq_length} tokens = {target_tokens / 1e6:.0f}M tokens")

    # Interleave sources
    def text_iterator():
        owm_iter = iter(streaming_datasets["open_web_math"])
        proof_iter = iter(streaming_datasets["proof_pile"])
        count = 0
        while count < max_chunks * 2:  # Generate more than needed for packing
            try:
                item = next(owm_iter)
                yield item.get("text", "")
                count += 1
            except StopIteration:
                break
            try:
                item = next(proof_iter)
                yield item.get("text", "")
                count += 1
            except StopIteration:
                break

    print("[CPT] Packing documents into chunks...")
    dataset = pack_documents(
        text_iterator(),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_chunks=max_chunks,
    )

    print(f"[CPT] Created {len(dataset)} chunks")
    return dataset
