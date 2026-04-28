"""
Process Reward Model Data Preprocessor
Extracts step-level labels from Math-Shepherd dataset.
"""

import re
from typing import Dict, List
from datasets import Dataset


STEP_BLOCK_RE = re.compile(r'(?ms)(Step\s+\d+:.*?)(?=\n\s*Step\s+\d+:|\Z)')
STEP_START_RE = re.compile(r'(?m)^\s*Step\s+\d+:')
LABEL_SUFFIX_RE = re.compile(r'\s*([+-])\s*$')


def _extract_prompt_prefix(text: str) -> str:
    """Return the problem statement before the first numbered step."""
    if not text:
        return ""

    match = STEP_START_RE.search(text)
    if not match:
        return text.strip()

    return text[:match.start()].strip()


def parse_math_shepherd(item: Dict) -> List[Dict]:
    """Parse a Math-Shepherd item into step-level training examples.

    Math-Shepherd format:
    - Each step ends with a label: "Step N: ... ки" followed by a label
    - Labels: '+' (correct), '-' (incorrect)

    Returns list of {input_text, step_text, label} dicts.
    """
    input_text = item.get("input", "")
    labeled_text = item.get("label", "")
    prompt_text = _extract_prompt_prefix(input_text or labeled_text)

    if not labeled_text:
        return []

    steps_with_labels = []

    for step_index, block in enumerate(STEP_BLOCK_RE.findall(labeled_text), start=1):
        step_block = block.strip()
        label_match = LABEL_SUFFIX_RE.search(step_block)
        if label_match is None:
            continue

        step_text = step_block[:label_match.start()].rstrip()
        step_text = re.sub(r'\s*ки\s*$', '', step_text).strip()
        if not step_text:
            continue

        steps_with_labels.append({
            "prompt_text": prompt_text,
            "step_text": step_text,
            "label": 1 if label_match.group(1) == "+" else 0,
            "step_index": step_index,
        })

    return steps_with_labels


def split_into_steps(solution: str) -> List[str]:
    """Split a solution into reasoning steps using heuristics."""
    # Try numbered steps first
    numbered = re.split(r'\n(?=Step \d|(?:\d+[\.\)])\s)', solution)
    if len(numbered) > 1:
        return [s.strip() for s in numbered if s.strip()]

    # Fall back to paragraph-based splitting
    paragraphs = solution.split("\n\n")
    if len(paragraphs) > 1:
        return [p.strip() for p in paragraphs if p.strip()]

    # Fall back to sentence-based splitting for short solutions
    sentences = re.split(r'(?<=[.!?])\s+', solution)
    steps = []
    current_step = ""
    for sent in sentences:
        current_step += sent + " "
        if len(current_step.split()) >= 20:  # ~20 words per step
            steps.append(current_step.strip())
            current_step = ""
    if current_step.strip():
        steps.append(current_step.strip())

    return steps if steps else [solution]


def prepare_prm_dataset(
    prm_raw_data,
    max_samples: int = 100_000,
) -> Dataset:
    """Full PRM data preparation pipeline.

    Processes Math-Shepherd data into step-level classification examples.
    Each example: prefix (all text up to step boundary) → label (0 or 1).
    """
    examples = []

    for idx, item in enumerate(prm_raw_data):
        if idx >= max_samples * 5:  # Process more items since many get filtered
            break

        steps = parse_math_shepherd(item)
        if not steps:
            continue

        prompt_text = steps[0].get("prompt_text", "")
        prefix_parts = [prompt_text] if prompt_text else []
        for step_info in steps:
            step_text = step_info["step_text"]
            label = step_info["label"]
            prefix_parts.append(step_text)
            prefix_text = "\n\n".join(part for part in prefix_parts if part).strip()

            examples.append({
                "text": prefix_text,
                "label": label,
                "num_steps": step_info.get("step_index", len(prefix_parts) - 1),
                "step_text": step_text,
                "task": item.get("task", "unknown"),
            })

        if len(examples) >= max_samples:
            break

    examples = examples[:max_samples]
    positive = sum(1 for e in examples if e['label'] == 1)
    negative = sum(1 for e in examples if e['label'] == 0)

    print(f"[PRM] Created {len(examples)} step-level examples")
    print(f"  Positive (correct steps): {positive}")
    print(f"  Negative (incorrect steps): {negative}")

    if not examples:
        raise ValueError("PRM preprocessing produced no training examples.")
    if positive == 0 or negative == 0:
        raise ValueError(
            "PRM preprocessing produced a degenerate label distribution. "
            f"Positive={positive}, Negative={negative}."
        )

    return Dataset.from_list(examples)
