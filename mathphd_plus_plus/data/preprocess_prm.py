"""
Process Reward Model Data Preprocessor
Extracts step-level labels from Math-Shepherd dataset.
"""

import re
from typing import Dict, List
from datasets import Dataset


STEP_DELIMITERS = [
    "\nStep ", "\n\nStep ", "\nstep ",
    "\n\n", "\nTherefore", "\nThus",
    "\nSo,", "\nHence",
]


def parse_math_shepherd(item: Dict) -> List[Dict]:
    """Parse a Math-Shepherd item into step-level training examples.

    Math-Shepherd format:
    - Each step ends with a label: "Step N: ... ки" followed by a label
    - Labels: '+' (correct), '-' (incorrect)

    Returns list of {input_text, step_text, label} dicts.
    """
    text = item.get("input", "")
    label = item.get("label", "")

    if not text:
        return []

    # Math-Shepherd uses ки as step delimiter with +/- labels
    steps_with_labels = []

    # Split by step markers
    parts = re.split(r'(Step \d+:)', text)

    current_text = ""
    for part in parts:
        if re.match(r'Step \d+:', part):
            if current_text.strip():
                # Check for label markers
                if "ки" in current_text:
                    step_text = current_text.split("ки")[0].strip()
                    # Look for + or - after ки
                    after_ki = current_text.split("ки")[-1].strip()
                    lbl = 1 if "+" in after_ki or not after_ki else 0
                    steps_with_labels.append({
                        "step_text": step_text,
                        "label": lbl,
                    })
            current_text = part
        else:
            current_text += part

    # Handle last step
    if current_text.strip():
        if "ки" in current_text:
            step_text = current_text.split("ки")[0].strip()
            after_ki = current_text.split("ки")[-1].strip()
            lbl = 1 if "+" in after_ki or not after_ki else 0
            steps_with_labels.append({
                "step_text": step_text,
                "label": lbl,
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

        # Build cumulative context
        prefix = ""
        for step_info in steps:
            step_text = step_info["step_text"]
            label = step_info["label"]
            prefix += step_text + "\n"

            examples.append({
                "text": prefix.strip(),
                "label": label,
                "num_steps": len(prefix.split("\n")),
            })

        if len(examples) >= max_samples:
            break

    examples = examples[:max_samples]
    print(f"[PRM] Created {len(examples)} step-level examples")
    print(f"  Positive (correct steps): {sum(1 for e in examples if e['label'] == 1)}")
    print(f"  Negative (incorrect steps): {sum(1 for e in examples if e['label'] == 0)}")

    return Dataset.from_list(examples)
