"""
Custom Data Collators
Implements response-only loss masking and structure-weighted loss masking.
"""

import torch
from typing import Dict, List, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


@dataclass
class CPTDataCollator:
    """Collator for Continued Pre-Training with structure weights.

    Pads batch and returns structure_weights for multi-objective loss.
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "structure_weights": [],
        }

        for f in features:
            batch["input_ids"].append(torch.tensor(f["input_ids"][:self.max_length], dtype=torch.long))
            batch["attention_mask"].append(torch.tensor(f["attention_mask"][:self.max_length], dtype=torch.long))
            batch["labels"].append(torch.tensor(f["labels"][:self.max_length], dtype=torch.long))
            batch["structure_weights"].append(
                torch.tensor(f["structure_weights"][:self.max_length], dtype=torch.float32)
            )

        # Stack (all same length due to packing)
        return {k: torch.stack(v) for k, v in batch.items()}


@dataclass
class SFTDataCollator:
    """Collator for SFT with response-only loss masking.

    Masks system and user tokens so loss is only computed on assistant responses.
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 1024
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Create labels with prompt masking
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = self.label_pad_token_id

        # Mask non-response tokens
        # Find assistant response boundaries in each sequence
        assistant_start_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

        for batch_idx in range(input_ids.size(0)):
            ids = input_ids[batch_idx].tolist()

            # Find all assistant response regions
            in_response = False
            for i in range(len(ids)):
                # Check for assistant start
                if (i + len(assistant_start_ids) <= len(ids) and
                        ids[i:i + len(assistant_start_ids)] == assistant_start_ids):
                    # Mask the start tokens themselves
                    for j in range(len(assistant_start_ids)):
                        if i + j < len(labels[batch_idx]):
                            labels[batch_idx][i + j] = self.label_pad_token_id
                    in_response = True
                    continue

                # Check for end of response
                if in_response and (i + len(im_end_ids) <= len(ids) and
                        ids[i:i + len(im_end_ids)] == im_end_ids):
                    in_response = False
                    continue

                # Mask non-response tokens
                if not in_response:
                    labels[batch_idx][i] = self.label_pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class PRMDataCollator:
    """Collator for Process Reward Model training."""
    tokenizer: PreTrainedTokenizer
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        labels = [f["label"] for f in features]

        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
