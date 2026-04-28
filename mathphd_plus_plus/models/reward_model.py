"""
Process Reward Model
Qwen2.5-0.5B with a classification head for step-level reward prediction.
"""

import math

import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProcessRewardModel(nn.Module):
    """Reward model that scores reasoning steps as correct/incorrect.

    Architecture: Qwen2.5-0.5B backbone + Linear(896, 1) reward head.
    The reward head replaces the language model head.
    Score is extracted at the last non-padding token position.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        hidden_dim: int = 896,
        torch_dtype: str = "float16",
        pos_weight: Optional[float] = None,
    ):
        super().__init__()

        dtype = getattr(torch, torch_dtype)

        # Load backbone (without LM head)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        # Replace LM head with reward head
        # Access the underlying transformer model
        if hasattr(self.backbone, 'model'):
            self.transformer = self.backbone.model
        elif hasattr(self.backbone, 'transformer'):
            self.transformer = self.backbone.transformer
        else:
            self.transformer = self.backbone

        self.reward_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.normal_(self.reward_head.weight, std=0.01)

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=dtype)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] float tensor (0 or 1)

        Returns:
            dict with 'rewards' [batch_size] and optionally 'loss' scalar
        """
        # Get hidden states from backbone
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state

        # Extract last non-padding token position for each sequence
        # Shape: [batch_size]
        seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lengths]  # [batch_size, hidden_dim]

        # Compute reward scores
        reward_logits = self.reward_head(last_hidden).squeeze(-1)  # [batch_size]
        rewards = torch.sigmoid(reward_logits).clamp(0.0, 1.0)

        # Guard against NaN from degenerate hidden states
        rewards = torch.where(torch.isnan(rewards), torch.tensor(0.5, device=rewards.device, dtype=rewards.dtype), rewards)

        result = {"rewards": rewards, "logits": reward_logits}

        if labels is not None:
            result["loss"] = self.loss_fn(reward_logits, labels)

        return result

    def score_steps(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        step_delimiter: str = "\n",
        device: str = "cuda",
    ) -> list:
        """Score each step in a solution.

        Splits solution by step_delimiter and scores each prefix.
        Returns list of (step_text, score) tuples.
        """
        self.eval()
        steps = [s.strip() for s in text.split(step_delimiter) if s.strip()]
        if not steps:
            return []

        scores = []

        prefix = ""
        for step in steps:
            prefix += step + step_delimiter
            encoding = tokenizer(
                prefix,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            with torch.no_grad():
                try:
                    result = self.forward(**encoding)
                    score = result["rewards"].item()
                    if math.isnan(score) or math.isinf(score):
                        score = 0.5
                except Exception:
                    score = 0.5

            scores.append((step, score))

        return scores

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing on backbone and disable use_cache."""
        if hasattr(self.transformer, 'config'):
            self.transformer.config.use_cache = False
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
