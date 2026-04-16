"""
Multi-Objective Loss for Continued Pre-Training
Implements L_total = L_NTP + alpha * L_structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiObjectiveLoss(nn.Module):
    """Combined NTP + Structure-Weighted loss for mathematical pre-training.

    L_total = L_NTP + alpha * L_structure

    Where:
    - L_NTP: Standard cross-entropy next-token prediction
    - L_structure: Same cross-entropy but with per-token weights that
      upweight tokens inside theorem/proof/definition/lemma regions.

    The structure loss encourages the model to pay extra attention to
    mathematically significant text regions.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        labels: torch.Tensor,  # [batch, seq_len]
        structure_weights: Optional[torch.Tensor] = None,  # [batch, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-objective loss.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            labels: Target token IDs [batch, seq_len]
            structure_weights: Per-token weights for structural regions [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            dict with 'loss' (total), 'ntp_loss', and 'structure_loss'
        """
        # Shift for next-token prediction
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape

        # Flatten for cross-entropy
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # L_NTP: Standard unweighted cross-entropy
        ntp_loss = F.cross_entropy(flat_logits, flat_labels, reduction='mean')

        # L_structure: Weighted cross-entropy
        if structure_weights is not None and self.alpha > 0:
            shift_weights = structure_weights[:, 1:].contiguous()

            # Apply mask
            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous().float()
                shift_weights = shift_weights * shift_mask

            flat_weights = shift_weights.view(-1)

            # Per-token loss (unreduced)
            per_token_loss = F.cross_entropy(
                flat_logits, flat_labels, reduction='none'
            )

            # Weighted mean
            structure_loss = (per_token_loss * flat_weights).sum() / (flat_weights.sum() + 1e-8)

            total_loss = ntp_loss + self.alpha * structure_loss
        else:
            structure_loss = torch.tensor(0.0, device=logits.device)
            total_loss = ntp_loss

        return {
            "loss": total_loss,
            "ntp_loss": ntp_loss.detach(),
            "structure_loss": structure_loss.detach() if isinstance(structure_loss, torch.Tensor) else structure_loss,
        }


class WeightedCrossEntropyLoss(nn.Module):
    """Simple weighted cross-entropy for SFT with response-only masking.

    Labels with value -100 are ignored (standard PyTorch convention).
    """

    def forward(
        self,
        logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        labels: torch.Tensor,  # [batch, seq_len] with -100 for masked positions
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
