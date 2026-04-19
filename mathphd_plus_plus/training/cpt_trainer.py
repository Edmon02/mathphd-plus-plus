"""
Stage 1: Continued Pre-Training Trainer
Implements multi-objective pre-training with curriculum learning.
"""

import os
import torch
from typing import Optional, Dict
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from torch.utils.data import DataLoader

from ..models.multi_objective_loss import MultiObjectiveLoss
from ..data.collator import CPTDataCollator
from ..training.callbacks import ColabCheckpointCallback, MetricsLogger


class CPTTrainer(Trainer):
    """Custom trainer for continued pre-training with multi-objective loss.

    Overrides compute_loss to use L_total = L_NTP + alpha * L_structure.
    """

    def __init__(self, structure_loss_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.multi_objective_loss = MultiObjectiveLoss(alpha=structure_loss_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to use multi-objective loss."""
        # Extract structure weights before passing to model
        structure_weights = inputs.pop("structure_weights", None)

        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Compute multi-objective loss
        loss_dict = self.multi_objective_loss(
            logits=outputs.logits,
            labels=inputs["labels"],
            structure_weights=structure_weights,
            attention_mask=inputs.get("attention_mask"),
        )

        loss = loss_dict["loss"]

        if return_outputs:
            return loss, outputs
        return loss


def run_cpt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    config=None,
    output_dir: str = "./checkpoints/cpt",
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Run Stage 1: Continued Pre-Training.

    Args:
        model: Pre-loaded base model
        tokenizer: Configured tokenizer
        train_dataset: Packed CPT dataset from preprocess_cpt
        config: CPTConfig instance
        output_dir: Where to save checkpoints
        resume_from_checkpoint: Path to resume from

    Returns:
        Path to final checkpoint
    """
    from ..configs.base_config import CPTConfig
    if config is None:
        config = CPTConfig()

    # Detect model dtype — only use fp16 mixed precision if model is in fp32
    model_dtype = next(model.parameters()).dtype
    use_fp16 = config.fp16 and model_dtype != torch.float16

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        fp16=use_fp16,
        bf16=config.bf16,
        gradient_checkpointing=True,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=3,
        remove_unused_columns=False,  # Keep structure_weights
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )

    collator = CPTDataCollator(tokenizer=tokenizer, max_length=config.max_seq_length)

    trainer = CPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        structure_loss_weight=config.structure_loss_weight,
        callbacks=[ColabCheckpointCallback()] if os.path.exists("/content/drive") else [],
    )

    print(f"\n{'='*60}")
    print(f"Stage 1: Continued Pre-Training")
    print(f"  Dataset: {len(train_dataset)} chunks")
    print(f"  Batch: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  LR: {config.learning_rate}, Epochs: {config.num_train_epochs}")
    print(f"  Structure loss weight: {config.structure_loss_weight}")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[CPT] Final model saved to {final_path}")

    return final_path
