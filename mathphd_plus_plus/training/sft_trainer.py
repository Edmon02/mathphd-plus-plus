"""
Stage 2: Supervised Fine-Tuning Trainer
Fine-tunes the CPT model on structured math instruction data.
"""

import os
import torch
from typing import Optional
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from ..data.collator import SFTDataCollator
from ..training.callbacks import ColabCheckpointCallback


def run_sft(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    eval_dataset=None,
    config=None,
    output_dir: str = "./checkpoints/sft",
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Run Stage 2: Supervised Fine-Tuning.

    Args:
        model: Model from CPT stage (or base model)
        tokenizer: Configured tokenizer
        train_dataset: SFT dataset with 'text' column in ChatML format
        eval_dataset: Optional validation set
        config: SFTConfig instance
        output_dir: Where to save checkpoints
        resume_from_checkpoint: Path to resume from

    Returns:
        Path to final checkpoint
    """
    from ..configs.base_config import SFTConfig
    if config is None:
        config = SFTConfig()

    # Detect model dtype — only use fp16 mixed precision if model is in fp32
    model_dtype = next(model.parameters()).dtype
    use_fp16 = config.fp16 and model_dtype != torch.float16
    if not use_fp16 and config.fp16:
        print(f"  [NOTE] Model is already {model_dtype}, disabling fp16 GradScaler")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=use_fp16,
        bf16=config.bf16,
        gradient_checkpointing=True,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.save_steps if eval_dataset else None,
        dataloader_pin_memory=True,
    )

    collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[ColabCheckpointCallback()] if os.path.exists("/content/drive") else [],
    )

    print(f"\n{'='*60}")
    print(f"Stage 2: Supervised Fine-Tuning")
    print(f"  Dataset: {len(train_dataset)} samples")
    print(f"  Batch: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print(f"  LR: {config.learning_rate}, Epochs: {config.num_train_epochs}")
    print(f"  Response-only masking: {config.mask_prompt_tokens}")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[SFT] Final model saved to {final_path}")

    return final_path
