"""
Stage 3: Process Reward Model Trainer
Trains a step-level reward model on Math-Shepherd data.
"""

import os
import shutil
import torch
from typing import Optional
from transformers import (
    TrainingArguments,
    AutoTokenizer,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..models.reward_model import ProcessRewardModel
from ..data.collator import PRMDataCollator
from ..training.callbacks import MetricsLogger


def _save_to_drive(state_dict, dest_dir: str, filename: str):
    """Save model checkpoint reliably to Google Drive.

    Google Drive's FUSE mount in Colab caches writes, so large files
    may never fully sync. This saves locally first, then copies to Drive.
    """
    is_drive = "/content/drive" in dest_dir
    os.makedirs(dest_dir, exist_ok=True)

    if is_drive:
        local_tmp = os.path.join("/content", "prm_tmp_save")
        os.makedirs(local_tmp, exist_ok=True)
        local_file = os.path.join(local_tmp, filename)
        torch.save(state_dict, local_file)
        shutil.copy2(local_file, os.path.join(dest_dir, filename))
        os.remove(local_file)
    else:
        torch.save(state_dict, os.path.join(dest_dir, filename))


def run_prm_training(
    tokenizer: AutoTokenizer,
    train_dataset,
    config=None,
    output_dir: str = "./checkpoints/prm",
    device: str = "cuda",
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Run Stage 3: Process Reward Model Training.

    Uses a custom training loop (not HF Trainer) because the PRM
    has a non-standard architecture with a reward head.

    Args:
        tokenizer: Configured tokenizer
        train_dataset: PRM dataset with 'text' and 'label' columns
        config: PRMConfig instance
        output_dir: Where to save checkpoints
        device: Device to train on
        resume_from_checkpoint: Path to resume from

    Returns:
        Path to final checkpoint
    """
    from ..configs.base_config import PRMConfig
    if config is None:
        config = PRMConfig()

    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"[PRM] Resuming from {resume_from_checkpoint}")
        model = ProcessRewardModel(model_name=config.model_name, hidden_dim=config.reward_head_dim)
        state = torch.load(os.path.join(resume_from_checkpoint, "prm_model.pt"), map_location="cpu")
        model.load_state_dict(state)
    else:
        model = ProcessRewardModel(
            model_name=config.model_name,
            hidden_dim=config.reward_head_dim,
        )

    model.gradient_checkpointing_enable()
    model = model.to(device)

    # Data
    collator = PRMDataCollator(tokenizer=tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Mixed precision scaler — only use if model is in fp32
    model_dtype = next(model.parameters()).dtype
    use_fp16 = config.fp16 and model_dtype != torch.float16
    if not use_fp16 and config.fp16:
        print(f"  [NOTE] Model is already {model_dtype}, disabling fp16 GradScaler")
    scaler = torch.amp.GradScaler('cuda') if use_fp16 else None

    logger = MetricsLogger(use_wandb=bool(os.environ.get("WANDB_API_KEY")))

    print(f"\n{'='*60}")
    print(f"Stage 3: Process Reward Model Training")
    print(f"  Dataset: {len(train_dataset)} step examples")
    print(f"  Model params: {model.num_parameters() / 1e6:.1f}M")
    print(f"  Batch: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print(f"  LR: {config.learning_rate}, Epochs: {config.num_train_epochs}")
    print(f"{'='*60}\n")

    global_step = 0
    model.train()

    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(dataloader, desc=f"PRM Epoch {epoch + 1}/{config.num_train_epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_fp16:
                with torch.amp.autocast('cuda'):
                    result = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = result["loss"] / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                result = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = result["loss"] / config.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps

            # Accuracy tracking
            predictions = (result["rewards"] > 0.5).float()
            epoch_correct += (predictions == batch["labels"]).sum().item()
            epoch_total += batch["labels"].size(0)

            # Gradient accumulation step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
                    logger.log({
                        "prm/loss": avg_loss,
                        "prm/accuracy": accuracy,
                        "prm/lr": scheduler.get_last_lr()[0],
                        "prm/epoch": epoch + 1,
                    }, step=global_step)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.3f}")

                if global_step % config.save_steps == 0:
                    ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    _save_to_drive(model.state_dict(), ckpt_path, "prm_model.pt")

        # End of epoch
        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        print(f"\nEpoch {epoch + 1} complete: loss={avg_loss:.4f}, accuracy={accuracy:.3f}")

    # Save final model
    final_path = os.path.join(output_dir, "final")
    _save_to_drive(model.state_dict(), final_path, "prm_model.pt")
    # Also save tokenizer for easy reloading
    tokenizer.save_pretrained(final_path)

    # Verify file was written
    save_path = os.path.join(final_path, "prm_model.pt")
    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"\n[PRM] Final model saved to {final_path} ({size_mb:.1f} MB)")
    else:
        print(f"\n[PRM] WARNING: Save appeared to fail — {save_path} not found!")

    return final_path
