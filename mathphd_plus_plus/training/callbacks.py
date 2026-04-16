"""
Training Callbacks
Handles checkpointing, crash recovery, and logging for Google Colab.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


class ColabCheckpointCallback(TrainerCallback):
    """Saves checkpoints to Google Drive for crash recovery.

    On Colab, sessions can die at any time. This callback:
    1. Saves to local checkpoint dir normally
    2. Copies best/latest checkpoint to Google Drive
    3. Saves training state (step, epoch, metrics) for resume
    """

    def __init__(
        self,
        gdrive_path: str = "/content/drive/MyDrive/mathphd_checkpoints",
        save_best: bool = True,
        metric_for_best: str = "loss",
    ):
        self.gdrive_path = gdrive_path
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.best_metric = float('inf')

        os.makedirs(gdrive_path, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        """Copy checkpoint to Google Drive after each save."""
        if not os.path.exists(self.gdrive_path):
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            return

        # Save training state for resume
        state_file = os.path.join(self.gdrive_path, "training_state.json")
        state_dict = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": self.best_metric,
            "checkpoint_dir": checkpoint_dir,
            "timestamp": time.time(),
        }
        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)

        # Copy latest checkpoint to gdrive
        gdrive_latest = os.path.join(self.gdrive_path, "latest")
        if os.path.exists(gdrive_latest):
            shutil.rmtree(gdrive_latest)
        shutil.copytree(checkpoint_dir, gdrive_latest)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track best metric for best-model saving."""
        if logs and self.metric_for_best in logs:
            current = logs[self.metric_for_best]
            if current < self.best_metric:
                self.best_metric = current


class GRPOProgressCallback(TrainerCallback):
    """Tracks GRPO training progress for crash recovery.

    Saves the current problem index so training can resume
    from where it stopped, not from the beginning.
    """

    def __init__(self, progress_file: str = "./grpo_progress.json"):
        self.progress_file = progress_file
        self.current_problem_idx = 0
        self.rewards_history = []

        # Load existing progress
        if os.path.exists(progress_file):
            with open(progress_file) as f:
                data = json.load(f)
                self.current_problem_idx = data.get("problem_idx", 0)
                self.rewards_history = data.get("rewards_history", [])

    def update(self, problem_idx: int, reward_info: dict):
        """Update progress after processing a problem."""
        self.current_problem_idx = problem_idx
        self.rewards_history.append({
            "idx": problem_idx,
            "reward": reward_info.get("total", 0),
            "correctness": reward_info.get("correctness", 0),
            "timestamp": time.time(),
        })

        # Save periodically
        if problem_idx % 10 == 0:
            self.save()

    def save(self):
        """Save progress to file."""
        with open(self.progress_file, "w") as f:
            json.dump({
                "problem_idx": self.current_problem_idx,
                "rewards_history": self.rewards_history[-1000:],  # Keep last 1000
                "timestamp": time.time(),
            }, f, indent=2)

    def get_resume_idx(self) -> int:
        """Get the problem index to resume from."""
        return self.current_problem_idx


class MetricsLogger:
    """Simple metrics logging compatible with both wandb and stdout."""

    def __init__(self, use_wandb: bool = True, project: str = "mathphd-plus-plus"):
        self.use_wandb = use_wandb
        self.step = 0

        if use_wandb:
            try:
                import wandb
                if not wandb.run:
                    wandb.init(project=project)
                self.wandb = wandb
            except ImportError:
                self.use_wandb = False
                print("[WARNING] wandb not available, logging to stdout only")

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics."""
        if step is not None:
            self.step = step

        if self.use_wandb:
            self.wandb.log(metrics, step=self.step)

        # Always print summary
        metrics_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                 for k, v in metrics.items())
        print(f"[Step {self.step}] {metrics_str}")
        self.step += 1
