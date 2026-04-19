"""
Stage 4: GRPO (Group Relative Policy Optimization) Trainer
The core RL training loop with verifiable rewards.

Algorithm:
  For each problem x:
    1. Generate G solutions from current policy
    2. Compute rewards (SymPy correctness + PRM process + format)
    3. Compute group-normalized advantages
    4. Update policy with clipped ratio + KL penalty
"""

import os
import copy
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards.composite_reward import CompositeReward
from ..training.callbacks import GRPOProgressCallback, MetricsLogger


class GRPOTrainer:
    """GRPO Trainer with verifiable math rewards.

    Implements the full GRPO algorithm:
    - Group generation (G=4 solutions per problem)
    - Composite reward computation
    - Group-normalized advantage estimation
    - Clipped policy gradient with KL penalty
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,  # Frozen reference (SFT checkpoint)
        tokenizer: AutoTokenizer,
        reward_fn: CompositeReward,
        config=None,
        device: str = "cuda",
    ):
        from ..configs.base_config import GRPOConfig
        self.config = config or GRPOConfig()

        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Mixed precision — only use GradScaler if model is in fp32
        model_dtype = next(self.model.parameters()).dtype
        self.use_fp16 = self.config.fp16 and model_dtype != torch.float16
        if not self.use_fp16 and self.config.fp16:
            print(f"  [NOTE] Model is already {model_dtype}, disabling fp16 GradScaler")
        self.scaler = torch.amp.GradScaler('cuda') if self.use_fp16 else None

        # Logging
        self.logger = MetricsLogger(use_wandb=bool(os.environ.get("WANDB_API_KEY")))
        self.progress = GRPOProgressCallback(
            progress_file=os.path.join(self.config.output_dir, "grpo_progress.json")
        )

    @torch.no_grad()
    def generate_solutions(
        self,
        problem: str,
        num_solutions: int = 4,
    ) -> List[Dict]:
        """Generate G solutions for a problem and store log-probs.

        Returns list of dicts with 'text', 'input_ids', 'log_probs' per solution.
        """
        self.model.eval()

        # Encode problem with system prompt
        prompt = (
            f"<|im_start|>system\nYou are MathPhD++, an advanced mathematical reasoning assistant. "
            f"Show your complete reasoning step-by-step. Verify your answer.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.size(1)

        solutions = []

        for _ in range(num_solutions):
            # Generate with sampling
            with torch.amp.autocast('cuda', enabled=self.use_fp16):
                output = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.generation_temperature,
                    top_p=self.config.generation_top_p,
                    do_sample=self.config.generation_do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = output.sequences[0]
            response_ids = generated_ids[prompt_len:]

            # Compute log probs of generated tokens
            all_scores = torch.stack(output.scores, dim=0)  # [gen_len, 1, vocab_size]
            all_log_probs = F.log_softmax(all_scores[:, 0, :], dim=-1)  # [gen_len, vocab_size]

            # Extract log probs of chosen tokens
            token_log_probs = all_log_probs[
                torch.arange(len(response_ids)),
                response_ids,
            ]  # [gen_len]

            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            solutions.append({
                "text": response_text,
                "full_ids": generated_ids,
                "response_ids": response_ids,
                "old_log_probs": token_log_probs.cpu(),  # Store on CPU
                "prompt_len": prompt_len,
            })

        self.model.train()
        return solutions

    def compute_log_probs(
        self,
        model: AutoModelForCausalLM,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute per-token log probs for response tokens under given model.

        Args:
            model: Model to evaluate
            full_ids: [seq_len] full sequence (prompt + response)
            prompt_len: Length of prompt prefix

        Returns:
            [response_len] log probs tensor
        """
        input_ids = full_ids.unsqueeze(0).to(self.device)  # [1, seq_len]

        with torch.amp.autocast('cuda', enabled=self.use_fp16):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Compute log probs for response tokens
        # logits[t] predicts token t+1
        response_logits = logits[prompt_len - 1:-1]  # [response_len, vocab_size]
        response_ids = full_ids[prompt_len:]  # [response_len]

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(len(response_ids)),
            response_ids.to(self.device),
        ]

        return token_log_probs

    def compute_grpo_loss(
        self,
        problem: str,
        ground_truth: str,
    ) -> Dict:
        """Compute GRPO loss for a single problem.

        1. Generate G solutions
        2. Compute rewards
        3. Compute advantages
        4. Compute clipped policy gradient + KL penalty
        """
        # 1. Generate solutions
        solutions = self.generate_solutions(problem, self.config.group_size)

        # 2. Compute rewards
        rewards = []
        for sol in solutions:
            reward_info = self.reward_fn.compute_reward(sol["text"], ground_truth)
            rewards.append(reward_info)
            sol["reward"] = reward_info

        total_rewards = [r["total"] for r in rewards]

        # 3. Compute group-normalized advantages
        mean_r = np.mean(total_rewards)
        std_r = np.std(total_rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in total_rewards]

        # Skip if no variance (all same reward)
        if std_r < 1e-6:
            return {
                "loss": torch.tensor(0.0, device=self.device),
                "mean_reward": mean_r,
                "rewards": rewards,
                "skipped": True,
            }

        # 4. Compute GRPO loss
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for sol, adv in zip(solutions, advantages):
            # Current policy log probs
            current_log_probs = self.compute_log_probs(
                self.model,
                sol["full_ids"],
                sol["prompt_len"],
            )

            # Reference policy log probs (for KL penalty)
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(
                    self.ref_model,
                    sol["full_ids"],
                    sol["prompt_len"],
                )

            # Old policy log probs (from generation time)
            old_log_probs = sol["old_log_probs"].to(self.device)

            # Ensure same length (might differ due to truncation)
            min_len = min(len(current_log_probs), len(old_log_probs), len(ref_log_probs))
            current_log_probs = current_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]
            ref_log_probs = ref_log_probs[:min_len]

            # Log ratio: log(pi_theta / pi_old)
            log_ratio = current_log_probs - old_log_probs
            ratio = torch.exp(log_ratio.sum())  # Product of per-token ratios

            # Clipped ratio
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon,
            )

            # Policy gradient loss (negative because we maximize reward)
            adv_tensor = torch.tensor(adv, device=self.device, dtype=torch.float32)
            pg_loss = -torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor)

            # KL penalty
            kl = (torch.exp(current_log_probs) * (current_log_probs - ref_log_probs)).sum()

            sol_loss = pg_loss + self.config.kl_beta * kl
            total_loss = total_loss + sol_loss

        total_loss = total_loss / len(solutions)

        return {
            "loss": total_loss,
            "mean_reward": mean_r,
            "rewards": rewards,
            "advantages": advantages,
            "skipped": False,
        }

    def train(
        self,
        train_dataset,
        num_epochs: Optional[int] = None,
    ):
        """Main GRPO training loop.

        Args:
            train_dataset: Dataset with 'problem' and 'answer' columns
            num_epochs: Override config num_grpo_epochs
        """
        num_epochs = num_epochs or self.config.num_grpo_epochs
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Resume from last problem if crash recovery
        start_idx = self.progress.get_resume_idx()
        if start_idx > 0:
            print(f"[GRPO] Resuming from problem {start_idx}")

        print(f"\n{'='*60}")
        print(f"Stage 4: GRPO Training")
        print(f"  Problems: {len(train_dataset)}")
        print(f"  Group size: {self.config.group_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Clip epsilon: {self.config.clip_epsilon}")
        print(f"  KL beta: {self.config.kl_beta}")
        print(f"  LR: {self.config.learning_rate}")
        print(f"{'='*60}\n")

        global_step = 0
        accumulated_loss = 0.0

        for epoch in range(num_epochs):
            epoch_rewards = []
            epoch_correct = 0
            epoch_total = 0

            pbar = tqdm(
                range(start_idx, min(len(train_dataset), self.config.problems_per_epoch)),
                desc=f"GRPO Epoch {epoch + 1}/{num_epochs}",
            )

            self.optimizer.zero_grad()

            for problem_idx in pbar:
                item = train_dataset[problem_idx]
                problem = item["problem"]
                answer = item["answer"]

                # Compute GRPO loss
                result = self.compute_grpo_loss(problem, answer)

                if not result["skipped"]:
                    loss = result["loss"] / self.config.gradient_accumulation_steps

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    accumulated_loss += loss.item() * self.config.gradient_accumulation_steps

                # Track metrics
                epoch_rewards.append(result["mean_reward"])
                for r in result["rewards"]:
                    epoch_total += 1
                    if r["correctness"] > 0.5:
                        epoch_correct += 1

                # Gradient step
                if (problem_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.logging_steps == 0:
                        avg_reward = np.mean(epoch_rewards[-50:]) if epoch_rewards else 0
                        solve_rate = epoch_correct / epoch_total if epoch_total > 0 else 0
                        self.logger.log({
                            "grpo/loss": accumulated_loss / max(global_step, 1),
                            "grpo/mean_reward": avg_reward,
                            "grpo/solve_rate": solve_rate,
                            "grpo/epoch": epoch + 1,
                            "grpo/problem_idx": problem_idx,
                        }, step=global_step)
                        pbar.set_postfix(
                            reward=f"{avg_reward:.3f}",
                            solve=f"{solve_rate:.2%}",
                        )

                # Save progress
                self.progress.update(problem_idx, result.get("rewards", [{}])[0] if result["rewards"] else {})

                # Checkpoint
                if global_step > 0 and global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step)

            # End of epoch stats
            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
            solve_rate = epoch_correct / epoch_total if epoch_total > 0 else 0
            print(f"\nEpoch {epoch + 1} complete: avg_reward={avg_reward:.4f}, solve_rate={solve_rate:.2%}")

            start_idx = 0  # Reset for next epoch

        # Save final model
        self._save_checkpoint("final")
        return os.path.join(self.config.output_dir, "final")

    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        ckpt_path = os.path.join(self.config.output_dir, f"checkpoint-{step}" if isinstance(step, int) else step)
        os.makedirs(ckpt_path, exist_ok=True)
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        self.progress.save()
        print(f"  [GRPO] Checkpoint saved: {ckpt_path}")


def run_grpo(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    reward_fn: CompositeReward,
    config=None,
    output_dir: str = "./checkpoints/grpo",
    device: str = "cuda",
) -> str:
    """Run Stage 4: GRPO Training.

    Args:
        model: SFT model to train
        tokenizer: Configured tokenizer
        train_dataset: GRPO dataset with 'problem' and 'answer' columns
        reward_fn: CompositeReward instance
        config: GRPOConfig instance
        output_dir: Where to save checkpoints
        device: Device to train on

    Returns:
        Path to final checkpoint
    """
    from ..configs.base_config import GRPOConfig
    if config is None:
        config = GRPOConfig()
    config.output_dir = output_dir

    # Create frozen reference model (copy of SFT)
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=config,
        device=device,
    )

    return trainer.train(train_dataset)
