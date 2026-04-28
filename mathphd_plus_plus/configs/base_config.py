"""
MathPhD++ Configuration
All hyperparameters for the 5-stage training pipeline on T4 GPU.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Base model configuration."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "float16"
    use_flash_attention: bool = False  # T4 doesn't support flash-attn2
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048

    # Special tokens for structured math output
    special_tokens: List[str] = field(default_factory=lambda: [
        "<theorem>", "</theorem>",
        "<proof>", "</proof>",
        "<definition>", "</definition>",
        "<lemma>", "</lemma>",
        "<thinking>", "</thinking>",
        "<answer>", "</answer>",
        "<verification>", "</verification>",
        "<step>", "</step>",
    ])


@dataclass
class CPTConfig:
    """Stage 1: Continued Pre-Training configuration."""
    # Data
    max_seq_length: int = 2048
    target_tokens: int = 500_000_000  # 500M tokens subset

    # Training
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # effective batch = 16
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.05
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    fp16: bool = True
    bf16: bool = False

    # Multi-objective loss
    structure_loss_weight: float = 0.1  # alpha in L_total = L_NTP + alpha * L_structure
    structure_upweight_factor: float = 2.0  # Upweight tokens inside theorem/proof regions

    # Checkpointing
    save_steps: int = 500
    logging_steps: int = 50
    output_dir: str = "./checkpoints/cpt"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class SFTConfig:
    """Stage 2: Supervised Fine-Tuning configuration."""
    # Data
    max_seq_length: int = 2048
    sft_data_mix: dict = field(default_factory=lambda: {
        "meta-math/MetaMathQA": 40_000,
        "DigitalLearningGmbH/MATH-lighteval": -1,  # all 7.5K
        "openai/gsm8k": -1,  # all 7.5K
        "nvidia/OpenMathInstruct-2": 5_000,
        "AI-MO/NuminaMath-CoT": 3_000,
    })

    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch = 16
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.03
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    fp16: bool = True
    bf16: bool = False

    # Loss masking
    mask_prompt_tokens: bool = True  # Only compute loss on assistant responses

    # Checkpointing
    save_steps: int = 200
    logging_steps: int = 25
    output_dir: str = "./checkpoints/sft"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class PRMConfig:
    """Stage 3: Process Reward Model configuration."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    reward_head_dim: int = 896  # Qwen2.5-0.5B hidden dim
    max_seq_length: int = 1024

    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch = 16
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.05
    weight_decay: float = 0.01
    num_train_epochs: int = 2
    fp16: bool = True
    bf16: bool = False

    # Checkpointing
    save_steps: int = 300
    logging_steps: int = 50
    output_dir: str = "./checkpoints/prm"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class GRPOConfig:
    """Stage 4: GRPO with Verifiable Rewards configuration."""
    # Generation
    group_size: int = 4  # G=4 (memory constraint)
    max_new_tokens: int = 640
    generation_temperature: float = 0.7
    generation_top_p: float = 0.95
    generation_do_sample: bool = True

    # GRPO
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1  # One problem at a time
    gradient_accumulation_steps: int = 8  # Accumulate over 8 problems
    num_grpo_epochs: int = 3
    problems_per_epoch: int = 3000
    fp16: bool = True
    bf16: bool = False

    # Reward weights
    reward_correctness_weight: float = 0.8
    reward_process_weight: float = 0.15
    reward_format_weight: float = 0.05

    # Checkpointing
    save_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "./checkpoints/grpo"
    resume_from_checkpoint: Optional[str] = None
    resume_from_problem_idx: int = 0  # For crash recovery


@dataclass
class InferenceConfig:
    """Stage 5: Inference strategies configuration."""
    # Self-consistency
    sc_num_samples: int = 16
    sc_temperature: float = 0.7
    sc_top_p: float = 0.95

    # Tree-of-Thoughts
    tot_beam_width: int = 3
    tot_max_depth: int = 8
    tot_branching_factor: int = 3
    tot_step_tokens: int = 100  # Max tokens per step

    # MCTS
    mcts_c_puct: float = 1.5
    mcts_num_simulations: int = 50
    mcts_max_depth: int = 15
    mcts_rollout_max_tokens: int = 512

    # Multi-agent debate
    debate_rounds: int = 3
    debate_max_tokens: int = 512

    # General
    max_new_tokens: int = 512
    device: str = "cuda"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    benchmarks: List[str] = field(default_factory=lambda: [
        "gsm8k", "math", "aime"
    ])
    max_new_tokens: int = 768
    temperature: float = 0.0  # Greedy for eval
    batch_size: int = 8
    output_dir: str = "./eval_results"


@dataclass
class MathPhDConfig:
    """Master configuration combining all stages."""
    model: ModelConfig = field(default_factory=ModelConfig)
    cpt: CPTConfig = field(default_factory=CPTConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    prm: PRMConfig = field(default_factory=PRMConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Google Drive paths for Colab persistence
    gdrive_checkpoint_root: str = "/content/drive/MyDrive/mathphd_checkpoints"
    use_wandb: bool = True
    wandb_project: str = "mathphd-plus-plus"
    seed: int = 42
