# MathPhD++

**A T4-trainable prototype for mathematical superintelligence.**

MathPhD++ implements a complete 5-stage training pipeline that transforms a small language model into a mathematical reasoning engine using Reinforcement Learning from Verifiable Rewards (RLVR). Designed to run on a single T4 GPU in Google Colab.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Edmon02/mathphd-plus-plus/blob/main/mathphd_plus_plus/notebooks/MathPhD_T4_Training.ipynb)

---

## Overview

Most mathematical AI systems require hundreds of GPUs and millions of dollars. MathPhD++ proves that the core techniques behind frontier math reasoning — GRPO, process reward models, MCTS proof search, and multi-agent debate — can be implemented and trained on a single consumer GPU.

### Training Pipeline

```
Qwen2.5-0.5B (base)
    |
    v
[Stage 1] Continued Pre-Training ──── Multi-objective loss (L_NTP + L_structure)
    |                                   Curriculum learning (easy → hard)
    v
[Stage 2] Supervised Fine-Tuning ───── 63K math problems in ChatML format
    |                                   Response-only loss masking
    v
[Stage 3] Process Reward Model ─────── Step-level correctness scoring
    |                                   Math-Shepherd data (800K steps)
    v
[Stage 4] GRPO ─────────────────────── Group Relative Policy Optimization
    |                                   SymPy-verified correctness rewards
    |                                   PRM process rewards + format rewards
    v
[Stage 5] Inference ────────────────── Self-Consistency | Tree-of-Thoughts
                                        MCTS Proof Search | Multi-Agent Debate
                                        Adversarial Conjecture Generation
```

### Key Features

- **Multi-Objective Pre-Training**: Custom loss that upweights theorem/proof regions
- **GRPO with Verifiable Rewards**: SymPy-based answer verification as ground-truth reward signal
- **Process Reward Model**: Step-level scoring for dense training signal
- **5 Inference Strategies**: Direct, Self-Consistency, Tree-of-Thoughts, MCTS, Multi-Agent Debate
- **Adversarial Conjecture Generator**: Discovers novel mathematical conjectures
- **Colab Crash Recovery**: Google Drive checkpointing with automatic resume

## Quick Start

### Google Colab (Recommended)

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Edmon02/mathphd-plus-plus/blob/main/mathphd_plus_plus/notebooks/MathPhD_T4_Training.ipynb)
2. Select **T4 GPU** runtime
3. Set `QUICK_TEST = True` for a fast validation run (~2-3 hours)
4. Run all cells

### Local Installation

```bash
git clone https://github.com/Edmon02/mathphd-plus-plus.git
cd mathphd-plus-plus
pip install -e ".[full]"
```

### Run Training Stages

```python
from mathphd_plus_plus.configs.base_config import MathPhDConfig
from mathphd_plus_plus.models.base_model import load_model_and_tokenizer
from mathphd_plus_plus.data.download import download_sft_data
from mathphd_plus_plus.data.preprocess_sft import prepare_sft_dataset
from mathphd_plus_plus.training.sft_trainer import run_sft

config = MathPhDConfig()
model, tokenizer = load_model_and_tokenizer()

sft_raw = download_sft_data()
sft_dataset = prepare_sft_dataset(sft_raw, tokenizer)

run_sft(model, tokenizer, sft_dataset, config=config.sft)
```

## Architecture

### Memory Budget (T4 — 16GB VRAM)

| Component | Memory |
|-----------|--------|
| Model weights (fp16) | 0.99 GB |
| Optimizer (AdamW) | 3.95 GB |
| Gradients | 0.99 GB |
| Activations (grad ckpt) | ~1.5 GB |
| **Total** | **~7.4 GB (46%)** |
| Headroom for generation | 8.6 GB |

### GRPO Algorithm

```
For each problem x:
  1. Generate G=4 solutions from policy π_θ
  2. Compute rewards:
     R = 0.6 × R_correctness (SymPy) + 0.3 × R_process (PRM) + 0.1 × R_format
  3. Normalize advantages: A_i = (R_i - mean) / std
  4. Update policy with clipped ratio + KL penalty against frozen SFT reference
```

### Reward System

| Component | Signal | Source |
|-----------|--------|--------|
| Correctness | Binary (0/1) | SymPy equivalence checking |
| Process | Step-level [0,1] | Trained Process Reward Model |
| Format | Structure bonus | Tag detection (`<thinking>`, `<answer>`) |

## Project Structure

```
mathphd_plus_plus/
├── configs/          # Hyperparameters for all 5 stages
├── data/             # Download, preprocessing, curriculum, collators
├── models/           # Base model, reward model, multi-objective loss
├── training/         # CPT, SFT, PRM, GRPO trainers + callbacks
├── rewards/          # SymPy verifier, code executor, composite reward
├── inference/        # Self-consistency, ToT, MCTS, debate, conjectures
├── evaluation/       # GSM8K, MATH benchmarks with breakdowns
└── notebooks/        # Comprehensive Colab training notebook
```

## Expected Results

| Benchmark | Base Qwen2.5-0.5B | After SFT | After GRPO | With MCTS |
|-----------|-------------------|-----------|------------|-----------|
| GSM8K | ~35% | ~55% | ~65% | ~72% |
| MATH (overall) | ~15% | ~28% | ~35% | ~42% |
| MATH Level 1-2 | ~45% | ~70% | ~80% | ~88% |
| MATH Level 4-5 | ~3% | ~10% | ~15% | ~22% |

## Training Time Estimates (T4)

| Stage | Duration | Colab Sessions |
|-------|----------|----------------|
| CPT | ~8h | 1 |
| SFT | ~6h | 1 |
| PRM | ~3h | 1 |
| GRPO | ~38h | 4 |
| Eval | ~3h | 1 |
| **Total** | **~58h** | **~7** |

## Datasets Used

| Dataset | Samples | Stage |
|---------|---------|-------|
| OpenWebMath | 500M tokens | CPT |
| MetaMathQA | 40K | SFT |
| MATH (Hendrycks) | 7.5K | SFT, GRPO, Eval |
| GSM8K | 7.5K | SFT, GRPO, Eval |
| NuminaMath-CoT | 3K | SFT, GRPO |
| Math-Shepherd | 100K steps | PRM |

## Roadmap

- [ ] Lean 4 integration for formal proof verification
- [ ] Scaling to Qwen2.5-1.5B with QLoRA
- [ ] Multi-modal support (diagram understanding)
- [ ] Distributed training for larger models
- [ ] Web interface for interactive problem solving
- [ ] Benchmark on FrontierMath and miniF2F

## Citation

```bibtex
@software{mathphd_plus_plus,
  title={MathPhD++: T4-Scale Mathematical Reasoning with GRPO and MCTS},
  author={Edmon02},
  year={2026},
  url={https://github.com/Edmon02/mathphd-plus-plus}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) for the base model
- [DeepSeek-R1](https://arxiv.org/abs/2401.02954) for GRPO inspiration
- [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) for MCTS proof search concepts
- [Math-Shepherd](https://arxiv.org/abs/2312.08935) for process reward data
