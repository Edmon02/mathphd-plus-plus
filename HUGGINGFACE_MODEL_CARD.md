---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - math
  - reasoning
  - chain-of-thought
  - qwen2
  - conversational
  - rlvr
base_model: Qwen/Qwen2.5-0.5B-Instruct
---

# MathPhD++ 0.5B

**MathPhD++** is a small (≈0.5B parameter) language model fine-tuned for **mathematical reasoning** in natural language. It is built on [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and trained with the **MathPhD++** open-source pipeline (see linked code repository in your Hub “Model sources” if you publish it): supervised fine-tuning (SFT) on curated math instruction data with structured `<thinking>` / `<answer>` (and related) tags, optional process reward modeling (PRM), and reinforcement learning from verifiable rewards (GRPO) using SymPy-backed correctness checks.

This Hub release is intended as a **reproducible checkpoint** for research and experimentation on math LLMs at the edge of what fits comfortably on a single consumer or Colab GPU.

## Model summary

| Attribute | Value |
|-----------|--------|
| **Architecture** | Qwen2 (causal LM), ~0.5B parameters |
| **Precision** | FP16 (typical Hub export) |
| **Chat format** | ChatML (`<|im_start|>` / `<|im_end|>`) — prefer `tokenizer.apply_chat_template` when available |
| **Primary use** | Step-by-step math word problems, competition-style reasoning (informal proofs / chain-of-thought) |
| **Developed by** | Edmon (Edmon02) — community research project |
| **Finetuned from** | `Qwen/Qwen2.5-0.5B-Instruct` |

## Training data (high level)

SFT mixes multiple public sources (non-exhaustive; see project config for exact caps):

- MetaMath-style QA
- Competition MATH (train)
- GSM8K (train)
- OpenMathInstruct-2 (subset)
- NuminaMath-CoT (subset)

Examples are formatted in **ChatML** with structured assistant outputs (reasoning blocks and final answers) to encourage verifiable extraction and consistent formatting for downstream RL.

## Evaluation (reported from project notebook run)

Results below are **indicative** and used a **200-sample** cap per benchmark (`QUICK_TEST`-style eval). For publication-quality numbers, run full GSM8K test (1,319) and a standard MATH split with fixed protocol.

| Benchmark | Subset / protocol | Accuracy |
|-----------|-------------------|----------|
| GSM8K | 200 / test | **18.5%** (37/200) |
| MATH | 200 / MATH-500 | **6.0%** (12/200) |

These scores reflect the **SFT-loaded** policy evaluated after the pipeline fix that loads fine-tuned weights from checkpoint storage (not the raw base model). A 0.5B model remains **capacity-limited** on MATH; GSM8K is the more informative “did SFT help?” signal at this scale.

## How to use

### Transformers (generate)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Edmon02/mathphd-plus-plus-0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

problem = "What is the sum of the first 100 positive integers?"
prompt = (
    "<|im_start|>system\nYou are MathPhD++, an advanced mathematical reasoning assistant. "
    "Show your complete reasoning step-by-step.<|im_end|>\n"
    f"<|im_start|>user\n{problem}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use **greedy or low temperature** for benchmarking; use sampling for exploratory interaction.

## Limitations

- **Small model:** Will underperform larger instruction models on hard competition math and long proofs.
- **Informal reasoning:** Outputs are not formally verified unless you pair the model with an external proof checker or code execution sandbox.
- **Data contamination:** Public math benchmarks overlap train/eval sources; treat leaderboard-style claims with care unless you hold out data strictly.
- **Language:** Primarily English math text; mixed-language or non-math prompts are out of distribution.

## Bias, safety, and responsible use

This model inherits behaviors and limitations of the base Qwen2.5 model and the fine-tuning corpora. It may produce confident but incorrect mathematics. **Do not** use as a sole authority for safety-critical, financial, medical, or legal reasoning. Prefer human review and independent verification.

## Environmental note

If your Hub UI shows an unrelated arXiv paper (e.g. carbon footprint of ML), that is often an **automatic metadata artifact**. This model card is the authoritative description; consider removing incorrect `arxiv:` tags under model settings.

## Links

- **Checkpoints / artifacts (author):** [Google Drive — mathphd_checkpoints](https://drive.google.com/drive/folders/14T6zF9B_Zh0JbKUW2nFEWz7QrYtW_r85?usp=sharing) (SFT, PRM, GRPO, eval exports — access as permitted by owner)
- **Base model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## Citation

If you use this model, cite the base model and this Hub repository as appropriate:

```bibtex
@misc{mathphd_plus_plus_05b,
  title        = {MathPhD++ 0.5B: Math Reasoning Model (Qwen2.5-0.5B-Instruct fine-tune)},
  author       = {Edmon02},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Edmon02/mathphd-plus-plus-0.5b}},
}
```

---

*Model card written for professional Hub documentation. Update the GitHub URL and evaluation table when you publish full-benchmark runs.*
