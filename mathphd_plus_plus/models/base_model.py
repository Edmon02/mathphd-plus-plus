"""
Base Model Loader
Handles loading Qwen2.5-0.5B with proper configuration for T4 GPU.
"""

import torch
from typing import Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    special_tokens: Optional[List[str]] = None,
) -> AutoTokenizer:
    """Load and configure tokenizer with math-specific special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add special math tokens
    if special_tokens:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens,
        })
        if num_added > 0:
            print(f"Added {num_added} special tokens to tokenizer")

    return tokenizer


def load_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    torch_dtype: str = "float16",
    gradient_checkpointing: bool = True,
    device_map: str = "auto",
    quantization: Optional[str] = None,  # None, "4bit", "8bit"
    tokenizer: Optional[AutoTokenizer] = None,
) -> AutoModelForCausalLM:
    """Load base model with T4-optimized settings.

    Args:
        model_name: HuggingFace model ID
        torch_dtype: "float16" for T4 (no bf16 support)
        gradient_checkpointing: Enable to save VRAM
        device_map: "auto" for single GPU
        quantization: None for full precision, "4bit"/"8bit" for QLoRA
        tokenizer: If provided, resize embeddings to match tokenizer

    Returns:
        Configured model ready for training
    """
    dtype = getattr(torch, torch_dtype)

    # Quantization config for QLoRA track
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",  # T4 doesn't support flash-attn2
    )

    # Resize embeddings if tokenizer has extra tokens
    if tokenizer and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings: {model.config.vocab_size} -> {len(tokenizer)}")

    # Enable gradient checkpointing for memory savings
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Print memory estimate
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model loaded: {model.num_parameters() / 1e6:.1f}M params, {param_bytes / 1e9:.2f} GB")

    return model


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    special_tokens: Optional[List[str]] = None,
    gradient_checkpointing: bool = True,
    quantization: Optional[str] = None,
):
    """Convenience function to load both model and tokenizer together."""
    tokenizer = load_tokenizer(model_name, special_tokens)
    model = load_model(
        model_name=model_name,
        gradient_checkpointing=gradient_checkpointing,
        quantization=quantization,
        tokenizer=tokenizer,
    )
    return model, tokenizer
