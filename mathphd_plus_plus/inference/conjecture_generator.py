"""
Adversarial Conjecture Generation
Generates and filters mathematical conjectures using generator-critic loop.
"""

import torch
import re
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards.code_executor import execute_code


MATH_DOMAINS = [
    "number theory", "graph theory", "combinatorics",
    "real analysis", "linear algebra", "group theory",
    "topology", "probability", "algebraic geometry",
    "differential equations", "complex analysis", "ring theory",
]


def generate_conjecture(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    domain: str = "number theory",
    max_new_tokens: int = 300,
    temperature: float = 0.9,
    device: str = "cuda",
) -> str:
    """Generate a mathematical conjecture in a given domain."""
    prompt = (
        f"<|im_start|>system\nYou are a creative mathematician who generates "
        f"novel, precise mathematical conjectures.<|im_end|>\n"
        f"<|im_start|>user\nGenerate a plausible but non-obvious mathematical conjecture "
        f"about {domain}. The conjecture should be:\n"
        f"(a) precisely stated with clear mathematical notation\n"
        f"(b) not obviously true or false\n"
        f"(c) testable for small cases with a computer\n\n"
        f"State just the conjecture, then provide Python code to test it for small cases.\n"
        f"Format:\nCONJECTURE: [precise statement]\nTEST_CODE:\n```python\n[code]\n```<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    return tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)


def critique_conjecture(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conjecture: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """Attempt to prove or disprove a conjecture."""
    prompt = (
        f"<|im_start|>system\nYou are a rigorous mathematician. "
        f"Attempt to prove OR disprove the following conjecture. "
        f"Check small cases first, then attempt a general proof or find a counterexample.<|im_end|>\n"
        f"<|im_start|>user\n{conjecture}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Low temp for rigorous analysis
                pad_token_id=tokenizer.pad_token_id,
            )

    return tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)


def parse_conjecture_output(text: str) -> Dict:
    """Parse generator output into conjecture and test code."""
    result = {"conjecture": "", "test_code": "", "raw": text}

    # Extract conjecture
    match = re.search(r'CONJECTURE:\s*(.+?)(?=TEST_CODE|```|$)', text, re.DOTALL)
    if match:
        result["conjecture"] = match.group(1).strip()

    # Extract test code
    code_match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if code_match:
        result["test_code"] = code_match.group(1).strip()

    return result


def evaluate_conjecture(conjecture_data: Dict) -> Dict:
    """Evaluate a conjecture by running test code.

    Returns:
        dict with 'status': 'confirmed' | 'refuted' | 'error' | 'untested'
    """
    if not conjecture_data.get("test_code"):
        return {"status": "untested", "details": "No test code provided"}

    result = execute_code(conjecture_data["test_code"], timeout_seconds=15)

    if not result.success:
        return {
            "status": "error",
            "details": result.error,
        }

    stdout = result.stdout.strip().lower()

    # Check for counterexample
    if any(word in stdout for word in ["false", "counterexample", "failed", "disproved"]):
        return {
            "status": "refuted",
            "details": result.stdout,
        }

    # Check for confirmation
    if any(word in stdout for word in ["true", "confirmed", "passed", "verified"]):
        return {
            "status": "confirmed",
            "details": result.stdout,
        }

    return {
        "status": "uncertain",
        "details": result.stdout,
    }


def run_conjecture_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_conjectures: int = 20,
    domains: Optional[List[str]] = None,
    max_critique_attempts: int = 2,
    device: str = "cuda",
) -> Dict:
    """Run the full adversarial conjecture generation loop.

    1. Generate conjectures across domains
    2. Test each with code execution
    3. Critique unresolved ones
    4. Classify results

    Args:
        model: Trained model
        tokenizer: Tokenizer
        num_conjectures: Total conjectures to generate
        domains: Math domains to sample from
        max_critique_attempts: Max attempts to resolve each conjecture
        device: Device

    Returns:
        dict with 'conjectures', 'resolved', 'unresolved', 'interesting'
    """
    domains = domains or MATH_DOMAINS
    all_conjectures = []
    resolved = []
    unresolved = []

    print(f"[Conjecture Generation] Generating {num_conjectures} conjectures...")

    for i in range(num_conjectures):
        domain = domains[i % len(domains)]
        print(f"  [{i+1}/{num_conjectures}] Domain: {domain}")

        # Generate
        raw_output = generate_conjecture(model, tokenizer, domain, device=device)
        parsed = parse_conjecture_output(raw_output)
        parsed["domain"] = domain
        parsed["id"] = i

        # Test with code
        test_result = evaluate_conjecture(parsed)
        parsed["test_result"] = test_result

        if test_result["status"] in ["confirmed", "refuted"]:
            parsed["resolution"] = test_result["status"]
            resolved.append(parsed)
        else:
            # Critique
            for attempt in range(max_critique_attempts):
                critique = critique_conjecture(
                    model, tokenizer,
                    parsed["conjecture"],
                    device=device,
                )
                parsed.setdefault("critiques", []).append(critique)

                # Check if critic resolved it
                critique_lower = critique.lower()
                if "trivially true" in critique_lower or "well known" in critique_lower:
                    parsed["resolution"] = "trivially_true"
                    resolved.append(parsed)
                    break
                elif "counterexample" in critique_lower or "disproved" in critique_lower:
                    parsed["resolution"] = "refuted_by_critic"
                    resolved.append(parsed)
                    break
            else:
                parsed["resolution"] = "unresolved"
                unresolved.append(parsed)

        all_conjectures.append(parsed)

    # Identify "interesting" conjectures (unresolved after all attempts)
    interesting = [c for c in unresolved if c.get("test_result", {}).get("status") == "confirmed"]

    print(f"\n[Results]")
    print(f"  Total: {len(all_conjectures)}")
    print(f"  Resolved: {len(resolved)}")
    print(f"  Unresolved: {len(unresolved)}")
    print(f"  Interesting (confirmed by code, unresolved by critic): {len(interesting)}")

    return {
        "conjectures": all_conjectures,
        "resolved": resolved,
        "unresolved": unresolved,
        "interesting": interesting,
    }
