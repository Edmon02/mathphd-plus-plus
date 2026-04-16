"""
Multi-Agent Debate for Hard Problems
Simulates multiple agents with different reasoning strategies using the same model.
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards.sympy_verifier import extract_answer_from_response, verify_answer


# Agent system prompts that induce different reasoning strategies
AGENT_PROMPTS = {
    "constructive": (
        "You are a constructive mathematician. Prefer direct proofs, explicit constructions, "
        "and forward reasoning. Build up the solution step by step from known facts. "
        "Avoid proof by contradiction unless absolutely necessary."
    ),
    "analytical": (
        "You are an analyst who excels at proof by contradiction and bounding arguments. "
        "Consider extreme cases, use inequalities, and think about what would happen if "
        "the conclusion were false. Look for counterexamples before attempting proofs."
    ),
    "algebraic": (
        "You are an algebraist. Look for algebraic structure, symmetry, and invariants. "
        "Transform the problem using substitutions, generating functions, or group actions. "
        "Reduce complex problems to simpler algebraic ones."
    ),
    "critic": (
        "You are a rigorous proof checker. Your job is to find ALL errors, gaps, and "
        "unjustified claims in mathematical proofs. Be extremely thorough. For each error, "
        "explain exactly why it is wrong and suggest a fix."
    ),
}


def generate_with_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    agent_type: str,
    context: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """Generate a response using a specific agent persona.

    Args:
        model: The model to use
        tokenizer: Tokenizer
        problem: Math problem
        agent_type: Key in AGENT_PROMPTS
        context: Additional context (e.g., previous agent's solution)
        max_new_tokens: Max generation length
        temperature: Sampling temperature
        device: Device

    Returns:
        Generated response text
    """
    system_prompt = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["constructive"])

    if context:
        user_message = f"{problem}\n\n[Previous attempt for reference]:\n{context}"
    else:
        user_message = problem

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
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

    response = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
    return response


def critique_solution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    solution: str,
    device: str = "cuda",
) -> str:
    """Have the critic agent review a solution."""
    return generate_with_agent(
        model, tokenizer,
        problem=f"Review this solution for correctness:\n\nProblem: {problem}\n\nSolution: {solution}",
        agent_type="critic",
        max_new_tokens=512,
        temperature=0.3,  # Lower temperature for critique
        device=device,
    )


def debate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: str,
    rounds: int = 3,
    ground_truth: Optional[str] = None,
    process_scorer=None,
    device: str = "cuda",
) -> Dict:
    """Run multi-agent debate protocol.

    Protocol:
    Round 1: Two provers independently attempt the problem
    Round 2: Critic reviews both solutions
    Round 3: Provers revise based on critiques

    Selection: Use SymPy verification if ground truth available,
    otherwise use PRM scoring or critic consensus.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        problem: Math problem text
        rounds: Number of debate rounds
        ground_truth: Optional answer for verification
        process_scorer: Optional PRM scorer
        device: Device

    Returns:
        dict with 'answer', 'solutions', 'critiques', 'method'
    """
    all_solutions = []
    all_critiques = []

    # Round 1: Independent solving
    print("  [Debate] Round 1: Independent solving...")
    proof_a = generate_with_agent(
        model, tokenizer, problem,
        agent_type="constructive",
        device=device,
    )
    proof_b = generate_with_agent(
        model, tokenizer, problem,
        agent_type="analytical",
        device=device,
    )
    all_solutions.extend([
        {"agent": "constructive", "round": 1, "text": proof_a},
        {"agent": "analytical", "round": 1, "text": proof_b},
    ])

    # Round 2: Critique
    print("  [Debate] Round 2: Critic review...")
    critique_a = critique_solution(model, tokenizer, problem, proof_a, device)
    critique_b = critique_solution(model, tokenizer, problem, proof_b, device)
    all_critiques.extend([
        {"target": "constructive", "text": critique_a},
        {"target": "analytical", "text": critique_b},
    ])

    # Round 3: Revision
    print("  [Debate] Round 3: Revision...")
    revised_a = generate_with_agent(
        model, tokenizer, problem,
        agent_type="constructive",
        context=f"Your previous solution:\n{proof_a}\n\nCritique:\n{critique_a}\n\nRevise your solution addressing the critique.",
        device=device,
    )
    revised_b = generate_with_agent(
        model, tokenizer, problem,
        agent_type="analytical",
        context=f"Your previous solution:\n{proof_b}\n\nCritique:\n{critique_b}\n\nRevise your solution addressing the critique.",
        device=device,
    )
    all_solutions.extend([
        {"agent": "constructive_revised", "round": 3, "text": revised_a},
        {"agent": "analytical_revised", "round": 3, "text": revised_b},
    ])

    # Optional: algebraic perspective
    if rounds > 2:
        proof_c = generate_with_agent(
            model, tokenizer, problem,
            agent_type="algebraic",
            device=device,
        )
        all_solutions.append({"agent": "algebraic", "round": 1, "text": proof_c})

    # Selection
    candidates = [s["text"] for s in all_solutions]
    answers = [extract_answer_from_response(c) for c in candidates]

    # If ground truth available, verify each
    if ground_truth:
        for i, (sol, ans) in enumerate(zip(all_solutions, answers)):
            score, method = verify_answer(ans, ground_truth)
            sol["verified"] = score > 0.5
            sol["answer"] = ans

        verified = [s for s in all_solutions if s.get("verified", False)]
        if verified:
            best = verified[-1]  # Prefer revised versions
            return {
                "answer": best["answer"],
                "reasoning": best["text"],
                "method": f"verified_{best['agent']}",
                "solutions": all_solutions,
                "critiques": all_critiques,
            }

    # If PRM available, score all
    if process_scorer:
        scores = []
        for sol in all_solutions:
            mean_score, _ = process_scorer.score_solution(sol["text"])
            sol["prm_score"] = mean_score
            scores.append(mean_score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = all_solutions[best_idx]
        return {
            "answer": answers[best_idx],
            "reasoning": best["text"],
            "method": f"prm_best_{best['agent']}",
            "solutions": all_solutions,
            "critiques": all_critiques,
        }

    # Fallback: prefer revised solutions
    best = all_solutions[-2]  # Last revised solution
    return {
        "answer": answers[-2],
        "reasoning": best["text"],
        "method": f"fallback_{best['agent']}",
        "solutions": all_solutions,
        "critiques": all_critiques,
    }
