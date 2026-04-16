"""
SymPy-Based Mathematical Answer Verifier
Verifies math answers using numeric comparison, symbolic equivalence, and string matching.
"""

import re
import math
import signal
from typing import Optional, Tuple
from contextlib import contextmanager

try:
    import sympy
    from sympy import simplify, sympify, Rational, pi, E, oo, sqrt, I
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: int = 5):
    """Timeout context manager for SymPy operations that may hang."""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def normalize_answer(answer: str) -> str:
    """Normalize a math answer string for comparison."""
    if not answer:
        return ""

    s = answer.strip()

    # Remove common LaTeX wrappers
    s = s.replace("$", "").strip()

    # Remove \\boxed{...}
    match = re.match(r'\\boxed\{(.+)\}', s)
    if match:
        s = match.group(1)

    # Normalize fractions: \\frac{a}{b} -> a/b
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)

    # Normalize common LaTeX
    replacements = {
        "\\pi": "pi",
        "\\infty": "oo",
        "\\sqrt": "sqrt",
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\;": "",
        "\\!": "",
        "\\text{": "",
        "\\mathrm{": "",
        "\\mathbf{": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # Remove trailing }
    while s.endswith("}"):
        s = s[:-1]

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def try_numeric_comparison(pred: str, target: str, tol: float = 1e-6) -> Optional[bool]:
    """Try to compare answers as numeric values."""
    def parse_number(s):
        s = s.strip().replace(",", "").replace(" ", "")

        # Handle fractions like 3/4
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    return None

        # Handle percentages
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100
            except ValueError:
                return None

        try:
            return float(s)
        except ValueError:
            return None

    pred_val = parse_number(normalize_answer(pred))
    target_val = parse_number(normalize_answer(target))

    if pred_val is not None and target_val is not None:
        if target_val == 0:
            return abs(pred_val) < tol
        return abs(pred_val - target_val) / max(abs(target_val), 1e-10) < tol

    return None  # Cannot compare numerically


def try_symbolic_comparison(pred: str, target: str, timeout_sec: int = 5) -> Optional[bool]:
    """Try to compare answers using SymPy symbolic equivalence."""
    if not SYMPY_AVAILABLE:
        return None

    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)

    try:
        with timeout(timeout_sec):
            # Try parsing as SymPy expressions
            pred_expr = sympify(pred_norm)
            target_expr = sympify(target_norm)

            # Check symbolic equivalence
            diff = simplify(pred_expr - target_expr)
            if diff == 0:
                return True

            # Try numerical evaluation
            try:
                pred_float = complex(pred_expr.evalf())
                target_float = complex(target_expr.evalf())
                if abs(pred_float - target_float) < 1e-6:
                    return True
            except (TypeError, ValueError):
                pass

            return False

    except (TimeoutError, Exception):
        return None


def try_latex_comparison(pred: str, target: str, timeout_sec: int = 5) -> Optional[bool]:
    """Try to compare answers by parsing LaTeX."""
    if not SYMPY_AVAILABLE:
        return None

    try:
        with timeout(timeout_sec):
            pred_expr = parse_latex(pred.strip("$"))
            target_expr = parse_latex(target.strip("$"))

            diff = simplify(pred_expr - target_expr)
            return diff == 0

    except (TimeoutError, Exception):
        return None


def verify_answer(predicted: str, ground_truth: str) -> Tuple[float, str]:
    """Verify if predicted answer matches ground truth.

    Tries multiple comparison strategies in order:
    1. Exact string match (after normalization)
    2. Numeric comparison
    3. SymPy symbolic equivalence
    4. LaTeX parsing + symbolic comparison

    Returns:
        (score, method) where score is 1.0 (correct), 0.0 (incorrect),
        or 0.5 (uncertain/timeout), and method describes how it was verified.
    """
    if not predicted or not ground_truth:
        return 0.0, "empty"

    # 1. Exact string match after normalization
    pred_norm = normalize_answer(predicted)
    target_norm = normalize_answer(ground_truth)

    if pred_norm == target_norm:
        return 1.0, "exact_match"

    # 2. Numeric comparison
    result = try_numeric_comparison(predicted, ground_truth)
    if result is True:
        return 1.0, "numeric"
    elif result is False:
        return 0.0, "numeric"

    # 3. SymPy symbolic equivalence
    result = try_symbolic_comparison(predicted, ground_truth)
    if result is True:
        return 1.0, "symbolic"
    elif result is False:
        return 0.0, "symbolic"

    # 4. LaTeX parsing
    result = try_latex_comparison(predicted, ground_truth)
    if result is True:
        return 1.0, "latex"
    elif result is False:
        return 0.0, "latex"

    # Cannot determine
    return 0.0, "unknown"


def extract_answer_from_response(response: str) -> str:
    """Extract final answer from model response."""
    # Try <answer> tags
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try \\boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()

    # Try "The answer is X"
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)', response)
    if match:
        return match.group(1).strip()

    # Try "#### X" (GSM8K format)
    match = re.search(r'####\s*(.+)', response)
    if match:
        return match.group(1).strip()

    # Last line as fallback
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return response.strip()
