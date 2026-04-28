"""
SymPy-Based Mathematical Answer Verifier
Verifies math answers using numeric comparison, symbolic equivalence, and string matching.
"""

import ast
import re
import math
import signal
from typing import Dict, List, Optional, Tuple
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

    # Strip common verbal prefixes before normalizing the remaining math.
    s = re.sub(
        r'^(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)?\s*',
        '',
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r'^(?:therefore|thus|hence|so),?\s*(?:the\s+answer\s*(?:is|=|:)?\s*)?',
        '',
        s,
        flags=re.IGNORECASE,
    )

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

    # Trim common trailing punctuation without disturbing fractions or decimals.
    s = s.strip(" \n\t.,;:!?")

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def _extract_braced_content(text: str, brace_start: int) -> Optional[Tuple[str, int]]:
    """Extract content from a balanced {...} region starting at brace_start."""
    if brace_start >= len(text) or text[brace_start] != "{":
        return None

    depth = 0
    chars: List[str] = []
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), idx + 1
            chars.append(ch)
        else:
            if depth >= 1:
                chars.append(ch)

    return None


def _clean_extracted_candidate(candidate: str) -> str:
    """Normalize a candidate answer extracted from free-form model output."""
    cleaned = candidate.strip()
    cleaned = cleaned.replace("<|im_end|>", "").strip()
    cleaned = re.sub(r'</?(?:answer|thinking|verification)>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^#+\s*', '', cleaned)
    cleaned = re.sub(
        r'^(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)?\s*',
        '',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'^(?:therefore|thus|hence|so),?\s*(?:the\s+answer\s*(?:is|=|:)?\s*)?',
        '',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip().strip(" \n\t.,;:!?")

    # Pull out a single inline math expression when the wrapper is obvious.
    if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()

    return cleaned.strip(" \n\t.,;:!?")


def extract_answer_candidates(response: str, max_candidates: int = 8) -> List[Dict[str, str]]:
    """Extract ranked answer candidates with provenance from a response."""
    if not response:
        return []

    candidates: List[Dict[str, str]] = []
    seen = set()

    def add_candidate(answer: str, method: str):
        cleaned = _clean_extracted_candidate(answer)
        if not cleaned:
            return
        key = normalize_answer(cleaned)
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append({"answer": cleaned, "method": method})

    # Structured tags are highest-confidence because SFT expects them.
    for match in re.finditer(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL | re.IGNORECASE):
        add_candidate(match.group(1), "answer_tag")

    # Handle nested \boxed{...} expressions.
    boxed_marker = "\\boxed"
    search_from = 0
    while True:
        marker_idx = response.find(boxed_marker, search_from)
        if marker_idx == -1:
            break
        brace_idx = marker_idx + len(boxed_marker)
        if brace_idx < len(response) and response[brace_idx] == "{":
            extracted = _extract_braced_content(response, brace_idx)
            if extracted is not None:
                content, next_idx = extracted
                add_candidate(content, "boxed")
                search_from = next_idx
                continue
        search_from = marker_idx + len(boxed_marker)

    phrase_patterns = [
        (r'(?im)(?:^|\n)\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)?\s*(.+)$', "answer_phrase"),
        (r'(?im)(?:^|\n)\s*(?:therefore|thus|hence|so),?\s*(?:the\s+answer\s*(?:is|=|:)?\s*)?(.+)$', "conclusion_phrase"),
        (r'(?im)(?:^|\n)\s*####\s*(.+)$', "gsm8k_delimiter"),
    ]
    for pattern, method in phrase_patterns:
        for match in re.finditer(pattern, response):
            add_candidate(match.group(1), method)

    # Scan upward from the end for concise math-like lines as a final fallback.
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    for line in reversed(lines[-6:]):
        if re.search(r'</?(?:answer|thinking|verification)>', line, flags=re.IGNORECASE):
            continue
        if any(token in line.lower() for token in ["answer", "boxed", "therefore", "thus", "hence", "####"]):
            add_candidate(line, "tail_line")
            continue
        if re.search(r'[0-9\\^=+\-*/()]', line) and re.fullmatch(r'[$\\()\[\]\-+*/^=0-9a-zA-Z_ .,]+', line):
            add_candidate(line, "tail_math_line")

    if not candidates and lines:
        add_candidate(lines[-1], "last_line")

    return candidates[:max_candidates]


def try_numeric_comparison(pred: str, target: str, tol: float = 1e-6) -> Optional[bool]:
    """Try to compare answers as numeric values."""
    def unwrap_outer_parens(value: str) -> str:
        trimmed = value.strip()
        if not (trimmed.startswith("(") and trimmed.endswith(")")):
            return trimmed

        depth = 0
        for idx, ch in enumerate(trimmed):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(trimmed) - 1:
                    return trimmed

        return trimmed[1:-1].strip()

    def parse_number(s):
        s = normalize_answer(s).strip().replace(",", "").replace(" ", "")

        previous = None
        while s != previous:
            previous = s
            s = unwrap_outer_parens(s)

        # Handle fractions like 3/4
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    left = unwrap_outer_parens(parts[0])
                    right = unwrap_outer_parens(parts[1])
                    return float(left) / float(right)
                except (ValueError, ZeroDivisionError):
                    return None

        # Handle percentages
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100
            except ValueError:
                return None

        try:
            expression = s.replace("^", "**")
            node = ast.parse(expression, mode="eval")

            def eval_node(current):
                if isinstance(current, ast.Expression):
                    return eval_node(current.body)
                if isinstance(current, ast.Constant) and isinstance(current.value, (int, float)):
                    return float(current.value)
                if isinstance(current, ast.UnaryOp) and isinstance(current.op, (ast.UAdd, ast.USub)):
                    value = eval_node(current.operand)
                    return value if isinstance(current.op, ast.UAdd) else -value
                if isinstance(current, ast.BinOp) and isinstance(current.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    left = eval_node(current.left)
                    right = eval_node(current.right)
                    if isinstance(current.op, ast.Add):
                        return left + right
                    if isinstance(current.op, ast.Sub):
                        return left - right
                    if isinstance(current.op, ast.Mult):
                        return left * right
                    if isinstance(current.op, ast.Div):
                        return left / right
                    return left ** right
                if isinstance(current, ast.Name) and current.id.lower() in {"pi", "e"}:
                    return math.pi if current.id.lower() == "pi" else math.e
                if isinstance(current, ast.Call) and isinstance(current.func, ast.Name):
                    func_name = current.func.id.lower()
                    if func_name == "sqrt" and len(current.args) == 1:
                        return math.sqrt(eval_node(current.args[0]))
                raise ValueError("Unsupported numeric expression")

            return float(eval_node(node))
        except Exception:
            pass

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

    # Cannot determine reliably; keep this distinct from a known wrong answer.
    return 0.5, "unknown"


def extract_answer_from_response(response: str) -> str:
    """Extract final answer from model response."""
    candidates = extract_answer_candidates(response, max_candidates=1)
    if candidates:
        return candidates[0]["answer"]
    return response.strip()
