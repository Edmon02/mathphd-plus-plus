"""
Sandboxed Code Executor
Safely executes Python code for reward computation.
"""

import sys
import signal
import traceback
from typing import Dict, Any, Optional
from io import StringIO


class CodeExecutionResult:
    """Result of code execution."""
    def __init__(self, stdout: str = "", stderr: str = "", result: Any = None,
                 success: bool = True, error: Optional[str] = None):
        self.stdout = stdout
        self.stderr = stderr
        self.result = result
        self.success = success
        self.error = error

    def __repr__(self):
        status = "OK" if self.success else f"FAIL: {self.error}"
        return f"CodeExecutionResult({status}, stdout={self.stdout[:100]})"


# Safe builtins whitelist
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'chr', 'complex',
    'dict', 'divmod', 'enumerate', 'filter', 'float', 'format',
    'frozenset', 'hash', 'hex', 'int', 'isinstance', 'issubclass',
    'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct',
    'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round',
    'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
}

# Allowed imports
ALLOWED_MODULES = {
    'math', 'cmath', 'fractions', 'decimal', 'statistics',
    'itertools', 'functools', 'operator', 'collections',
    'sympy', 'numpy', 'scipy',
}


def execute_code(
    code: str,
    timeout_seconds: int = 10,
    max_output_chars: int = 10000,
) -> CodeExecutionResult:
    """Execute Python code in a restricted sandbox.

    Args:
        code: Python code string to execute
        timeout_seconds: Maximum execution time
        max_output_chars: Maximum characters in stdout/stderr

    Returns:
        CodeExecutionResult with execution output
    """
    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = StringIO()
    sys.stderr = captured_stderr = StringIO()

    result = CodeExecutionResult()

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {timeout_seconds}s")

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        # Create restricted globals
        safe_globals = {"__builtins__": {}}

        # Add safe builtins
        import builtins
        for name in SAFE_BUILTINS:
            safe_globals["__builtins__"][name] = getattr(builtins, name)

        # Add safe __import__
        original_import = __import__

        def restricted_import(name, *args, **kwargs):
            if name.split('.')[0] not in ALLOWED_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed")
            return original_import(name, *args, **kwargs)

        safe_globals["__builtins__"]["__import__"] = restricted_import

        # Pre-import common math modules
        try:
            import math as _math
            import sympy as _sympy
            safe_globals["math"] = _math
            safe_globals["sympy"] = _sympy
        except ImportError:
            pass

        # Execute
        exec_result = {}
        exec(code, safe_globals, exec_result)

        result.success = True
        result.result = exec_result.get("result", exec_result.get("answer", None))

    except TimeoutError as e:
        result.success = False
        result.error = str(e)
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {str(e)}"
        result.stderr = traceback.format_exc()
    finally:
        # Restore
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        result.stdout = captured_stdout.getvalue()[:max_output_chars]
        result.stderr = (result.stderr or "") + captured_stderr.getvalue()[:max_output_chars]

        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return result


def verify_with_code(code: str, expected_answer: str) -> float:
    """Execute code and verify if its output matches expected answer.

    Returns 1.0 if match, 0.0 if mismatch, 0.5 if execution failed.
    """
    from .sympy_verifier import verify_answer, extract_answer_from_response

    result = execute_code(code)

    if not result.success:
        return 0.5  # Partial credit for reasonable attempt

    # Check result variable
    if result.result is not None:
        score, _ = verify_answer(str(result.result), expected_answer)
        if score > 0:
            return score

    # Check stdout
    if result.stdout.strip():
        score, _ = verify_answer(result.stdout.strip().split('\n')[-1], expected_answer)
        if score > 0:
            return score

    return 0.0
