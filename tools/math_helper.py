import os
from typing import Union
import wolframalpha
from langchain_core.tools import tool


@tool
def calculator(a: Union[int, float], b: Union[int, float], operation: str) -> float:
    """
    Perform basic arithmetic between two numeric values.

    Supported operations:
      - add (alias: sum)
      - subtract
      - multiply
      - divide (alias: div)
      - modulus (alias: mod)

    Args:
        a: First operand (int or float).
        b: Second operand (int or float).
        operation: Operation name as string.

    Returns:
        The result as float.

    Error Handling (Raises ValueError):
        Returns a structured error message if an unsupported
        operation is provided or if division by zero is attempted.
    """
    if operation == "add" or operation == "sum":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide" or operation == "div":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        result = a / b
    elif operation == "modulus" or operation == "mod":
        result = a % b
    else:
        err_msg = f"Unsupported operation: {operation}. "
        err_msg += "Use one of: add, subtract, multiply, divide, modulus "
        err_msg += "or try the 'wolfram_query' tool for complex math."
        raise ValueError(err_msg)

    return float(result)


@tool
def wolfram_query(expression: str) -> str:
    """
    Compute complex mathematical expressions or queries via the
    Wolfram Alpha API.

    Args:
        expression: A natural-language or Wolfram-style math query,
                    e.g. "integrate x^2", "solve x^2 + 3x + 2 = 0",
                    "derivative of sin(x)".

    Returns:
        The API’s primary result as text, or a structured error message.

    Error Handling:
        - Missing API key: returns an error string.
        - No result found: informs the user.
        - API errors or exceptions: returns the exception message.
    """
    app_id = os.getenv("WOLFRAM_APP_ID")
    if not app_id:
        return (
            "[Tool: wolfram_query ERROR] Missing WOLFRAM_APP_ID environment variable."
        )

    try:
        client = wolframalpha.Client(app_id)
        res = client.query(expression)
        # Attempt to get the first pod’s plaintext result
        pod = next(res.results, None)
        if pod is None or not hasattr(pod, "text") or not pod.text:
            return f"[Tool: wolfram_query] No result for query: '{expression}'."
        return f"[Tool: wolfram_query] {pod.text}"
    except StopIteration:
        return f"[Tool: wolfram_query] No results returned for '{expression}'."
    except Exception as e:
        return f"[Tool: wolfram_query ERROR] {str(e)}"
