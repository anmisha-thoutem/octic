import ast
import math

_SAFE_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return _ALLOWED_BINOPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return _ALLOWED_UNARYOPS[op_type](operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        if len(node.args) != 1 or node.keywords:
            raise ValueError("Functions must accept exactly one positional argument.")
        return _SAFE_FUNCTIONS[func_name](_safe_eval(node.args[0]))

    if isinstance(node, ast.Name):
        if node.id not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported name: {node.id}")
        return _SAFE_FUNCTIONS[node.id]

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_expression(expression: str):
    if expression is None or not str(expression).strip():
        raise ValueError("Expression cannot be empty.")

    cleaned_expression = str(expression).strip()
    try:
        parsed = ast.parse(cleaned_expression, mode="eval")
        result = _safe_eval(parsed.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        raise ValueError("Invalid mathematical expression.")

    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result
