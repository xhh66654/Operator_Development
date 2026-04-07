"""复杂表达式解析：${step1} + ${step2} * 0.8 —— 使用 AST 安全求值，禁止任意代码执行"""
import ast
import operator as _op
import re
from typing import Any, Dict

_SAFE_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_SAFE_UNARYOPS = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}


def _safe_eval_node(node: ast.AST) -> float:
    """递归求值 AST 节点，仅允许数字与基本算术运算"""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    # Python 3.7 兼容
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_BINOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"不允许的运算符: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return float(op_func(left, right))
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_UNARYOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"不允许的一元运算符: {type(node.op).__name__}")
        return float(op_func(_safe_eval_node(node.operand)))
    raise ValueError(f"表达式中包含不允许的语法: {ast.dump(node)}")


def _safe_math_eval(expr_str: str) -> float:
    """安全的数学表达式求值，只允许数字和 +-*/% ** // 运算"""
    tree = ast.parse(expr_str.strip(), mode="eval")
    return _safe_eval_node(tree)


def resolve_expression(expr: str, context: Dict[str, Any]) -> Any:
    """
    将表达式中的 ${key} 替换为 context[key]，再安全求值（仅数学运算）。
    """
    if not isinstance(expr, str) or "${" not in expr:
        return expr

    def repl(match):
        k = match.group(1).strip()
        if k not in context:
            raise ValueError(f"表达式引用了不存在的步骤结果: ${{{k}}}")
        val = context[k]
        return str(val)

    resolved = re.sub(r"\$\{\s*([^}]+)\s*\}", repl, expr)
    return _safe_math_eval(resolved)
