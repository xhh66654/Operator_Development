"""统一输出包装：精简版，只返回 type 和 data。"""

from typing import Any, Dict

from .data_model import DataValue, TableValue

# 已无独立「连接算子」；保留 es_connect 便于若仍有该名算子时输出 type=connection
_CONNECTION_OPS = frozenset({"es_connect"})


def build_output_payload(value: Any, operator_name: str) -> Dict[str, Any]:
    """Build simplified output payload."""
    data_value = DataValue.from_python(value)
    out_type = data_value.type
    if out_type == "table":
        if isinstance(data_value, TableValue):
            out_type = "rows"
    if operator_name in _CONNECTION_OPS:
        out_type = "connection"
    return {
        "type": out_type,
        "data": data_value.to_dict(),
    }
