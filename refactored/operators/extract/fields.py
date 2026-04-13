"""字段级提取：extract_only。"""
import math
from typing import Any, Dict, List

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...utils import extract_field_value
from .._common import _ctx, column_dict_to_records, normalize_config_input


def clean_nan_records(records: List[Dict]) -> List[Dict]:
    """pandas 的 NaN/Infinity → None（供 excel 等复用）。"""
    def _fix(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return [{k: _fix(v) for k, v in row.items()} for row in records]


@OperatorRegistry.register("extract_only")
class ExtractOnlyOperator(BaseOperator):
    """仅提取字段；数据来源使用顺序参数 first_value。"""
    name = "extract_only"
    config_schema = {"type": "object", "properties": {"first_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "table"}

    def _resolve_config(self, config):
        return normalize_config_input(super()._resolve_config(config))

    def execute(self, data, config, context: ExecutionContext):
        field = config.get("first_value")
        if field is None:
            return None
        raw = extract_field_value(data, field, _ctx(context))
        if raw is None:
            return []
        if isinstance(raw, dict) and raw and all(isinstance(v, list) for v in raw.values()):
            return column_dict_to_records(raw)
        if (
            isinstance(raw, list)
            and len(raw) == 1
            and isinstance(raw[0], dict)
            and raw[0]
            and all(isinstance(v, list) for v in raw[0].values())
        ):
            return column_dict_to_records(raw[0])
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return clean_nan_records(raw)
        out_key = config.get("output_key")
        if not out_key and isinstance(field, str):
            fk = field.strip()
            if fk.startswith("${") and fk.endswith("}"):
                inner = fk[2:-1].strip()
                if "." in inner:
                    out_key = inner.split(".")[-1].strip() or "value"
                else:
                    out_key = "value"
            else:
                out_key = fk.split(".")[-1] if "." in fk else fk
        else:
            out_key = out_key or "value"
        vals = raw if isinstance(raw, list) else [raw]
        return [{str(out_key): v} for v in vals]
