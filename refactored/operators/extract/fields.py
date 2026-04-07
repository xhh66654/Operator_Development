"""字段级提取：extract_only。"""
import math
from typing import Any, Dict, List

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...utils import extract_field_value
from .._common import _ctx, normalize_config_input, normalize_config_source_field, rows_to_field_list_dict


def clean_nan_records(records: List[Dict]) -> List[Dict]:
    """pandas 的 NaN/Infinity → None（供 excel 等复用）。"""
    def _fix(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return [{k: _fix(v) for k, v in row.items()} for row in records]


@OperatorRegistry.register("extract_only")
class ExtractOnlyOperator(BaseOperator):
    """仅提取字段；数据来源统一为 source / source_field / field。"""
    name = "extract_only"
    config_schema = {"type": "object", "properties": {"input": {}, "source": {}, "source_field": {}, "field": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "table"}

    def _resolve_config(self, config):
        c = normalize_config_input(super()._resolve_config(config))
        if c.get("field") in (None, "") and c.get("first_value") not in (None, ""):
            c["field"] = c.get("first_value")
        return normalize_config_source_field(
            c, canonical_key="field", legacy_keys=("input", "source_field", "source")
        )

    def execute(self, data, config, context: ExecutionContext):
        field = config.get("field") or config.get("source") or config.get("source_field")
        if field is None:
            return None
        raw = extract_field_value(data, field, _ctx(context))
        if raw is None:
            return [{}]
        if isinstance(raw, dict) and raw and all(isinstance(v, list) for v in raw.values()):
            return [raw]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and not (
            len(raw) == 1 and raw[0] and all(isinstance(v, list) for v in raw[0].values())
        ):
            return rows_to_field_list_dict(clean_nan_records(raw))
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
        return [{str(out_key): vals}]
