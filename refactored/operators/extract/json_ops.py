"""JSON 遍历提取算子。"""
from typing import Any, Dict, List

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from .._common import get_value, normalize_config_input, rows_to_field_list_dict


def _is_column_bundle(x: Any) -> bool:
    return (
        isinstance(x, list)
        and len(x) == 1
        and isinstance(x[0], dict)
        and (not x[0] or all(isinstance(v, list) for v in x[0].values()))
    )


def _get_nested(obj: Dict, path: str, default: Any = None) -> Any:
    if not path or not isinstance(obj, dict):
        return default
    current = obj
    for key in path.split("."):
        key = key.strip()
        if not key:
            continue
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _matches_filter(row: Dict, filter_cond: Dict) -> bool:
    if not filter_cond or not isinstance(row, dict):
        return True
    for key, expected in filter_cond.items():
        if row.get(key) != expected:
            return False
    return True


@OperatorRegistry.register("json_extract")
class JsonExtractOperator(BaseOperator):
    name = "json_extract"
    config_schema = {
        "type": "object",
        "properties": {
            "input": {},
            "source": {},
            "field": {"type": "string"},
            "fields": {"type": "array"},
            "nested": {"type": "string"},
            "filter": {"type": "object"},
            "default": {},
            "mode": {"type": "string"},
        },
    }
    default_config: Dict[str, Any] = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "table"}

    def _resolve_config(self, config):
        c = normalize_config_input(super()._resolve_config(config))
        if c.get("source") in (None, "") and c.get("input") not in (None, ""):
            c["source"] = c["input"]
        # 兼容顺序参数（文档使用 first_value/second_value/third_value）
        # first_value -> source
        # second_value -> fields（数组）或 field（字符串）
        # third_value -> nested
        if c.get("source") in (None, "") and c.get("first_value") not in (None, ""):
            c["source"] = c.get("first_value")
        if c.get("fields") in (None, "") and c.get("field") in (None, "") and c.get("second_value") not in (None, ""):
            sv = c.get("second_value")
            if isinstance(sv, list):
                c["fields"] = sv
            elif isinstance(sv, str) and sv.strip():
                c["field"] = sv.strip()
        if c.get("nested") in (None, "") and c.get("third_value") not in (None, ""):
            c["nested"] = c.get("third_value")
        return c

    def execute(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        source_cfg = config.get("source")
        if source_cfg is None:
            raise OperatorException(
                "json_extract 缺少数据来源: source（可为字段名或 ${step_key}）",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )

        raw = get_value(data, source_cfg, context)
        if raw is None:
            return config.get("default")

        if _is_column_bundle(raw):
            d0 = raw[0]
            field_name = config.get("field")
            fields = config.get("fields")
            default = config.get("default")
            if isinstance(fields, list) and fields:
                keys = [str(k).strip() for k in fields if str(k).strip()]
                return [{k: d0.get(k, default) for k in keys}] if keys else [{}]
            if field_name:
                return [{str(field_name): d0.get(field_name, default)}]
            return raw

        field_name = config.get("field")
        fields = config.get("fields")
        nested_path = config.get("nested") or ""
        filter_cond = config.get("filter") if isinstance(config.get("filter"), dict) else {}
        default = config.get("default")
        mode = (config.get("mode") or "columns").lower()

        def _pick(item: Dict, key: str):
            if isinstance(key, str) and "." in key:
                return _get_nested(item, key, default)
            if nested_path:
                return _get_nested(item, nested_path, default)
            return item.get(key, default)

        if isinstance(raw, list):
            if isinstance(fields, list) and fields:
                keys = [str(k).strip() for k in fields if str(k).strip()]
                if not keys:
                    return [{}]
                if mode == "rows":
                    out_rows: List[Dict[str, Any]] = []
                    for item in raw:
                        if not isinstance(item, dict):
                            continue
                        if filter_cond and not _matches_filter(item, filter_cond):
                            continue
                        out_rows.append({k: _pick(item, k) for k in keys})
                    return rows_to_field_list_dict(out_rows)
                out_cols: Dict[str, List[Any]] = {k: [] for k in keys}
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    if filter_cond and not _matches_filter(item, filter_cond):
                        continue
                    for k in keys:
                        out_cols[k].append(_pick(item, k))
                return [out_cols]

            out: List[Any] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if filter_cond and not _matches_filter(item, filter_cond):
                    continue
                if nested_path:
                    val = _get_nested(item, nested_path, default)
                elif field_name:
                    val = item.get(field_name, default)
                else:
                    val = item
                out.append(val)
            fn = field_name or "value"
            vals = out if isinstance(out, list) else [out]
            return [{str(fn): vals}]

        if isinstance(raw, dict):
            if isinstance(fields, list) and fields:
                keys = [str(k).strip() for k in fields if str(k).strip()]
                if not keys:
                    return [{}]
                inner = {k: _pick(raw, k) for k in keys}
                if inner and all(isinstance(v, list) for v in inner.values()):
                    return [inner]
                col_lists: Dict[str, List[Any]] = {
                    k: (v if isinstance(v, list) else [v]) for k, v in inner.items()
                }
                return [col_lists]
            if nested_path:
                v = _get_nested(raw, nested_path, default)
                if v is None:
                    return config.get("default")
                vl = v if isinstance(v, list) else [v]
                return [{nested_path.split(".")[-1] or "value": vl}]
            if field_name:
                v = raw.get(field_name, default)
                if v is None:
                    return config.get("default")
                vl = v if isinstance(v, list) else [v]
                return [{str(field_name): vl}]
            if raw and all(isinstance(v, list) for v in raw.values()):
                return [raw]
            return rows_to_field_list_dict([raw])

        return default
