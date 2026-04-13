"""JSON 遍历提取算子。"""
from typing import Any, Dict, List

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from .._common import column_dict_to_records, get_value, normalize_config_input


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
            "first_value": {},
            "second_value": {},
            "third_value": {},
            "fourth_value": {},
            "fifth_value": {},
            "sixth_value": {},
        },
    }
    default_config: Dict[str, Any] = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "table"}

    def _resolve_config(self, config):
        return normalize_config_input(super()._resolve_config(config))

    def execute(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        source_cfg = config.get("first_value")
        if source_cfg is None:
            raise OperatorException(
                "json_extract 缺少数据来源: first_value（可为字段名或 ${step_key}）",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )

        if isinstance(source_cfg, (dict, list)):
            raw = source_cfg
        else:
            raw = get_value(data, source_cfg, context)
        if raw is None:
            raise OperatorException(
                "json_extract 数据来源为空或未找到: first_value",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )

        selector = config.get("second_value")
        field_name = None
        fields = None
        if isinstance(selector, list):
            fields = selector
        elif isinstance(selector, str) and selector.strip():
            field_name = selector.strip()

        if _is_column_bundle(raw):
            d0 = raw[0]
            default = config.get("fifth_value")
            if not d0:
                return []
            n0 = len(next(iter(d0.values())))
            if isinstance(fields, list) and fields:
                keys = [str(k).strip() for k in fields if str(k).strip()]
                if not keys:
                    return []
                sub: Dict[str, List[Any]] = {}
                for k in keys:
                    if k in d0:
                        sub[k] = d0[k]
                    else:
                        sub[k] = [default] * n0
                return column_dict_to_records(sub)
            if field_name:
                fn = str(field_name)
                if fn in d0:
                    return column_dict_to_records({fn: d0[fn]})
                return [{fn: default} for _ in range(n0)]
            return column_dict_to_records(d0)

        nested_path = str(config.get("third_value") or "").strip()
        filter_cond = config.get("fourth_value") if isinstance(config.get("fourth_value"), dict) else {}
        default = config.get("fifth_value")
        mode = str(config.get("sixth_value") or "columns").lower()

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
                    return []
                if mode == "rows":
                    out_rows: List[Dict[str, Any]] = []
                    for item in raw:
                        if not isinstance(item, dict):
                            continue
                        if filter_cond and not _matches_filter(item, filter_cond):
                            continue
                        out_rows.append({k: _pick(item, k) for k in keys})
                    return out_rows
                out_cols: Dict[str, List[Any]] = {k: [] for k in keys}
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    if filter_cond and not _matches_filter(item, filter_cond):
                        continue
                    for k in keys:
                        out_cols[k].append(_pick(item, k))
                return column_dict_to_records(out_cols)

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
            return [{str(fn): v} for v in vals]

        if isinstance(raw, dict):
            if isinstance(fields, list) and fields:
                keys = [str(k).strip() for k in fields if str(k).strip()]
                if not keys:
                    return []
                inner = {k: _pick(raw, k) for k in keys}
                if inner and all(isinstance(v, list) for v in inner.values()):
                    return column_dict_to_records(inner)
                col_lists: Dict[str, List[Any]] = {
                    k: (v if isinstance(v, list) else [v]) for k, v in inner.items()
                }
                return column_dict_to_records(col_lists)
            if nested_path:
                v = _get_nested(raw, nested_path, default)
                if v is None:
                    return default
                vl = v if isinstance(v, list) else [v]
                nk = nested_path.split(".")[-1] or "value"
                return column_dict_to_records({nk: vl})
            if field_name:
                v = raw.get(field_name, default)
                if v is None:
                    return default
                vl = v if isinstance(v, list) else [v]
                return column_dict_to_records({str(field_name): vl})
            if raw and all(isinstance(v, list) for v in raw.values()):
                return column_dict_to_records(raw)
            return [raw]

        return default
