"""字段提取（带缓存、少拷贝）"""
import json
from typing import Any, Dict, List, Tuple, Union, Optional

_parsed_cache: Dict[str, Any] = {}
_CACHE_MAX = 1000


def _cache_key(obj: Any) -> Optional[str]:
    if isinstance(obj, str) and (obj.strip().startswith("{") or obj.strip().startswith("[")):
        return obj
    return None


def _unwrap_column_bundle(raw: Any) -> Any:
    """若为上一步提取类输出的 [{字段: 取值列表}]，解包为 {字段: 列表} 便于按列名取值。"""
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        d0 = raw[0]
        if d0 and all(isinstance(v, list) for v in d0.values()):
            return d0
    return raw


def _column_array_from_row_list(raw: Any, subfield: str) -> Optional[List[Any]]:
    """
    当 context[step] 为行列表 Record[] 时，将 `${step}.列名` 解析为该列在行序上的取值数组，
    与列包解包后 `dict[列名]` 的形态一致，便于下游计算算子统一使用。

    与空列包 `[{}]` 区分：单行空对象不按行列表取列（返回 None）。
    与列包 `[{col: [...]}]` 区分：若唯一一行且该行所有 value 均为 list，视为列包（应由 _unwrap 处理），此处返回 None。
    """
    if not isinstance(raw, list) or not subfield:
        return None
    if not raw or not all(isinstance(r, dict) for r in raw):
        return None
    if len(raw) == 1 and raw[0] == {}:
        return None
    if len(raw) == 1:
        d0 = raw[0]
        if d0 and all(isinstance(v, list) for v in d0.values()):
            return None
    return [r.get(subfield) for r in raw]


def _parse_context_ref(field_name: str) -> Optional[Tuple[str, Optional[str]]]:
    """${step_key} 或 ${step_key}.列名 → (step_key, subfield|None)。"""
    if not (isinstance(field_name, str) and field_name.startswith("${")):
        return None
    inner = field_name.strip()
    if not inner.startswith("${"):
        return None
    rest = inner[2:]
    if "}" not in rest:
        return None
    step_key, _, tail = rest.partition("}")
    step_key = step_key.strip()
    if not step_key:
        return None
    subfield: Optional[str] = None
    if tail.startswith("."):
        sub = tail[1:].strip()
        subfield = sub if sub else None
    return step_key, subfield


def extract_field_value(
    data: Union[Dict, List, str],
    field_name: str,
    context: Optional[Dict[str, Any]] = None,
) -> Union[float, int, List, None]:
    """
    从任意结构中提取字段值；支持上下文引用 ${step_key}、${step_key}.列名。
    列名引用在「列包」与「行列表」两种上一步输出上一致：均得到该列的取值数组（行序一致）。
    若 data 为已解析过的结构，不会重复解析。
    """
    cref = _parse_context_ref(field_name) if context is not None else None
    if cref is not None:
        step_key, subfield = cref
        raw = context.get(step_key, None)
        raw = _unwrap_column_bundle(raw)
        if subfield:
            if isinstance(raw, dict) and subfield in raw:
                return raw[subfield]
            row_col = _column_array_from_row_list(raw, subfield)
            if row_col is not None:
                return row_col
            return None
        return raw

    # DB-only 模式：若传的是裸字段名，且当前 record 中找不到，
    # 则尝试从“最新提取列数据”中取值。
    if (
        context is not None
        and isinstance(field_name, str)
        and not field_name.startswith("${")
        and isinstance(context.get("_latest_columns"), dict)
        and field_name in context["_latest_columns"]
    ):
        return context["_latest_columns"][field_name]

    if isinstance(data, str):
        k = _cache_key(data)
        if k is not None and k in _parsed_cache:
            data = _parsed_cache[k]
        else:
            try:
                data = json.loads(data)
                if k is not None and len(_parsed_cache) < _CACHE_MAX:
                    _parsed_cache[k] = data
            except json.JSONDecodeError:
                return None

    data_list = data if isinstance(data, list) else [data]
    values: List[Any] = []

    def recursive_extract(obj: Any) -> None:
        if isinstance(obj, dict):
            if field_name in obj:
                val = obj[field_name]
                if isinstance(val, str):
                    stripped = val.strip()
                    try:
                        f = float(stripped)
                        values.append(int(f) if f == int(f) and "." not in stripped else f)
                    except (ValueError, TypeError):
                        values.append(val)  # 日期时间字符串等原样返回
                elif isinstance(val, (int, float, list)):
                    values.append(val)
                else:
                    values.append(val)
            else:
                for v in obj.values():
                    recursive_extract(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item)

    for item in data_list:
        recursive_extract(item)

    if len(values) == 0:
        try:
            return float(field_name)
        except (ValueError, TypeError):
            return None
    if len(values) == 1:
        return values[0]
    return values


def safe_convert_to_number(value: Any) -> Union[int, float, None]:
    """安全转为数值"""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value) if "." in value else int(value)
        except (ValueError, TypeError):
            return None
    return None
