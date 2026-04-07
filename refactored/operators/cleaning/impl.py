"""数据清洗算子：filter_by_condition, select_fields, remove_duplicates, remove_nulls,
              remove_outliers, sort_data, limit_data, group_by, aggregate_by_group,
              type_conversion, rename_fields, merge_data, flatten_data"""
import math
import statistics
from typing import Any, Dict, List, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from ...core import BaseOperator, ExecutionContext, OperatorException, OperatorRegistry
from ...core.exceptions import ErrorCode
from ...utils import extract_field_value
from .._common import _ctx, get_value, normalize_config_source_field, rows_to_field_list_dict


def _to_list(raw_data: Any) -> List:
    if isinstance(raw_data, list):
        return raw_data
    return [raw_data] if raw_data is not None else []


def _is_columns_dict(x: Any) -> bool:
    """
    提取类算子输出解包后的列字典形态：
    { "fieldA": [v1,v2,...], "fieldB": [v1,v2,...] }
    """
    return isinstance(x, dict) and (not x or all(isinstance(v, list) for v in x.values()))


def _unwrap_column_bundle_to_columns_dict(raw_source: Any) -> Any:
    """
    兼容两种“列包”形态：
    1) {field: [values...]}  （列字典）
    2) [{field: [values...]}]（列包 list 包一层）
    返回列字典；若不是列包则返回 None。
    """
    if _is_columns_dict(raw_source):
        return raw_source
    if isinstance(raw_source, list) and len(raw_source) == 1 and isinstance(raw_source[0], dict):
        d0 = raw_source[0]
        if d0 and all(isinstance(v, list) for v in d0.values()):
            return d0
        if d0 == {}:
            return {}
    return None


def _columns_dict_to_rows(columns: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """列字典转行列表：{col:[...]} -> [{col:v,...}, ...]"""
    if not columns:
        return []
    keys = list(columns.keys())
    lengths = [len(columns[k]) for k in keys]
    if lengths and len(set(lengths)) != 1:
        raise OperatorException(
            f"列包格式列长度不一致，无法对齐：{ {k: len(columns[k]) for k in keys} }",
            code=ErrorCode.SCHEMA_MISMATCH,
            operator="cleaning",
        )
    n = lengths[0] if lengths else 0
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append({k: columns[k][i] for k in keys})
    return out


def _maybe_return_columns(input_is_columns: bool, rows: List[Dict[str, Any]]):
    return rows_to_field_list_dict(rows) if input_is_columns else rows


def _resolve_item_value(item: dict, field_name: str):
    """从记录中取字段值，支持字段名去首尾引号后匹配。返回 (value, key_exists)。"""
    s = (field_name or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if s in item:
        return item[s], True
    if field_name in item:
        return item[field_name], True
    return None, False


def _compare_for_filter(item_value: Any, op: str, target: Any) -> bool:
    """做一次条件比较；对 gt/lt/ge/le 尝试数值比较，避免 CSV 字符串与数字无法比。"""
    # 新增：区间比较（支持多个区间 OR）
    # - between/range: target 可以是 [min, max] 或 {"min":..,"max":..} 或 {"ranges":[{min,max},...]}
    if op in ("between", "range"):
        try:
            iv = float(item_value) if item_value is not None else None
        except (TypeError, ValueError):
            return False
        if iv is None:
            return False
        ranges = None
        if isinstance(target, dict) and "ranges" in target:
            ranges = target.get("ranges")
        elif isinstance(target, list) and len(target) == 2:
            ranges = [{"min": target[0], "max": target[1]}]
        elif isinstance(target, dict) and ("min" in target or "max" in target):
            ranges = [target]
        if not isinstance(ranges, list) or not ranges:
            return False
        for r in ranges:
            if not isinstance(r, dict):
                continue
            lo = r.get("min", float("-inf"))
            hi = r.get("max", float("inf"))
            try:
                lo_n = float(lo) if lo is not None and lo != "" else float("-inf")
                hi_n = float(hi) if hi is not None and hi != "" else float("inf")
            except (TypeError, ValueError):
                continue
            if lo_n <= iv <= hi_n:
                return True
        return False

    if op in ("gt", "lt", "ge", "le"):
        try:
            iv = float(item_value) if item_value is not None else None
            tv = float(target) if target is not None else None
            if iv is not None and tv is not None:
                if op == "gt": return iv > tv
                if op == "lt": return iv < tv
                if op == "ge": return iv >= tv
                if op == "le": return iv <= tv
        except (TypeError, ValueError):
            pass
    if op == "eq": return item_value == target
    if op == "ne": return item_value != target
    if op == "gt": return item_value is not None and target is not None and item_value > target
    if op == "lt": return item_value is not None and target is not None and item_value < target
    if op == "ge": return item_value is not None and target is not None and item_value >= target
    if op == "le": return item_value is not None and target is not None and item_value <= target
    if op == "in": return item_value in target if isinstance(target, (list, tuple)) else False
    if op == "not_in": return item_value not in target if isinstance(target, (list, tuple)) else True
    if op in ("contains", "includes"):
        return str(target) in str(item_value) if item_value is not None else False
    if op in ("not_contains", "excludes"):
        return str(target) not in str(item_value) if item_value is not None else True
    return False


def _is_blank_field(field_name: str) -> bool:
    """字段名为空或仅空白时视为「按元素本身」条件，用于单纯数组。"""
    return not (field_name or "").strip()


@OperatorRegistry.register("filter_by_condition")
class FilterByConditionOperator(BaseOperator):
    """按条件过滤：支持记录列表（按字段）或单纯数组（按元素值）。数据来源统一为 source_field，兼容 field/list_field/source。"""
    name = "filter_by_condition"
    config_schema = {"type": "object", "properties": {"source_field": {}, "field": {}, "list_field": {}, "source": {},  "conditions": {}}}
    default_config = {"conditions": {}}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["conditions"] = merged.get("second_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("source_field"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        input_is_columns = columns_dict is not None
        raw_data = _columns_dict_to_rows(columns_dict) if input_is_columns else _to_list(raw_source)
        raw_conditions = config.get("conditions", {})

        cond_list: list = self._normalize_conditions(raw_conditions)

        out = []
        for item in raw_data:
            if isinstance(item, dict):
                if self._match_record(item, cond_list):
                    out.append(item)
            else:
                if self._match_scalar(item, cond_list):
                    out.append(item)
        return _maybe_return_columns(input_is_columns, out)

    @staticmethod
    def _normalize_conditions(raw) -> list:
        """
        统一 conditions 为 [{field, operator, value}, ...] 列表。
        兼容三种入参：
          1. list[{field, operator, value}]           — 新数组格式
          2. dict[field_name -> {operator, value}]    — 旧 dict 格式
          3. dict[field_name -> plain_value]           — 极简旧格式
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            result = []
            for field_name, condition in raw.items():
                if isinstance(condition, dict):
                    result.append({
                        "field": field_name,
                        "operator": condition.get("operator", "eq"),
                        "value": condition.get("value"),
                    })
                else:
                    result.append({"field": field_name, "operator": "eq", "value": condition})
            return result
        return []

    @staticmethod
    def _match_record(item: dict, cond_list: list) -> bool:
        for c in cond_list:
            field_name = c.get("field", "")
            if _is_blank_field(field_name):
                continue
            item_value, key_exists = _resolve_item_value(item, field_name)
            if not key_exists:
                return False
            op = c.get("operator", "eq")
            target = c.get("value")
            if not _compare_for_filter(item_value, op, target):
                return False
        return True

    @staticmethod
    def _match_scalar(item, cond_list: list) -> bool:
        for c in cond_list:
            field_name = c.get("field", "")
            if not _is_blank_field(field_name):
                continue
            op = c.get("operator", "eq")
            target = c.get("value")
            if not _compare_for_filter(item, op, target):
                return False
        return True


def _normalize_field_name(name: str) -> str:
    """去掉字段名首尾的多余引号，避免前端传 "\\"姓名\\"" 导致与 CSV 列名 姓名 不匹配。"""
    if not name or not isinstance(name, str):
        return name
    s = name.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s


@OperatorRegistry.register("select_fields")
class SelectFieldsOperator(BaseOperator):
    """保留指定字段算子；数据来源统一为 source_field，兼容 field/list_field/source。"""
    name = "select_fields"
    config_schema = {"type": "object", "properties": {"source_field": {}, "field": {}, "list_field": {}, "source": {},  "fields": {"type": "array"}}}
    default_config = {"fields": []}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["fields"] = merged.get("second_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("source_field"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        if columns_dict is not None:
            fields = config.get("fields", [])
            if not fields:
                return [columns_dict]
            normalized = [_normalize_field_name(f) for f in fields]
            out_dict: Dict[str, Any] = {}
            for orig, norm in zip(fields, normalized):
                key = norm if norm in columns_dict else (orig if orig in columns_dict else norm)
                if key in columns_dict:
                    out_dict[norm] = columns_dict[key]
            return [out_dict] if out_dict else [{}]

        raw_data = _to_list(raw_source)
        fields = config.get("fields", [])
        if not fields:
            return raw_data
        # 规范化字段名，便于匹配 CSV 列名（如 姓名 而不是 "姓名"）
        normalized = [_normalize_field_name(f) for f in fields]

        def _resolve_key(orig: str, item: dict) -> str:
            n = _normalize_field_name(orig)
            if n in item:
                return n
            if orig in item:
                return orig
            return n

        out = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            row = {}
            for i, f in enumerate(fields):
                key = _resolve_key(f, item)
                if key in item:
                    row[normalized[i] if i < len(normalized) else key] = item[key]
            if row:
                out.append(row)
        if len(fields) == 1:
            key_use = normalized[0]
            return [item.get(key_use) if key_use in item else item.get(fields[0]) for item in raw_data if isinstance(item, dict) and (key_use in item or fields[0] in item)]
        return out


@OperatorRegistry.register("remove_duplicates")
class RemoveDuplicatesOperator(BaseOperator):
    """去重算子；数据来源统一为 source_field，兼容 field/list_field/source。"""
    name = "remove_duplicates"
    config_schema = {"type": "object", "properties": {"source_field": {}, "field": {}, "list_field": {}, "source": {},  "key_fields": {"type": "array"}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["key_fields"] = merged.get("second_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("source_field"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        input_is_columns = columns_dict is not None
        raw_data = _columns_dict_to_rows(columns_dict) if input_is_columns else _to_list(raw_source)
        key_fields = config.get("key_fields") or config.get("fields")
        seen = set()
        out = []
        for item in raw_data:
            try:
                if key_fields:
                    if not isinstance(item, dict):
                        continue
                    key = tuple(item.get(f) for f in key_fields if f in item)
                else:
                    key = tuple(sorted(item.items())) if isinstance(item, dict) else item
                if key not in seen:
                    seen.add(key)
                    out.append(item)
            except TypeError:
                import json
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    out.append(item)
        return _maybe_return_columns(input_is_columns, out if input_is_columns else out)


@OperatorRegistry.register("remove_nulls")
class RemoveNullsOperator(BaseOperator):
    """空值处理算子；数据来源统一为 source_field，兼容 field/list_field/source。"""
    name = "remove_nulls"
    config_schema = {"type": "object", "properties": {"source_field": {}, "field": {}, "list_field": {}, "source": {},  "fields": {}, "strategy": {}, "fill_value": {}, "null_values": {}}}
    default_config = {"strategy": "remove", "fill_value": 0, "null_values": [None, "", "null", "NULL", "N/A", "n/a"]}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["fields"] = merged.get("second_value")
        if merged.get("third_value") not in (None, ""):
            merged["strategy"] = merged.get("third_value")
        if merged.get("fourth_value") not in (None, ""):
            merged["null_values"] = merged.get("fourth_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("source_field"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        input_is_columns = columns_dict is not None
        raw_data = _columns_dict_to_rows(columns_dict) if input_is_columns else _to_list(raw_source)
        fields_to_check = config.get("fields")
        strategy = config.get("strategy", "remove")
        fill_value = config.get("fill_value", 0)
        null_values = config.get("null_values", [None, "", "null", "NULL", "N/A", "n/a"])

        def _is_null(v):
            if v is None:
                return True
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return True
            return v in null_values

        out = []
        for item in raw_data:
            if not isinstance(item, dict):
                if not _is_null(item):
                    out.append(item)
                elif strategy == "fill":
                    out.append(fill_value)
                continue
            check_fields = fields_to_check or list(item.keys())
            item_copy = item.copy()
            has_null = False
            for field in check_fields:
                if field in item and _is_null(item[field]):
                    has_null = True
                    if strategy == "fill":
                        item_copy[field] = fill_value
            if strategy == "remove" and has_null:
                continue
            out.append(item_copy)
        return _maybe_return_columns(input_is_columns, out)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if _HAS_NUMPY:
        return float(np.percentile(values, p))
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


@OperatorRegistry.register("remove_outliers")
class RemoveOutliersOperator(BaseOperator):
    """
    异常值处理算子。
    - 标量列表（如 [85,72,22]）：直接对数值列表做 IQR/zscore 过滤，无需指定字段。
    - 记录列表（list of dict）：需通过 field/target_fields 指定要检测的字段名。
    IQR 方法：下界 = Q1 - threshold*IQR，上界 = Q3 + threshold*IQR，默认 threshold=1.5。
    zscore 方法：|z| > threshold 视为异常，默认 threshold=3。
    """
    name = "remove_outliers"
    config_schema = {"type": "object", "properties": {
        "source_field": {}, "field": {}, "list_field": {}, "source": {},
        "target_fields": {}, "method": {}, "threshold": {}, "manual_ranges": {}, "strategy": {}
    }}
    default_config = {"method": "iqr", "threshold": 1.5, "strategy": "remove"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["target_fields"] = merged.get("second_value")
        if merged.get("third_value") not in (None, ""):
            merged["method"] = merged.get("third_value")
        if merged.get("fourth_value") not in (None, ""):
            merged["threshold"] = merged.get("fourth_value")
        return merged

    @staticmethod
    def _compute_bounds(values: list, method: str, threshold: float) -> tuple:
        """返回 (lower, upper) 合法范围边界。"""
        if method == "iqr":
            q1, q3 = _percentile(values, 25), _percentile(values, 75)
            iqr = q3 - q1
            return q1 - threshold * iqr, q3 + threshold * iqr
        else:  # zscore
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            return mean_val - threshold * std_val, mean_val + threshold * std_val

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("source_field"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        input_is_columns = columns_dict is not None
        raw_data = _columns_dict_to_rows(columns_dict) if input_is_columns else _to_list(raw_source)
        if not raw_data:
            return _maybe_return_columns(input_is_columns, raw_data)

        method = config.get("method", "iqr")
        threshold = float(config.get("threshold", 1.5 if method == "iqr" else 3.0))
        strategy = config.get("strategy", "remove")
        manual_ranges = config.get("manual_ranges", {})

        is_scalar_list = all(isinstance(item, (int, float)) for item in raw_data)
        if is_scalar_list:
            if manual_ranges:
                lo = manual_ranges.get("min", float("-inf"))
                hi = manual_ranges.get("max", float("inf"))
            else:
                lo, hi = self._compute_bounds(raw_data, method, threshold)
            out = []
            for v in raw_data:
                if lo <= v <= hi:
                    out.append(v)
                elif strategy == "clip":
                    out.append(max(lo, min(hi, v)))
                # strategy == "remove": 丢弃
            return _maybe_return_columns(input_is_columns, out)


        target_fields = list(config.get("target_fields") or [])
        if not target_fields and config.get("field"):
            target_fields = [config["field"]]
        if not target_fields:
            return _maybe_return_columns(input_is_columns, raw_data)

        field_bounds: dict = {}
        if manual_ranges:
            field_bounds = manual_ranges
        else:
            for field in target_fields:
                values = [
                    item[field] for item in raw_data
                    if isinstance(item, dict) and field in item and isinstance(item[field], (int, float))
                ]
                if values:
                    lo, hi = self._compute_bounds(values, method, threshold)
                    field_bounds[field] = {"min": lo, "max": hi}

        out = []
        for item in raw_data:
            if not isinstance(item, dict):
                out.append(item)
                continue
            item_copy = item.copy()
            is_outlier = False
            for field in target_fields:
                if field not in item or field not in field_bounds:
                    continue
                val = item[field]
                if not isinstance(val, (int, float)):
                    continue
                lo = field_bounds[field].get("min", float("-inf"))
                hi = field_bounds[field].get("max", float("inf"))
                if val < lo or val > hi:
                    if strategy == "clip":
                        item_copy[field] = max(lo, min(hi, val))
                    else:
                        is_outlier = True
                        break
            if not is_outlier or strategy == "clip":
                out.append(item_copy)
        return out


@OperatorRegistry.register("sort_data")
class SortDataOperator(BaseOperator):
    """排序算子；数据来源统一为 source_field，兼容 field/list_field/source。"""
    name = "sort_data"
    config_schema = {"type": "object", "properties": {"source_field": {}, "field": {}, "list_field": {}, "source": {},  "sort_by": {}, "order": {}, "null_placement": {}}}
    default_config = {"order": "asc", "null_placement": "last"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["sort_by"] = merged.get("second_value")
        if merged.get("third_value") not in (None, ""):
            merged["order"] = "asc" if bool(merged.get("third_value")) else "desc"
        if merged.get("fourth_value") not in (None, ""):
            merged["null_placement"] = merged.get("fourth_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("source_field"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        input_is_columns = columns_dict is not None
        raw_data = _columns_dict_to_rows(columns_dict) if input_is_columns else _to_list(raw_source)
        if not raw_data:
            return _maybe_return_columns(input_is_columns, raw_data)
        sort_by = config.get("sort_by")
        order = config.get("order", "asc")
        null_placement = config.get("null_placement", "last")
        sort_by_list = [sort_by] if isinstance(sort_by, str) and sort_by else (sort_by or [])
        order_list = [order] if isinstance(order, str) else (order or [])
        if sort_by_list and len(order_list) < len(sort_by_list):
            order_list += [order_list[0]] * (len(sort_by_list) - len(order_list))

        def key_fn(item):
            parts = []
            for field in sort_by_list:
                val = item.get(field) if isinstance(item, dict) else None
                if isinstance(val, (int, float)):
                    val = float(val)
                elif isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = val.lower()
                elif val is not None:
                    val = str(val)
                is_none = val is None
                safe_val = float('-inf') if is_none else val
                if null_placement == "last":
                    parts.append((is_none, safe_val))
                else:
                    parts.append((not is_none, safe_val))
            return tuple(parts) if parts else (0, str(item))

        reverse = order_list[0].lower() == "desc" if order_list else False
        try:
            out = sorted(raw_data, key=key_fn, reverse=reverse)
            return _maybe_return_columns(input_is_columns, out)
        except TypeError:
            out = sorted(raw_data, key=lambda x: str(x).lower(), reverse=reverse)
            return _maybe_return_columns(input_is_columns, out)


@OperatorRegistry.register("rename_fields")
class RenameFieldsOperator(BaseOperator):
    """字段重命名；数据来源统一为 source_field，兼容 field/list_field/source。"""
    name = "rename_fields"
    config_schema = {"type": "object", "properties": {"source_field": {}, "field": {}, "list_field": {}, "source": {},  "mappings": {}}}
    default_config = {"mappings": {}}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["mappings"] = merged.get("second_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_data = _to_list(extract_field_value(data, config.get("source_field"), _ctx(context)))
        mappings = config.get("mappings", {})
        out = []
        for item in raw_data:
            if not isinstance(item, dict):
                out.append(item)
                continue
            out.append({mappings.get(k, k): v for k, v in item.items()})
        return out


@OperatorRegistry.register("merge_data")
class MergeDataOperator(BaseOperator):
    """多源合并"""
    name = "merge_data"
    config_schema = {"type": "object", "properties": {"sources": {"type": "array"}, "merge_type": {}, "join_key": {}}}
    default_config = {"sources": [], "merge_type": "concat"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        if merged.get("sources") in (None, "") and merged.get("first_value") not in (None, "") and merged.get("second_value") not in (None, ""):
            merged["sources"] = [merged.get("first_value"), merged.get("second_value")]
        if merged.get("merge_type") in (None, "") and merged.get("fourth_value") not in (None, ""):
            mv = str(merged.get("fourth_value")).strip().lower()
            # 文档里 inner/left/right/outer 统一映射到 join
            merged["merge_type"] = "join" if mv in {"inner", "left", "right", "outer", "join"} else mv
        if merged.get("join_key") in (None, "") and merged.get("third_value") not in (None, ""):
            jk = merged.get("third_value")
            if isinstance(jk, list) and jk:
                merged["join_key"] = str(jk[0])
            else:
                merged["join_key"] = str(jk) if jk not in (None, "") else jk
        return merged

    def execute(self, data, config, context: ExecutionContext):
        sources = config.get("sources", [])
        merge_type = config.get("merge_type", "concat")
        join_key = config.get("join_key")
        ctx = _ctx(context)
        if merge_type == "concat":
            out = []
            for src in sources:
                v = extract_field_value(data, src, ctx)
                if isinstance(v, list):
                    out.extend(v)
                elif v is not None:
                    out.append(v)
            return out
        if merge_type == "join" and join_key and len(sources) >= 2:
            base = _to_list(extract_field_value(data, sources[0], ctx))
            for src in sources[1:]:
                join_data = _to_list(extract_field_value(data, src, ctx))
                index = {it[join_key]: it for it in join_data if isinstance(it, dict) and join_key in it}
                merged = []
                for b in base:
                    if isinstance(b, dict) and join_key in b and b[join_key] in index:
                        merged.append({**b, **index[b[join_key]]})
                    else:
                        merged.append(b)
                base = merged
            return base
        return []


@OperatorRegistry.register("flatten_data")
class FlattenDataOperator(BaseOperator):
    """嵌套列表打平；数据来源统一为 source_field，兼容 field/list_field/source。举例：[[1,2],[3,4]] → [1,2,3,4]"""
    name = "flatten_data"
    config_schema = {"type": "object", "properties": {"source_field": {}, "field": {}, "list_field": {}, "source": {},  "nested_field": {}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = normalize_config_source_field(merged)
        if merged.get("second_value") not in (None, ""):
            merged["nested_field"] = merged.get("second_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_data = _to_list(extract_field_value(data, config.get("source_field"), _ctx(context)))
        nested_field = config.get("nested_field")
        out = []
        for item in raw_data:
            if isinstance(item, list):
                out.extend(item)
                continue
            if not isinstance(item, dict):
                out.append(item)
                continue
            if nested_field and nested_field in item and isinstance(item[nested_field], list):
                for nested_item in item[nested_field]:
                    flat = {k: v for k, v in item.items() if k != nested_field}
                    if isinstance(nested_item, dict):
                        flat.update(nested_item)
                    else:
                        flat[f"{nested_field}_value"] = nested_item
                    out.append(flat)
            else:
                out.append(item)
        return out
