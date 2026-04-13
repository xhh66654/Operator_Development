"""数据清洗算子：filter_by_condition, select_fields, remove_duplicates, remove_nulls,
              remove_outliers, sort_data, limit_data, group_by, aggregate_by_group,
              type_conversion, rename_fields, merge_data, flatten_data"""
import json
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
from .._common import _ctx, column_dict_to_records, get_value, _looks_like_context_ref


def _to_list(raw_data: Any) -> List:
    if isinstance(raw_data, list):
        return raw_data
    return [raw_data] if raw_data is not None else []


def _looks_like_column_bundle(raw: Any) -> bool:
    """列包 / 列字典：本模块不再支持，仅用于拒绝。"""
    if isinstance(raw, dict) and (not raw or all(isinstance(v, list) for v in raw.values())):
        return True
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        d0 = raw[0]
        if d0 and all(isinstance(v, list) for v in d0.values()):
            return True
    return False


def _require_rows(raw: Any, *, operator: str) -> List[Any]:
    """清洗算子输入须为行式 List[Dict]（可为空列表）；不接受列包。"""
    if raw is None:
        return []
    if _looks_like_column_bundle(raw):
        raise OperatorException(
            f"算子 {operator} 仅支持行式 List[Dict]，不再支持列包/列字典",
            code=ErrorCode.SCHEMA_MISMATCH,
            operator=operator,
        )
    if isinstance(raw, list):
        return raw
    return [raw]


def _is_columns_dict(x: Any) -> bool:
    """列字典：{ 列名: [每行值…] }。"""
    return isinstance(x, dict) and (not x or all(isinstance(v, list) for v in x.values()))


def _unwrap_column_bundle_to_columns_dict(raw_source: Any) -> Any:
    """列字典或单元素列包 [{col: [...]}] → 列字典；否则返回 None。"""
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
    """列字典转行列表。"""
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


def _dedupe_key_hashable(parts: Any) -> Any:
    """去重用键：可 hash 则原样返回，否则 JSON 序列化（避免 list/dict 作 tuple 元素导致 unhashable）。"""
    try:
        hash(parts)
        return parts
    except TypeError:
        return json.dumps(parts, sort_keys=True, default=str)


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


_FILTER_SLOT_OPS = frozenset({
    "eq", "ne", "gt", "ge", "lt", "le", "in", "not_in", "between", "range",
    "contains", "includes", "not_contains", "excludes",
})


@OperatorRegistry.register("filter_by_condition")
class FilterByConditionOperator(BaseOperator):
    """按条件过滤：记录列表或标量数组。

    主路径为「四槽」（整表过滤、输出保留行内全部列）：
    - first_value：行表数据源
    - second_value：比较列名（**字符串**），或 **字符串数组**（多列 AND，共用 third/fourth）
    - third_value：比较符（ge/gt/eq/...）
    - fourth_value：比较右值

    亦可传显式 ``conditions``，或由 second 提供条件对象列表 / 字段条件字典（见 _resolve_config）。
    """
    name = "filter_by_condition"
    config_schema = {
        "type": "object",
        "properties": {
            "first_value": {},
            "second_value": {},
            "third_value": {},
            "fourth_value": {},
            "conditions": {},
        },
    }
    default_config = {"conditions": {}}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        raw_conds = merged.get("conditions")
        has_explicit = False
        if isinstance(raw_conds, list) and len(raw_conds) > 0:
            has_explicit = True
        if isinstance(raw_conds, dict) and len(raw_conds) > 0:
            has_explicit = True

        if has_explicit:
            return merged

        sv0 = merged.get("second_value")
        fv0 = merged.get("first_value")
        tv = merged.get("third_value")

        # second 为「条件对象」列表（每项为 dict）→ 直接作为 conditions
        if isinstance(sv0, list) and len(sv0) > 0 and all(isinstance(x, dict) for x in sv0):
            merged["conditions"] = sv0
            merged["second_value"] = None
            return merged

        # 新：整表 + second 为字段名列表 + third 为比较符 + fourth 为比较值（输出仍保留行内全部列）
        if (
            isinstance(sv0, list)
            and len(sv0) > 0
            and all(isinstance(x, str) for x in sv0)
            and any(str(x).strip() for x in sv0)
            and isinstance(tv, str)
            and tv.strip().lower() in _FILTER_SLOT_OPS
        ):
            op = tv.strip().lower()
            rhs = merged.get("fourth_value")
            merged["conditions"] = [
                {
                    "field": _normalize_field_name(str(x).strip()),
                    "operator": op,
                    "value": rhs,
                }
                for x in sv0
                if isinstance(x, str) and str(x).strip()
            ]
            merged["second_value"] = None
            merged["third_value"] = None
            merged["fourth_value"] = None
            return merged

        if isinstance(sv0, dict):
            merged["conditions"] = sv0
            merged["second_value"] = None
            return merged

        tv = merged.get("third_value")
        sv = merged.get("second_value")
        op_third = isinstance(tv, str) and tv.strip().lower() in _FILTER_SLOT_OPS
        op_second = isinstance(sv, str) and sv.strip().lower() in _FILTER_SLOT_OPS

        if (
            fv0 not in (None, "")
            and op_third
            and isinstance(sv, str)
            and not op_second
        ):
            merged["conditions"] = [
                {
                    "field": _normalize_field_name(sv.strip()),
                    "operator": tv.strip().lower(),
                    "value": merged.get("fourth_value"),
                }
            ]
            merged["second_value"] = None
            merged["third_value"] = None
            merged["fourth_value"] = None
            return merged

        return merged

    def execute(self, data, config, context: ExecutionContext):
        ctx = _ctx(context)
        fv = config.get("first_value")
        raw_source = extract_field_value(data, fv, ctx)
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        raw_data = _columns_dict_to_rows(columns_dict) if columns_dict is not None else _to_list(raw_source)
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
        return out

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


def _select_field_names_from_config(data: Dict, sec: Any, context: ExecutionContext) -> List:
    """
    select_fields 的字段列表：纯字符串列名（无 ${}）按字面量使用，避免 get_value 在整表上
    递归同名嵌套字段把 second_value 误展开成数值列表。
    """
    if sec in (None, ""):
        return []
    if isinstance(sec, list) and sec and all(isinstance(x, str) for x in sec):
        if any(_looks_like_context_ref(str(x)) for x in sec):
            return get_value(data, sec, context) or []
        return list(sec)
    return get_value(data, sec, context) or []


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
    """保留指定字段算子；数据来源为 first_value；字段列表由 second_value 提供。"""
    name = "select_fields"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("first_value"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        if columns_dict is not None:
            fields = _select_field_names_from_config(
                data, config.get("second_value"), context
            )
            if not fields:
                return column_dict_to_records(columns_dict)
            normalized = [_normalize_field_name(f) for f in fields]
            out_dict: Dict[str, Any] = {}
            for orig, norm in zip(fields, normalized):
                key = norm if norm in columns_dict else (orig if orig in columns_dict else norm)
                if key in columns_dict:
                    out_dict[norm] = columns_dict[key]
            return column_dict_to_records(out_dict) if out_dict else []

        raw_data = _to_list(raw_source)
        fields = _select_field_names_from_config(data, config.get("second_value"), context)
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
        # 始终返回行式 List[Dict]；单列时不再退化为标量列表（避免丢失同表其它列的语义期望）。
        return out


@OperatorRegistry.register("remove_duplicates")
class RemoveDuplicatesOperator(BaseOperator):
    """去重算子；数据来源为 first_value；key_fields 为 second_value；third_value 为 first/last 保留策略。"""
    name = "remove_duplicates"
    config_schema = {
        "type": "object",
        "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "key_fields": {"type": "array"}},
    }
    default_config = {}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        if merged.get("second_value") not in (None, ""):
            merged["key_fields"] = merged.get("second_value")
            merged["second_value"] = None
        tv = merged.get("third_value")
        if tv not in (None, ""):
            s = str(tv).strip().lower()
            if s in ("first", "last"):
                merged["dedupe_keep"] = s
            merged["third_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("first_value"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        raw_data = _columns_dict_to_rows(columns_dict) if columns_dict is not None else _to_list(raw_source)
        keep = str(config.get("dedupe_keep") or "first").strip().lower()
        if keep == "last":
            raw_data = list(reversed(raw_data))
        key_fields = config.get("key_fields")
        seen = set()
        out = []
        for item in raw_data:
            if key_fields:
                if not isinstance(item, dict):
                    continue
                parts = tuple(item.get(f) for f in key_fields if f in item)
                key = _dedupe_key_hashable(parts)
            else:
                if isinstance(item, dict):
                    key = _dedupe_key_hashable(tuple(sorted(item.items())))
                else:
                    key = _dedupe_key_hashable(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
        if keep == "last":
            out = list(reversed(out))
        return out


@OperatorRegistry.register("remove_nulls")
class RemoveNullsOperator(BaseOperator):
    """空值处理算子；数据来源为 first_value；检查字段/策略/空值集合由 second/third/fourth_value 提供。"""
    name = "remove_nulls"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "strategy": {}, "fill_value": {}, "null_values": {}}}
    default_config = {"strategy": "remove", "fill_value": 0, "null_values": [None, "", "null", "NULL", "N/A", "n/a"]}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        if merged.get("second_value") not in (None, ""):
            merged["fields_to_check"] = merged.get("second_value")
            merged["second_value"] = None
        if merged.get("third_value") not in (None, ""):
            merged["strategy"] = merged.get("third_value")
            merged["third_value"] = None
        if merged.get("fourth_value") not in (None, ""):
            merged["null_values"] = merged.get("fourth_value")
            merged["fourth_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = get_value(data, config.get("first_value"), context)
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        raw_data = _columns_dict_to_rows(columns_dict) if columns_dict is not None else _to_list(raw_source)
        fields_to_check = config.get("fields_to_check")
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
        return out


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
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {},
        "target_fields": {}, "method": {}, "threshold": {}, "manual_ranges": {}, "strategy": {}
    }}
    default_config = {"method": "iqr", "threshold": 1.5, "strategy": "remove"}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        if merged.get("second_value") not in (None, ""):
            merged["target_fields"] = merged.get("second_value")
            merged["second_value"] = None
        if merged.get("third_value") not in (None, ""):
            merged["method"] = merged.get("third_value")
            merged["third_value"] = None
        if merged.get("fourth_value") not in (None, ""):
            merged["threshold"] = merged.get("fourth_value")
            merged["fourth_value"] = None
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
        raw_source = extract_field_value(data, config.get("first_value"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        raw_data = _columns_dict_to_rows(columns_dict) if columns_dict is not None else _to_list(raw_source)
        if not raw_data:
            return raw_data

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
            return out


        target_fields = list(config.get("target_fields") or [])
        if not target_fields:
            return raw_data

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
    """排序算子；数据来源为 first_value；sort_by/order/null_placement 分别来自 second/third/fourth_value。"""
    name = "sort_data"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "sort_by": {}, "order": {}, "null_placement": {}}}
    default_config = {"order": "asc", "null_placement": "last"}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        if merged.get("second_value") not in (None, ""):
            merged["sort_by"] = merged.get("second_value")
            merged["second_value"] = None
        if merged.get("third_value") not in (None, ""):
            tv = merged.get("third_value")
            if isinstance(tv, str) and tv.strip().lower() in ("asc", "desc"):
                merged["order"] = tv.strip().lower()
            else:
                merged["order"] = "asc" if bool(tv) else "desc"
            merged["third_value"] = None
        if merged.get("fourth_value") not in (None, ""):
            merged["null_placement"] = merged.get("fourth_value")
            merged["fourth_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_source = extract_field_value(data, config.get("first_value"), _ctx(context))
        columns_dict = _unwrap_column_bundle_to_columns_dict(raw_source)
        raw_data = _columns_dict_to_rows(columns_dict) if columns_dict is not None else _to_list(raw_source)
        if not raw_data:
            return raw_data
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
            return out
        except TypeError:
            out = sorted(raw_data, key=lambda x: str(x).lower(), reverse=reverse)
            return out


@OperatorRegistry.register("rename_fields")
class RenameFieldsOperator(BaseOperator):
    """字段重命名；数据来源为 first_value；``second_value``+``third_value`` 为单字段改名，或 ``second_value`` 为映射表 dict。"""
    name = "rename_fields"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "mappings": {}}}
    default_config = {"mappings": {}}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        sec = merged.get("second_value")
        third = merged.get("third_value")
        if isinstance(sec, dict):
            merged["mappings"] = sec
            merged["second_value"] = None
        elif isinstance(sec, str) and sec.strip():
            if isinstance(third, str) and third.strip():
                merged["mappings"] = {sec.strip(): third.strip()}
                merged["second_value"] = None
                merged["third_value"] = None
            else:
                raise OperatorException(
                    "rename_fields：second_value 为原列名（字符串）时，必须同时提供 third_value 为新列名",
                    code=ErrorCode.CONFIG_MISSING,
                    operator=self.name,
                    config=config,
                )
        elif sec not in (None, ""):
            merged["mappings"] = sec
            merged["second_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_data = _to_list(extract_field_value(data, config.get("first_value"), _ctx(context)))
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
    """多源合并（仅支持顺序槽位）"""
    name = "merge_data"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}, "third_value": {}},
    }
    default_config = {"merge_type": "concat"}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        # second_value / third_value 为显式顺序槽位输入，应覆盖默认值
        if merged.get("second_value") not in (None, ""):
            merged["merge_type"] = merged.get("second_value")
            merged["second_value"] = None
        if merged.get("third_value") not in (None, ""):
            merged["join_key"] = merged.get("third_value")
            merged["third_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        sources_cfg = config.get("first_value")
        if isinstance(sources_cfg, list):
            # sources 本身就是列表字面量（最常见），不要走 get_value 的“列表合并”语义
            sources_raw = list(sources_cfg)
        else:
            # 支持 sources 从 data/ctx 中取到一个列表（例如某一步计算得到的 sources 列表）
            sources_raw = get_value(data, sources_cfg, context)
        if sources_raw is None:
            raise OperatorException(
                "merge_data 缺少 sources：first_value 必须提供来源列表（字段名、${step_key} 或字面量列表）",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        if not isinstance(sources_raw, list) or len(sources_raw) == 0:
            raise OperatorException(
                "merge_data 的 first_value 必须为非空列表（sources）",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )

        sources = sources_raw
        merge_type_raw = config.get("merge_type", "concat")
        merge_type = str(merge_type_raw).strip().lower() if merge_type_raw not in (None, "") else "concat"

        if merge_type == "concat":
            out = []
            for i, src in enumerate(sources):
                v = get_value(data, src, context)
                if v is None:
                    raise OperatorException(
                        f"merge_data concat 第{i + 1}个 source 取不到值: {src!r}",
                        code=ErrorCode.DATA_NOT_FOUND,
                        operator=self.name,
                        config=config,
                    )
                if isinstance(v, list):
                    out.extend(v)
                else:
                    out.append(v)
            return out

        if merge_type == "join":
            if len(sources) < 2:
                raise OperatorException(
                    "merge_data join 至少需要 2 个 source",
                    code=ErrorCode.CONFIG_MISSING,
                    operator=self.name,
                    config=config,
                )
            join_key_raw = config.get("join_key")
            join_key = str(join_key_raw).strip() if join_key_raw not in (None, "") else ""
            if not join_key:
                raise OperatorException(
                    "merge_data join 缺少 join_key：third_value 必须提供 join_key",
                    code=ErrorCode.CONFIG_MISSING,
                    operator=self.name,
                    config=config,
                )

            base0 = get_value(data, sources[0], context)
            if base0 is None:
                raise OperatorException(
                    f"merge_data join 第1个 source 取不到值: {sources[0]!r}",
                    code=ErrorCode.DATA_NOT_FOUND,
                    operator=self.name,
                    config=config,
                )
            base = _to_list(base0)
            # 为了保证输出可被 TableValue 表示，join 后需要对齐所有行的字段集合（缺失字段补 None）。
            union_keys = set()
            for r in base:
                if isinstance(r, dict):
                    union_keys.update(r.keys())

            for i, src in enumerate(sources[1:], start=2):
                join_raw = get_value(data, src, context)
                if join_raw is None:
                    raise OperatorException(
                        f"merge_data join 第{i}个 source 取不到值: {src!r}",
                        code=ErrorCode.DATA_NOT_FOUND,
                        operator=self.name,
                        config=config,
                    )
                join_data = _to_list(join_raw)

                index = {}
                join_keys = set()
                for it in join_data:
                    if not isinstance(it, dict) or join_key not in it:
                        continue
                    join_keys.update(it.keys())
                    k = it.get(join_key)
                    try:
                        hash(k)
                    except Exception as e:
                        raise OperatorException(
                            f"merge_data join 的 join_key 值不可作为 key（不可哈希）: {k!r}",
                            code=ErrorCode.TYPE_ERROR,
                            operator=self.name,
                            config=config,
                            cause=e,
                        )
                    index[k] = it

                merged_rows = []
                for b in base:
                    if isinstance(b, dict) and join_key in b and b.get(join_key) in index:
                        merged_rows.append({**b, **index[b.get(join_key)]})
                    else:
                        merged_rows.append(b)
                base = merged_rows
                union_keys.update(join_keys)

            # 补齐缺失字段，避免 TableValue 校验失败
            if union_keys:
                normalized = []
                for r in base:
                    if not isinstance(r, dict):
                        normalized.append(r)
                        continue
                    rr = dict(r)
                    for k in union_keys:
                        if k not in rr:
                            rr[k] = None
                    normalized.append(rr)
                base = normalized
            return base

        raise OperatorException(
            f"merge_data 不支持的 merge_type: {merge_type!r}（仅支持 concat/join）",
            code=ErrorCode.CONFIG_INVALID,
            operator=self.name,
            config=config,
        )


@OperatorRegistry.register("flatten_data")
class FlattenDataOperator(BaseOperator):
    """嵌套列表打平；数据来源为 first_value；nested_field 为 second_value。举例：[[1,2],[3,4]] → [1,2,3,4]"""
    name = "flatten_data"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "nested_field": {}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        if merged.get("second_value") not in (None, ""):
            merged["nested_field"] = merged.get("second_value")
            merged["second_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        raw_data = _to_list(get_value(data, config.get("first_value"), context))
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
