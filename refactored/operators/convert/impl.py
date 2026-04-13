"""类型转换算子：as_number, as_string, to_bool, split_string, join_list, rows_to_columns, columns_to_rows"""
import json
from typing import Any, Dict, List, Optional
from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from ...utils import safe_convert_to_number
from .._common import get_value, normalize_punct


def _ensure_list(val):
    """若为 JSON 数组字符串（如 "[1,2,3]"）则解析为 list，否则原样返回。用于前端传字符串形式的列表。"""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                pass
    return val


def _is_columns_dict(x: Any) -> bool:
    """列式结构：{col: [v1, v2, ...], ...}"""
    return isinstance(x, dict) and (not x or all(isinstance(v, list) for v in x.values()))


def _unwrap_column_bundle_to_columns_dict(raw: Any) -> Optional[Dict[str, List[Any]]]:
    """
    兼容提取类列包输出：
    - {col: [..]} -> 原样返回
    - [{col: [..]}] -> 解包返回
    - [{}] -> 返回 {}
    """
    if _is_columns_dict(raw):
        return raw  # type: ignore[return-value]
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        d0 = raw[0]
        if d0 == {}:
            return {}
        if d0 and all(isinstance(v, list) for v in d0.values()):
            return d0  # type: ignore[return-value]
    return None


def _rows_to_columns(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """行式结构 -> 列式结构。缺失字段用 None 补齐。"""
    if not rows:
        return {}
    keys: List[str] = []
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            raise OperatorException(
                "rows_to_columns 要求输入为 List[Dict]（行式结构）",
                code=ErrorCode.TYPE_ERROR,
            )
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    out: Dict[str, List[Any]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            out[k].append(r.get(k))
    return out


def _columns_to_rows(columns: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """列式结构 -> 行式结构。列长度必须一致。"""
    if not columns:
        return []
    if not _is_columns_dict(columns):
        raise OperatorException(
            "columns_to_rows 要求输入为 Dict[str, List]（列式结构）",
            code=ErrorCode.TYPE_ERROR,
        )
    keys = list(columns.keys())
    lengths = [len(columns[k]) for k in keys]
    if lengths and len(set(lengths)) != 1:
        raise OperatorException(
            f"列式结构列长度不一致，无法对齐：{ {k: len(columns[k]) for k in keys} }",
            code=ErrorCode.SCHEMA_MISMATCH,
        )
    n = lengths[0] if lengths else 0
    return [{k: columns[k][i] for k in keys} for i in range(n)]


def _input_value(data: Dict[str, Any], ref: Any, context: ExecutionContext) -> Any:
    """
    convert 类算子常见用法：
    - ref 为字符串：字段名或 ${step} 引用 → 走 get_value
    - ref 为 list/dict：前端直接传字面量 → 直接返回（避免 get_value 把 dict 当字段名查找）
    """
    if ref is None:
        return None
    if isinstance(ref, (list, dict)):
        return ref
    return get_value(data, ref, context)


@OperatorRegistry.register("as_number")
class AsNumberOperator(BaseOperator):
    """字符串转数字；数据来源为 first_value。"""
    name = "as_number"
    config_schema = {"type": "object", "properties": {"first_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("first_value")
        val = get_value(data, ref, context)
        if val is None:
            return None
        if isinstance(val, list):
            return [safe_convert_to_number(v) for v in val]
        return safe_convert_to_number(val)


@OperatorRegistry.register("as_string")
class AsStringOperator(BaseOperator):
    """转文本；数据来源为 first_value。"""
    name = "as_string"
    config_schema = {"type": "object", "properties": {"first_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("first_value")
        val = get_value(data, ref, context)
        if val is None:
            return None
        if isinstance(val, list):
            return [str(v) for v in val]
        return str(val) if val is not None else None


@OperatorRegistry.register("to_bool")
class ToBoolOperator(BaseOperator):
    """
    转布尔算子：支持单值或列表转换
    
    功能说明：
    - 单值：直接转换为布尔
    - 列表：逐元素转换，返回布尔值列表
    
    真值定义：1/true/"1"/"true"/"yes"/"是"（不区分大小写）
    其他值均为 False
    """
    name = "to_bool"
    config_schema = {"type": "object", "properties": {"first_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("first_value")
        val = get_value(data, ref, context)
        if val is None:
            return None
        
        # 新增：支持列表输入
        if isinstance(val, list):
            return [self._to_bool_single(v) for v in val]
        
        # 单值模式
        return self._to_bool_single(val)
    
    def _to_bool_single(self, val):
        """将单个值转为布尔"""
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val == 1 or val == 1.0
        if isinstance(val, str):
            s = val.strip().lower()
            return s in ("true", "1", "yes", "是")
        n = safe_convert_to_number(val)
        return n == 1 if n is not None else False


@OperatorRegistry.register("split_string")
class SplitStringOperator(BaseOperator):
    """字符串→列表；数据来源为 first_value；separator 为 second_value。"""
    name = "split_string"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "separator": {}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        if merged.get("separator") in (None, "") and merged.get("second_value") not in (None, ""):
            merged["separator"] = merged.get("second_value")
            merged["second_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("first_value")
        val = get_value(data, ref, context)
        if val is None:
            return None
        if not isinstance(val, str):
            raise OperatorException(
                "split_string 要求数据来源为字符串",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        sep = config.get("separator")
        if sep is None or sep == "":
            return list(val)
        # 同时规范化数据中的中文标点，使 "a，b，c".split(",") 能正常工作
        return normalize_punct(val).split(sep)


@OperatorRegistry.register("join_list")
class JoinListOperator(BaseOperator):
    """列表→字符串；数据来源为 first_value；separator 为 second_value；quote_elements 为 third_value。"""
    name = "join_list"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "separator": {}, "quote_elements": {}}}
    default_config = {"separator": ",", "quote_elements": False}

    def _resolve_config(self, config):
        merged = dict(super()._resolve_config(config))
        if merged.get("separator") in (None, "") and merged.get("second_value") not in (None, ""):
            merged["separator"] = merged.get("second_value")
            merged["second_value"] = None
        if merged.get("quote_elements") in (None, "") and merged.get("third_value") not in (None, ""):
            merged["quote_elements"] = bool(merged.get("third_value"))
            merged["third_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("first_value")
        val = get_value(data, ref, context)
        if val is None:
            return None
        list_val = _ensure_list(val)
        if not isinstance(list_val, list):
            raise OperatorException(
                "join_list 要求数据来源为列表或 JSON 数组字符串（如 \"[1,2,3]\"）",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        sep = config.get("separator") or ","
        quote = config.get("quote_elements", False)
        if quote:
            return sep.join(f'"{v}"' for v in list_val)
        return sep.join(str(v) for v in list_val)


@OperatorRegistry.register("rows_to_columns")
class RowsToColumnsOperator(BaseOperator):
    """
    行式结构 -> 列式结构：
    - 输入：[{min:0,label:"D"}, {min:60,label:"C"}]
    - 输出：{min:[0,60], label:["D","C"]}
    也兼容提取类列包 [{col:[...]}]，会原样输出为列字典。
    """
    name = "rows_to_columns"
    config_schema = {"type": "object", "properties": {"first_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("first_value")
        raw = _input_value(data, ref, context)
        cols = _unwrap_column_bundle_to_columns_dict(raw)
        if cols is not None:
            return cols
        if raw is None:
            return {}
        if not isinstance(raw, list):
            raise OperatorException(
                "rows_to_columns 要求输入为 List[Dict]（行式结构）或列包 [{col:[...]}]",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        return _rows_to_columns(raw)


@OperatorRegistry.register("columns_to_rows")
class ColumnsToRowsOperator(BaseOperator):
    """
    列式结构 -> 行式结构：
    - 输入：{min:[0,60], label:["D","C"]}
    - 输出：[{min:0,label:"D"}, {min:60,label:"C"}]
    也兼容提取类列包 [{col:[...]}]。
    """
    name = "columns_to_rows"
    config_schema = {"type": "object", "properties": {"first_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("first_value")
        raw = _input_value(data, ref, context)
        cols = _unwrap_column_bundle_to_columns_dict(raw)
        if cols is None:
            if raw is None:
                return []
            if not isinstance(raw, dict):
                raise OperatorException(
                    "columns_to_rows 要求输入为 Dict[str, List]（列式结构）或列包 [{col:[...]}]",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            cols = raw
        return _columns_to_rows(cols)
