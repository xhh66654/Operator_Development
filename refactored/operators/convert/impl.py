"""类型转换算子：as_number, as_string, to_bool, split_string, join_list"""
import json
from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from ...utils import safe_convert_to_number
from .._common import get_value, normalize_config_source_field, normalize_punct


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


@OperatorRegistry.register("as_number")
class AsNumberOperator(BaseOperator):
    """字符串转数字；数据来源为 source/field。"""
    name = "as_number"
    config_schema = {"type": "object", "properties": {"field": {}, "source_field": {}, "source": {}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = normalize_config_source_field(
            super()._resolve_config(config), canonical_key="field", legacy_keys=("source_field", "source")
        )
        return merged

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("field") or config.get("source")
        val = get_value(data, ref, context)
        if val is None:
            return None
        if isinstance(val, list):
            return [safe_convert_to_number(v) for v in val]
        return safe_convert_to_number(val)


@OperatorRegistry.register("as_string")
class AsStringOperator(BaseOperator):
    """转文本；数据来源统一为 field，兼容 source_field。"""
    name = "as_string"
    config_schema = {"type": "object", "properties": {"field": {}, "source_field": {}, "source": {}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = normalize_config_source_field(
            super()._resolve_config(config), canonical_key="field", legacy_keys=("source_field", "source")
        )
        return merged

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("field") or config.get("source")
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
    config_schema = {"type": "object", "properties": {"field": {}, "source_field": {}, "source": {}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = normalize_config_source_field(
            super()._resolve_config(config), canonical_key="field", legacy_keys=("source_field", "source")
        )
        return merged

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("field") or config.get("source")
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
    """字符串→列表；数据来源为 source/field；separator 为前端配置。"""
    name = "split_string"
    config_schema = {"type": "object", "properties": {"field": {}, "source_field": {}, "source": {}, "separator": {}}}
    default_config = {}

    def _resolve_config(self, config):
        merged = normalize_config_source_field(
            super()._resolve_config(config), canonical_key="field", legacy_keys=("source_field", "source")
        )
        if merged.get("separator") in (None, "") and merged.get("second_value") not in (None, ""):
            merged["separator"] = merged.get("second_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("field") or config.get("source")
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
    """列表→字符串；数据来源为 source/field；separator、quote_elements 为前端配置。"""
    name = "join_list"
    config_schema = {"type": "object", "properties": {"field": {}, "source_field": {}, "source": {}, "separator": {}, "quote_elements": {}}}
    default_config = {"separator": ",", "quote_elements": False}

    def _resolve_config(self, config):
        merged = normalize_config_source_field(
            super()._resolve_config(config), canonical_key="field", legacy_keys=("source_field", "source")
        )
        if merged.get("separator") in (None, "") and merged.get("second_value") not in (None, ""):
            merged["separator"] = merged.get("second_value")
        if merged.get("quote_elements") in (None, "") and merged.get("third_value") not in (None, ""):
            merged["quote_elements"] = bool(merged.get("third_value"))
        return merged

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("field") or config.get("source")
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
