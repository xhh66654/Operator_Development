"""
比较与阈值算子：
  极值类   max / min
  阈值类   compare_threshold
  范围比较 支持多区间和单阈值
"""
import json
from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from ...utils import safe_convert_to_number
from .._common import get_value, to_number


def _flatten_to_numbers(val) -> list:
    """将单值、列表或 JSON 数组字符串摊平为数值列表；无法转数的跳过。"""
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for v in val:
            n = safe_convert_to_number(v)
            if n is not None:
                out.append(float(n))
        return out
    if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return _flatten_to_numbers(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    n = safe_convert_to_number(val)
    return [float(n)] if n is not None else []


def _collect_numbers(data, config, context, operator_name: str) -> list:
    """仅从 first_value/second_value/... 收集数据源并摊平为数值列表。"""
    numbers = _collect_numbers_from_seq_values(data, config, context, operator_name)
    if not numbers:
        raise OperatorException(
            "请至少提供 first_value（可继续提供 second_value/third_value...）作为数据来源",
            code=ErrorCode.CONFIG_MISSING,
            operator=operator_name,
            config=config,
        )
    return numbers


_SEQ_KEYS = [
    "first_value",
    "second_value",
    "third_value",
    "fourth_value",
    "fifth_value",
    "sixth_value",
    "seventh_value",
    "eighth_value",
    "ninth_value",
    "tenth_value",
]


def _collect_numbers_from_seq_values(data, config, context, operator_name: str) -> list:
    """从 first_value/second_value/... 收集并摊平为数值列表（支持单值、列表、JSON 数组字符串）。"""
    numbers = []
    for k in _SEQ_KEYS:
        if k not in config:
            continue
        ref = config.get(k)
        if ref in (None, ""):
            continue
        val = get_value(data, ref, context)
        numbers.extend(_flatten_to_numbers(val))
    return numbers


@OperatorRegistry.register("max")
class MaxOperator(BaseOperator):
    """
    最大值算子：找出数据中的最大值
    
    功能说明：
    - 从数据源中取最大值
    - 支持多个数据源，返回全局最大值
    
    计算逻辑：
    result = max(all_values)
    
    配置参数：
    - first_value/second_value/third_value...：按顺序提供待比较的数值来源（字段名、${step_key} 或字面量）
    
    输入数据格式：
    base_data: {
        "sales1": 1000,
        "sales2": 1500,
        "sales3": 800
    }
    
    配置示例：
    {
        "operator": "max",
        "config": {
            "first_value": "sales1",
            "second_value": "sales2",
            "third_value": "sales3"
        }
    }
    
    输出格式：
    1500 (数值)
    
    使用场景：
    - 最高销售额
    - 最高分数
    - 最大误差
    """
    name = "max"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {k: {} for k in _SEQ_KEYS},
    }
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        numbers = _collect_numbers_from_seq_values(data, config, context, self.name)
        if not numbers:
            val = get_value(data, config.get("first_value"), context)
            numbers = _flatten_to_numbers(val)
        if not numbers:
            raise OperatorException(
                "max 要求提供至少一个数值（first_value/second_value/...），可为单值或数值列表",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        return max(numbers)


@OperatorRegistry.register("min")
class MinOperator(BaseOperator):
    """
    最小值算子：找出数据中的最小值
    
    功能说明：
    - 从数据源中取最小值
    
    计算逻辑：
    result = min(all_values)
    
    配置参数：
    - first_value/second_value/third_value...：按顺序提供待比较的数值来源
    
    输入数据格式：
    base_data: {
        "price1": 99,
        "price2": 79,
        "price3": 199
    }
    
    配置示例：
    {
        "operator": "min",
        "config": {
            "first_value": "price1",
            "second_value": "price2",
            "third_value": "price3"
        }
    }
    
    输出格式：
    79 (数值)
    
    使用场景：
    - 最低价格
    - 最低分数
    - 最小值分析
    """
    name = "min"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {k: {} for k in _SEQ_KEYS},
    }
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        numbers = _collect_numbers_from_seq_values(data, config, context, self.name)
        if not numbers:
            val = get_value(data, config.get("first_value"), context)
            numbers = _flatten_to_numbers(val)
        if not numbers:
            raise OperatorException(
                "min 要求提供至少一个数值（first_value/second_value/...），可为单值或数值列表",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        return min(numbers)


@OperatorRegistry.register("compare_threshold")
class CompareThresholdOperator(BaseOperator):
    """
    阈值比较（统一参数名）：
    - first_value: 待比较值（接口字段名或 ${step_key}）
    - second_value: 阈值（用于单阈值模式）
    - third_value: compare_type（gt/lt/ge/le/eq，默认 eq）
    - fourth_value: result_mapping（如 {"true":"PASS","false":"FAIL"}）

    说明：仅支持顺序槽位 first_value/second_value/third_value/fourth_value。
    """
    name = "compare_threshold"
    config_schema = {
        "type": "object",
        "properties": {
            "first_value": {},
            "second_value": {},
            "third_value": {},
            "fourth_value": {},
            "ranges": {"type": "array"},
            "default_label": {"type": "string"},
        },
    }
    default_config = {"result_mapping": {"true": "是", "false": "否"}}

    def execute(self, data, config, context: ExecutionContext):
        source_ref = config.get("first_value")
        field_val = get_value(data, source_ref, context)
        if field_val is None:
            raise OperatorException(
                "待比较字段为空或未找到",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        # 支持数组整体传入：按顺序逐个比较，返回结果列表
        if isinstance(field_val, list):
            out_list = []
            for v in field_val:
                out_list.append(
                    self.execute(
                        {"_v": v},
                        {**config, "first_value": "_v"},
                        context,
                    )
                )
            return out_list

        value = to_number(field_val)

        ranges = config.get("ranges")
        if isinstance(ranges, str):
            try:
                import json as _json
                ranges = _json.loads(ranges)
            except (TypeError, ValueError, Exception):
                ranges = None
        if ranges and isinstance(ranges, list) and len(ranges) > 0:
            for i, r in enumerate(ranges):
                if not isinstance(r, dict):
                    continue
                try:
                    min_v = r.get("min")
                    max_v = r.get("max")
                    label = r.get("label") or ""
                    min_n = float(min_v) if min_v is not None and min_v != "" else None
                    max_n = float(max_v) if max_v is not None and max_v != "" else None
                except (TypeError, ValueError):
                    continue
                if min_n is None:
                    continue
                inclusive = r.get("inclusive", False)
                if max_n is None:
                    if value >= min_n:
                        return label
                else:
                    if inclusive:
                        if min_n <= value <= max_n:
                            return label
                    else:
                        if min_n <= value < max_n:
                            return label
            return config.get("default_label") or "未匹配"
        if ranges is not None and isinstance(ranges, list):
            return config.get("default_label") or "未匹配"

        # 统一格式（推荐）：允许用 threshold + compare_type + result_mapping 自动生成区间标签
        # 这样“单阈值”和“多区间”对外表现一致：都返回 label 字符串。
        threshold_ref = config.get("second_value", config.get("threshold"))
        threshold_raw = get_value(data, threshold_ref, context)
        if threshold_raw is None:
            raise OperatorException(
                "阈值为空或未找到（使用多区间时请配置 ranges，使用单阈值时请配置 second_value）",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        threshold_val = to_number(threshold_raw)
        compare_type = (config.get("third_value") or config.get("compare_type") or "eq").lower()
        result_mapping = (
            config.get("fourth_value")
            or config.get("result_mapping")
            or self.default_config.get("result_mapping")
            or {}
        )
        out = False
        if compare_type == "gt":
            out = value > threshold_val
        elif compare_type == "lt":
            out = value < threshold_val
        elif compare_type == "ge":
            out = value >= threshold_val
        elif compare_type == "le":
            out = value <= threshold_val
        elif compare_type == "eq":
            out = value == threshold_val
        # 兼容两种 mapping key：{"true": "..."} 或 {"True": "..."} 或 {"false": "..."}
        k = "true" if out else "false"
        return result_mapping.get(k, result_mapping.get(str(out).lower(), config.get("default_label") or "未知"))
