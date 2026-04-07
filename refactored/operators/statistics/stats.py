"""
基础统计算子：
  均值类   arithmetic_mean / harmonic_mean / geometric_mean / weighted_average (average)
  分布类   median / mode / range / std_dev / variance / cv
  分位数   percentile / quartiles / iqr
  加权类   weighted_variance / weighted_std / weighted_sum_squares / sum_squares
  比率类   growth_rate / decline_rate / percentage、proportion_by_count、percentage_by_count
  计数类   count_items

相关性与误差指标 → correlation.py
归一化          → normalization.py
"""
import json
import logging
import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Union

from ...utils import safe_convert_to_number

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from .._common import get_value, _ctx, to_number, normalize_config_to_fields

logger = logging.getLogger(__name__)


def _ensure_number_list(v: Any) -> List[float]:
    """
    将取值转为数值列表。支持：标量、列表、JSON 数组字符串如 "[1,2,3]"。
    用于平均值等统计算子，可正确计算 [] 里的数。
    """
    from ...utils import safe_convert_to_number

    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend(_ensure_number_list(x))
        return out
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return _ensure_number_list(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
    n = safe_convert_to_number(v)
    if n is not None:
        return [float(n)]
    raise OperatorException(
        f"统计算子无法将值转为数字: {v!r}",
        code=ErrorCode.FORMAT_ERROR,
    )


def _collect_values(data: Dict, config: Dict, context: ExecutionContext) -> List[float]:
    """
    从 config 的 operands/fields 收集数值（数据来源：接口字段名或 ${step_key}）。
    _resolve_config 已将 operands/field 合并进 fields，此处读 fields，缺省时回退 operands/field。
    """
    values: List[float] = []
    fields = config.get("fields") or config.get("inputs") or config.get("operands") or []

    if not fields:
        field = config.get("field")
        if field is None:
            field = config.get("input")
        if field is not None:
            fields = field if isinstance(field, list) else [field]
    # 兼容顺序参数：允许统计类算子直接用 first_value/second_value/... 指定数据来源（含 ${step}.field）。
    if not fields:
        seq_keys = [
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
        seq_fields = [config.get(k) for k in seq_keys if config.get(k) not in (None, "")]
        if seq_fields:
            fields = seq_fields

    def extend_with(val: Any) -> None:
        values.extend(_ensure_number_list(val))

    for f in fields:
        v = get_value(data, f, context)
        if v is None:
            # 配置里显式声明了数据来源（字段名或 ${step_key}），但取不到值时不要静默吞掉，
            # 否则上层会把“取值失败”误判为“无数据=0.0”。
            raise OperatorException(
                f"统计数据来源为空或未找到: {f}",
                code=ErrorCode.DATA_NOT_FOUND,
            )
        extend_with(v)
    return values


def _weights_flat(
    data: Dict[str, Any],
    weights_entries: Any,
    context: ExecutionContext,
    target_len: int,
    *,
    operator: str,
    config: Dict[str, Any],
) -> List[float]:
    """
    将 `weights` 配置（可含多个 `${step}.列名` 等引用）展开为与样本等长的权重列表。
    每项引用若解析为列向量，则按 `_ensure_number_list` 展平后依次拼接。
    展开后 **必须与样本数 target_len 完全一致**，不再用末项重复补齐或截断，避免静默错配。
    """
    if target_len <= 0:
        return []
    if not weights_entries:
        return [1.0] * target_len
    entries = weights_entries if isinstance(weights_entries, list) else [weights_entries]
    out: List[float] = []
    for w in entries:
        raw = get_value(data, w, context) if isinstance(w, str) else w
        out.extend(_ensure_number_list(raw))
    if len(out) != target_len:
        raise OperatorException(
            f"{operator} 权重项数必须与样本数一致（当前样本 n={target_len}，展开后权重数={len(out)}）；"
            "请为每条样本提供对应权重，勿依赖自动补齐。",
            code=ErrorCode.SCHEMA_MISMATCH,
            operator=operator,
            config=config,
        )
    # 权重归一化：前端可传任意权重（不要求和为 1），统一除以权重总和。
    total_w = sum(out)
    if total_w != 0:
        out = [w / total_w for w in out]
    return out


def _resolve_first_second_values_weights(merged: Dict[str, Any]) -> Dict[str, Any]:
    """
    顺序参数：first_value=观测值，second_value=权重。
    若不处理，_collect_values 会把 first/second 都当样本拼接；weights 为空时 _weights_flat 会退化为全 1；非空权重须与样本等长。
    若已配置非空 weights，仍强制 fields=[first_value]，避免 second_value 被当第二路数据。
    """
    fv = merged.get("first_value")
    sv = merged.get("second_value")
    if fv in (None, "") or sv in (None, ""):
        return merged
    out = dict(merged)
    out["fields"] = [fv]
    if not out.get("weights"):
        out["weights"] = sv
    return out


def _list_element_count(raw: Any) -> float:
    """用于「按条数」比例算子：列表用 len，非列表视为 1 条，None 视为 0。"""
    if raw is None:
        return 0.0
    if isinstance(raw, list):
        return float(len(raw))
    return 1.0


def _scalar_or_number_vector(
    val: Any,
    *,
    label: str,
    operator: str,
    config: Dict[str, Any],
) -> Union[float, List[float]]:
    """增长率/下降率：标量或数值数组（嵌套列表会按 _ensure_number_list 展平）。"""
    if val is None:
        raise OperatorException(
            f"{operator} 的 {label} 未取到值",
            code=ErrorCode.DATA_NOT_FOUND,
            operator=operator,
            config=config,
        )
    if isinstance(val, list):
        nums = _ensure_number_list(val)
        if not nums:
            raise OperatorException(
                f"{operator} 的 {label} 不能为空列表",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=operator,
                config=config,
            )
        return nums
    n = safe_convert_to_number(val)
    if n is None:
        raise OperatorException(
            f"{operator} 的 {label} 无法转为数字: {val!r}",
            code=ErrorCode.FORMAT_ERROR,
            operator=operator,
            config=config,
        )
    return float(n)


def _growth_rate_pairwise(
    old: Union[float, List[float]],
    new: Union[float, List[float]],
    *,
    operator: str,
    config: Dict[str, Any],
) -> Union[float, List[float]]:
    """(现期 − 基期) / 基期 × 100；支持标量↔数组广播、等长数组逐元素。"""

    def rate(o: float, n: float) -> float:
        if o == 0:
            return 0.0
        return ((n - o) / o) * 100.0

    if isinstance(old, float) and isinstance(new, float):
        return rate(old, new)
    if isinstance(old, float) and isinstance(new, list):
        return [rate(old, n) for n in new]
    if isinstance(old, list) and isinstance(new, float):
        return [rate(o, new) for o in old]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            raise OperatorException(
                f"{operator} 在数组对数组时要求长度一致（基期 {len(old)} 个，现期 {len(new)} 个）",
                code=ErrorCode.CONFIG_INVALID,
                operator=operator,
                config=config,
            )
        return [rate(o, n) for o, n in zip(old, new)]
    raise OperatorException(
        f"{operator} 内部类型组合异常",
        code=ErrorCode.CALC_LOGIC_ERROR,
        operator=operator,
        config=config,
    )


def _decline_rate_pairwise(
    old: Union[float, List[float]],
    new: Union[float, List[float]],
    *,
    operator: str,
    config: Dict[str, Any],
) -> Union[float, List[float]]:
    """(基期 − 现期) / 基期 × 100；广播规则同 growth_rate。"""

    def rate(o: float, n: float) -> float:
        if o == 0:
            return 0.0
        return ((o - n) / o) * 100.0

    if isinstance(old, float) and isinstance(new, float):
        return rate(old, new)
    if isinstance(old, float) and isinstance(new, list):
        return [rate(old, n) for n in new]
    if isinstance(old, list) and isinstance(new, float):
        return [rate(o, new) for o in old]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            raise OperatorException(
                f"{operator} 在数组对数组时要求长度一致（基期 {len(old)} 个，现期 {len(new)} 个）",
                code=ErrorCode.CONFIG_INVALID,
                operator=operator,
                config=config,
            )
        return [rate(o, n) for o, n in zip(old, new)]
    raise OperatorException(
        f"{operator} 内部类型组合异常",
        code=ErrorCode.CALC_LOGIC_ERROR,
        operator=operator,
        config=config,
    )


@OperatorRegistry.register("arithmetic_mean")
class ArithmeticMeanOperator(BaseOperator):
    """
    算术平均值算子：计算数据的算术平均数
    
    功能说明：
    - 计算所有数值的和除以数值个数
    - 最常见的平均值计算方法
    
    计算逻辑：
    result = (x1 + x2 + ... + xn) / n
    例: [10, 20, 30] 的算术平均值 = 60/3 = 20
    
    配置参数：
    - fields (array): 数据源字段列表
    - operands (array): 同fields，备选参数
    - field (any): 单个字段时使用
    
    输入数据格式：
    base_data: {
        "score1": 90,
        "score2": 85,
        "score3": 95
    }
    
    配置示例：
    {
        "operator": "arithmetic_mean",
        "config": {
            "fields": ["score1", "score2", "score3"]
        }
    }
    
    输出格式：
    90.0 (数值)
    
    数据支持：
    - 单值: 100
    - 数组: [90, 85, 95]
    - JSON数组字符串: "[90, 85, 95]"
    
    特殊情况：
    - 空数据返回 0.0
    - 支持多个数组，自动展开合并
    
    使用场景：
    - 学生成绩平均分
    - 销售数据平均值
    - 性能指标平均
    """
    name = "arithmetic_mean"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        return statistics.mean(values)


@OperatorRegistry.register("harmonic_mean")
class HarmonicMeanOperator(BaseOperator):
    """调和平均数：n / (1/x1 + 1/x2 + … + 1/xn)。常用于平均速率、平均价格。"""
    name = "harmonic_mean"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        if any(v == 0 for v in values):
            from ...core.exceptions import OperatorException, ErrorCode
            raise OperatorException("调和平均数的数据中不能包含 0", code=ErrorCode.CALC_LOGIC_ERROR, operator=self.name, config=config)
        return len(values) / sum(1.0 / v for v in values)


@OperatorRegistry.register("geometric_mean")
class GeometricMeanOperator(BaseOperator):
    """几何平均数：(x1*x2*…*xn)^(1/n)。常用于增长率、比例的平均。"""
    name = "geometric_mean"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        if any(v <= 0 for v in values):
            from ...core.exceptions import OperatorException, ErrorCode
            raise OperatorException("几何平均数要求所有值均大于 0", code=ErrorCode.CALC_LOGIC_ERROR, operator=self.name, config=config)
        return math.exp(sum(math.log(v) for v in values) / len(values))


@OperatorRegistry.register("weighted_average")
class WeightedAverageOperator(BaseOperator):
    """加权平均值：(w1*x1 + w2*x2 + … + wn*xn) / (w1+w2+…+wn)。"""
    name = "weighted_average"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {},
        "weights": {"type": "array"},
        "first_value": {},
        "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(w * v for w, v in zip(weights, values)) / total_w


@OperatorRegistry.register("median")
class MedianOperator(BaseOperator):
    """
    中位数算子：计算数据的中位数
    
    功能说明：
    - 将数据从小到大排序，取中间值
    - 不受极端值影响，比平均值更稳健
    
    计算逻辑：
    - 数据个数为奇数: 取中间值
    - 数据个数为偶数: 取中间两值的平均
    例: [1, 3, 5] → 3, [1, 2, 3, 4] → 2.5
    
    配置参数：
    - fields (array): 数据源字段列表
    
    输入数据格式：
    base_data: {
        "values": [95, 85, 90, 100, 88]
    }
    
    配置示例：
    {
        "operator": "median",
        "config": {
            "fields": ["values"]
        }
    }
    
    输出格式：
    90.0 (数值)
    
    vs 平均值：
    - 数据: [10, 20, 30, 40, 1000]
    - 平均值: 220.0
    - 中位数: 30.0 (更能代表数据中心)
    
    使用场景：
    - 房价中位数（避免极端房价影响）
    - 收入中位数分析
    - 数据质量评估
    """
    name = "median"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            raise OperatorException(
                "median 未收集到任何数值（请检查 fields/operands/first_value 是否能取到值；"
                "若引用了 ${step}.field，请确保该 step 在本节点之前已执行）",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        return statistics.median(values)


@OperatorRegistry.register("mode")
class ModeOperator(BaseOperator):
    name = "mode"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"},
        "fields": {"type": "array"},
        "field": {},
        "first_value": {},
        "second_value": {},
        "third_value": {},
        "fourth_value": {},
        "fifth_value": {},
        "sixth_value": {},
        "seventh_value": {},
        "eighth_value": {},
        "ninth_value": {},
        "tenth_value": {},
    }}
    default_config = {"fields": []}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            raise OperatorException(
                "mode 未收集到任何数值（请检查 fields/operands/first_value 等是否能取到值）",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        # 诊断日志：用于确认是否命中新代码路径（以及 values 是否被正确解析）
        logger.info("mode(values) count=%s sample=%s", len(values), values[:5])
        counts = Counter(values)
        max_count = counts.most_common(1)[0][1]
        modes = [v for v, c in counts.items() if c == max_count]
        out = modes[0] if len(modes) == 1 else modes
        logger.info("mode(result)=%s", out if not isinstance(out, list) else f"list(len={len(out)})")
        return out


@OperatorRegistry.register("range")
class RangeOperator(BaseOperator):
    name = "range"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        return max(values) - min(values)


@OperatorRegistry.register("std_dev")
class StdDevOperator(BaseOperator):
    name = "std_dev"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)


@OperatorRegistry.register("variance")
class VarianceOperator(BaseOperator):
    name = "variance"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if len(values) < 2:
            return 0.0
        return statistics.variance(values)


@OperatorRegistry.register("growth_rate")
class GrowthRateOperator(BaseOperator):
    name = "growth_rate"
    config_schema = {
        "type": "object",
        "properties": {
            "old_value": {},
            "new_value": {},
            "first_value": {},
            "second_value": {},
        },
    }
    default_config = {}
    input_spec = {"type": "table"}
    # 标量或等长数值列表（数值-数组 / 数组-数值 / 数组-数组）
    output_spec = None

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("old_value") in (None, "") and c.get("first_value") not in (None, ""):
            c["old_value"] = c.get("first_value")
        if c.get("new_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["new_value"] = c.get("second_value")
        return c

    def execute(self, data, config, context: ExecutionContext):
        raw_old = get_value(data, config.get("old_value"), context)
        raw_new = get_value(data, config.get("new_value"), context)
        old_v = _scalar_or_number_vector(
            raw_old, label="基期(first_value/old_value)", operator=self.name, config=config
        )
        new_v = _scalar_or_number_vector(
            raw_new, label="现期(second_value/new_value)", operator=self.name, config=config
        )
        return _growth_rate_pairwise(old_v, new_v, operator=self.name, config=config)


@OperatorRegistry.register("decline_rate")
class DeclineRateOperator(BaseOperator):
    name = "decline_rate"
    config_schema = {
        "type": "object",
        "properties": {
            "old_value": {},
            "new_value": {},
            "first_value": {},
            "second_value": {},
        },
    }
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = None

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("old_value") in (None, "") and c.get("first_value") not in (None, ""):
            c["old_value"] = c.get("first_value")
        if c.get("new_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["new_value"] = c.get("second_value")
        return c

    def execute(self, data, config, context: ExecutionContext):
        raw_old = get_value(data, config.get("old_value"), context)
        raw_new = get_value(data, config.get("new_value"), context)
        old_v = _scalar_or_number_vector(
            raw_old, label="基期(first_value/old_value)", operator=self.name, config=config
        )
        new_v = _scalar_or_number_vector(
            raw_new, label="现期(second_value/new_value)", operator=self.name, config=config
        )
        return _decline_rate_pairwise(old_v, new_v, operator=self.name, config=config)


@OperatorRegistry.register("percentage")
class PercentageOperator(BaseOperator):
    """百分比计算：part / total * 100（列表按元素求和，见 `to_number`）。"""
    name = "percentage"
    config_schema = {"type": "object", "properties": {"part": {}, "total": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def execute(self, data, config, context: ExecutionContext):
        part_val = to_number(get_value(data, config.get("part"), context))
        total_val = to_number(get_value(data, config.get("total"), context))
        if total_val == 0:
            raise OperatorException(
                "百分比算子 total(总体) 不能为 0",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return (part_val / total_val) * 100


@OperatorRegistry.register("percentage_by_count")
class PercentageByCountOperator(BaseOperator):
    """条数占比：len(part) / len(total) * 100（非列表视为 1 条，None 为 0 条）。"""
    name = "percentage_by_count"
    config_schema = {"type": "object", "properties": {"part": {}, "total": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def execute(self, data, config, context: ExecutionContext):
        part_val = _list_element_count(get_value(data, config.get("part"), context))
        total_val = _list_element_count(get_value(data, config.get("total"), context))
        if total_val == 0:
            raise OperatorException(
                "percentage_by_count 的 total 条数不能为 0",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return (part_val / total_val) * 100




@OperatorRegistry.register("weighted_sum_squares")
class WeightedSumSquaresOperator(BaseOperator):
    name = "weighted_sum_squares"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "weights": {"type": "array"},
        "first_value": {}, "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        return sum(w * (v ** 2) for w, v in zip(weights, values))


@OperatorRegistry.register("sum_squares")
class SumSquaresOperator(BaseOperator):
    name = "sum_squares"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        return sum(x ** 2 for x in values)




@OperatorRegistry.register("percentile")
class PercentileOperator(BaseOperator):
    name = "percentile"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"},
        "fields": {"type": "array"},
        "field": {},
        "percentile": {},
        # 兼容顺序参数：first_value/second_value
        "first_value": {},
        "second_value": {},
    }}
    # 不在 default_config 里写 percentile：否则合并后始终为 50，second_value 无法写入 percentile
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        # second_value 表示分位点（0~100），须在 normalize_config_to_fields 之前写入 percentile，
        # 否则旧逻辑会把 first_value+second_value 都塞进 fields，样本里混入分位点数值。
        if merged.get("percentile") in (None, "") and merged.get("second_value") not in (None, ""):
            merged["percentile"] = merged.get("second_value")
        merged = normalize_config_to_fields(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        # 优先 first_value：避免与非严格算子一样把 second_value 当「第二列数据」收集
        if config.get("first_value") not in (None, ""):
            raw = get_value(data, config.get("first_value"), context)
            values = _ensure_number_list(raw)
        else:
            values = _collect_values(data, config, context)
        if not values:
            raise OperatorException(
                "percentile 未收集到任何数值（请检查 fields/operands/first_value 是否能取到值）",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        p_raw = config.get("percentile", 50)
        p = safe_convert_to_number(p_raw)
        if p is None:
            raise OperatorException(
                "percentile 的分位点必须为数值（0~100）",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        p = float(p)
        if p < 0 or p > 100:
            raise OperatorException(
                f"percentile 分位点超出范围（0~100）: {p}",
                code=ErrorCode.CONFIG_INVALID,
                operator=self.name,
                config=config,
            )
        if _HAS_NUMPY:
            return float(np.percentile(values, p))
        values_sorted = sorted(values)
        k = (len(values_sorted) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(values_sorted) else f
        return values_sorted[f] + (k - f) * (values_sorted[c] - values_sorted[f])


@OperatorRegistry.register("cv")
class CvOperator(BaseOperator):
    name = "cv"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if len(values) < 2:
            return 0.0
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.0
        return statistics.stdev(values) / mean_val * 100




@OperatorRegistry.register("quartiles")
class QuartilesOperator(BaseOperator):
    name = "quartiles"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "list"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return [0.0, 0.0, 0.0]
        if _HAS_NUMPY:
            q1, q2, q3 = np.percentile(values, [25, 50, 75])
            return [float(q1), float(q2), float(q3)]
        s = sorted(values)
        n = len(s)

        def _interp(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            return s[f] + (k - f) * (s[c] - s[f])

        q1 = _interp(0.25)
        q2 = statistics.median(s)
        q3 = _interp(0.75)
        return [q1, q2, q3]


@OperatorRegistry.register("iqr")
class IqrOperator(BaseOperator):
    name = "iqr"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)

        def _interp(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            return s[f] + (k - f) * (s[c] - s[f])

        if _HAS_NUMPY:
            q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
        else:
            q1, q3 = _interp(0.25), _interp(0.75)
        return q3 - q1


@OperatorRegistry.register("weighted_variance")
class WeightedVarianceOperator(BaseOperator):
    name = "weighted_variance"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "weights": {"type": "array"},
        "first_value": {}, "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        w_mean = sum(w * v for w, v in zip(weights, values)) / total_w
        return sum(w * (v - w_mean) ** 2 for w, v in zip(weights, values)) / total_w


@OperatorRegistry.register("weighted_std")
class WeightedStdOperator(BaseOperator):
    """加权标准差::√[Σ(wi*(vi - μ)²) / Σwi]，其中μ=Σ(wi*vi)/Σwi。"""
    name = "weighted_std"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "weights": {"type": "array"},
        "first_value": {}, "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        w_mean = sum(w * v for w, v in zip(weights, values)) / total_w
        wvar = sum(w * (v - w_mean) ** 2 for w, v in zip(weights, values)) / total_w
        return math.sqrt(wvar)


@OperatorRegistry.register("trimmed_mean")
class TrimmedMeanOperator(BaseOperator):
    """截尾均值：去掉极端值后计算平均值"""
    name = "trimmed_mean"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "trim_percent": {"type": "number"}}}
    default_config = {"fields": [], "trim_percent": 0.1}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        trim_percent = float(config.get("trim_percent", 0.1))
        trim_percent = max(0, min(trim_percent, 0.5))
        n = len(values)
        trim_count = int(n * trim_percent)
        sorted_vals = sorted(values)
        trimmed = sorted_vals[trim_count : n - trim_count] if trim_count > 0 else sorted_vals
        return statistics.mean(trimmed) if trimmed else 0.0


@OperatorRegistry.register("winsorized_mean")
class WinsorizedMeanOperator(BaseOperator):
    """缩尾均值：极端值替换为临界值后计算平均值"""
    name = "winsorized_mean"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "winsor_percent": {"type": "number"}}}
    default_config = {"fields": [], "winsor_percent": 0.1}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        winsor_percent = float(config.get("winsor_percent", 0.1))
        winsor_percent = max(0, min(winsor_percent, 0.5))
        n = len(values)
        winsor_count = int(n * winsor_percent)
        sorted_vals = sorted(values)
        if winsor_count > 0:
            lower_bound = sorted_vals[winsor_count]
            upper_bound = sorted_vals[n - winsor_count - 1]
            winsorized = [max(lower_bound, min(upper_bound, v)) for v in values]
        else:
            winsorized = values
        return statistics.mean(winsorized)


@OperatorRegistry.register("frequency")
class FrequencyOperator(BaseOperator):
    """频数：统计值在列表中出现的次数"""
    name = "frequency"
    config_schema = {"type": "object", "properties": {"field": {}, "input": {}, "target_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("target_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["target_value"] = c.get("second_value")
        return c

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("field") or config.get("input"), context)
        target = get_value(data, config.get("target_value"), context)
        if not isinstance(val, list):
            val = [val] if val is not None else []
        return val.count(target)


@OperatorRegistry.register("relative_frequency")
class RelativeFrequencyOperator(BaseOperator):
    """频率：值出现的次数 / 总数"""
    name = "relative_frequency"
    config_schema = {"type": "object", "properties": {"field": {}, "input": {}, "target_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("target_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["target_value"] = c.get("second_value")
        return c

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("field") or config.get("input"), context)
        target = get_value(data, config.get("target_value"), context)
        if not isinstance(val, list):
            val = [val] if val is not None else []
        if not val:
            return 0.0
        count = val.count(target)
        return count / len(val)


@OperatorRegistry.register("mean_square")
class MeanSquareOperator(BaseOperator):
    """均方：Σ(xi²) / n"""
    name = "mean_square"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        return sum(v ** 2 for v in values) / len(values)


@OperatorRegistry.register("root_mean_square")
class RootMeanSquareOperator(BaseOperator):
    """均方根：sqrt(Σ(xi²) / n)"""
    name = "root_mean_square"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        ms = sum(v ** 2 for v in values) / len(values)
        return math.sqrt(ms)


@OperatorRegistry.register("log_sum_square")
class LogSumSquareOperator(BaseOperator):
    """对数平方和：Σ(ln(xi)²)"""
    name = "log_sum_square"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values or any(v <= 0 for v in values):
            raise OperatorException(
                "log_sum_square 要求所有值为正数",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return sum(math.log(v) ** 2 for v in values)


@OperatorRegistry.register("weighted_log_sum_square")
class WeightedLogSumSquareOperator(BaseOperator):
    """加权对数平方和：Σ(w * ln(xi)²)"""
    name = "weighted_log_sum_square"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "weights": {"type": "array"},
        "first_value": {}, "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values or any(v <= 0 for v in values):
            raise OperatorException(
                "weighted_log_sum_square 要求所有值为正数",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        return sum(w * math.log(v) ** 2 for w, v in zip(weights, values))


@OperatorRegistry.register("exp_sum_square")
class ExpSumSquareOperator(BaseOperator):
    """指数平方和：Σ(e^(xi)²)"""
    name = "exp_sum_square"
    config_schema = {"type": "object", "properties": {"operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}}}
    default_config = {"fields": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        return sum(math.exp(v) ** 2 for v in values)


@OperatorRegistry.register("weighted_exp_sum_square")
class WeightedExpSumSquareOperator(BaseOperator):
    """加权指数平方和：Σ(w * e^(xi)²)"""
    name = "weighted_exp_sum_square"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "weights": {"type": "array"},
        "first_value": {}, "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        return sum(w * math.exp(v) ** 2 for w, v in zip(weights, values))


@OperatorRegistry.register("weighted_log_mean")
class WeightedLogMeanOperator(BaseOperator):
    """加权对数均值：Σ(w * ln(xi)) / Σ(w)"""
    name = "weighted_log_mean"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "weights": {"type": "array"},
        "first_value": {}, "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values or any(v <= 0 for v in values):
            raise OperatorException(
                "weighted_log_mean 要求所有值为正数",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(w * math.log(v) for w, v in zip(weights, values)) / total_w


@OperatorRegistry.register("weighted_exp_mean")
class WeightedExpMeanOperator(BaseOperator):
    """加权指数均值：Σ(w * e^(xi)) / Σ(w)"""
    name = "weighted_exp_mean"
    config_schema = {"type": "object", "properties": {
        "operands": {"type": "array"}, "fields": {"type": "array"}, "field": {}, "weights": {"type": "array"},
        "first_value": {}, "second_value": {},
    }}
    default_config = {"fields": [], "weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return normalize_config_to_fields(merged)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(w * math.exp(v) for w, v in zip(weights, values)) / total_w


@OperatorRegistry.register("count_items")
class CountItemsOperator(BaseOperator):
    """
    总数算子：计算列表总数（长度）。
    config: field（必填，列表数据来源）。
    """
    name = "count_items"
    config_schema = {"type": "object", "properties": {"field": {}, "input": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("field") or config.get("input"), context)
        if val is None:
            return 0
        if not isinstance(val, list):
            raise OperatorException(
                "count_items 要求数据来源为列表",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        return len(val)


OperatorRegistry._operators["average"] = OperatorRegistry._operators["arithmetic_mean"]
