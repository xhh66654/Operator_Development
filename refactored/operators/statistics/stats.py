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
from .._common import get_value, _ctx, to_number

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
    从顺序参数 first_value/second_value/... 收集数值（数据来源：接口字段名或 ${step_key}）。
    说明：仅从顺序槽位取数，不做旧键兼容读取。
    """
    values: List[float] = []
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
    fields = [config.get(k) for k in seq_keys if config.get(k) not in (None, "")]

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

    约定：
    - 凡调用本函数的算子，execute 中必须用 _collect_values_first_only 取观测，
      不得用 _collect_values（否则 second_value 解析出的权重会被拼进样本）。
    - 权重来自 weights 配置，或在与 first_value 同时配置 second_value 时由 second_value 覆盖写入 weights。
    """
    fv = merged.get("first_value")
    sv = merged.get("second_value")
    if fv in (None, "") or sv in (None, ""):
        return merged
    out = dict(merged)
    # second_value 为显式顺序槽位输入，应覆盖默认/具名 weights
    out["weights"] = sv
    out["second_value"] = None
    return out


def _number_list_from_first_value_only(
    data: Dict, config: Dict, context: ExecutionContext, *, operator: str
) -> List[float]:
    """
    仅从 first_value 解析数值列表（可空列表），禁止多顺序槽位合并。
    用于加权类、离散类等：避免 second_value（权重/目标等）被 _collect_values 拼进样本。
    """
    fv = config.get("first_value")
    if fv in (None, ""):
        raise OperatorException(
            f"{operator} 必须配置 first_value 作为样本来源，不支持多顺序槽位合并样本",
            code=ErrorCode.CONFIG_MISSING,
            operator=operator,
            config=config,
        )
    raw = get_value(data, fv, context)
    if raw is None:
        raise OperatorException(
            f"{operator} 未取到样本（first_value 引用缺失或为空）",
            code=ErrorCode.DATA_NOT_FOUND,
            operator=operator,
            config=config,
        )
    return _ensure_number_list(raw)


def _collect_values_first_only(data: Dict, config: Dict, context: ExecutionContext, *, operator: str) -> List[float]:
    """加权类算子：同 `_number_list_from_first_value_only`。"""
    return _number_list_from_first_value_only(data, config, context, operator=operator)


def _ensure_sample_from_first_value_only(
    data: Dict, config: Dict, context: ExecutionContext, *, operator: str
) -> List[float]:
    """分位数等：first_value 样本且至少含一个数值。"""
    values = _number_list_from_first_value_only(data, config, context, operator=operator)
    if not values:
        raise OperatorException(
            f"{operator} 样本解析后无任何数值",
            code=ErrorCode.DATA_NOT_FOUND,
            operator=operator,
            config=config,
        )
    return values


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
    """算术平均数 (arithmetic_mean)
    计算逻辑：所有样本数值之和除以样本数量（最常用的平均值）
    数学公式：μ = Σxi / n
    数据来源：first_value（样本集合）
    特殊处理：无数据时返回 0.0
    """
    name = "arithmetic_mean"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        return statistics.mean(values)


@OperatorRegistry.register("harmonic_mean")
class HarmonicMeanOperator(BaseOperator):
    """调和平均数 (harmonic_mean)
    计算逻辑：样本数量的倒数除以各数值倒数之和（常用于速率、密度平均）
    数学公式：H = n / Σ(1/xi)
    数据来源：first_value（样本集合）
    异常处理：样本中包含 0 时抛出 CALC_LOGIC_ERROR
    """
    name = "harmonic_mean"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
    """几何平均数 (geometric_mean)
        计算逻辑：n个数值乘积的n次方根（常用于增长率、指数平均）
        数学公式：G = (Πxi)^(1/n) = exp(Σln(xi) / n)
        数据来源：first_value（样本集合）
        异常处理：样本中包含非正数（<=0）时抛出 CALC_LOGIC_ERROR
        """
    name = "geometric_mean"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
    """加权平均值 (weighted_average)
    计算逻辑：各数值乘以对应权重后的总和除以权重总和
    数学公式：Sum(wi * xi) / Sum(wi)
    数据来源：first_value（样本集合）；权重来源：weights（或兼容 second_value）
    异常处理：未获取到观测值时抛出 DATA_NOT_FOUND；权重和为 0 时返回 0.0
    """
    name = "weighted_average"
    config_schema = {"type": "object", "properties": {
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
        if not values:
            raise OperatorException(
                "weighted_average 未取到任何观测值（顺序槽位解析后为空或引用缺失）",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(w * v for w, v in zip(weights, values)) / total_w


@OperatorRegistry.register("median")
class MedianOperator(BaseOperator):
    """中位数 (median)
    计算逻辑：将样本排序后位于中间位置的数值（50%分位点）
    数学公式：statistics.median(values)
    数据来源：first_value（样本集合）
    异常处理：未收集到任何数值时抛出 DATA_NOT_FOUND
    """
    name = "median"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            raise OperatorException(
                "median 未收集到任何数值（请检查 first_value/second_value/... 是否能取到值；"
                "若引用了 ${step}.field，请确保该 step 在本节点之前已执行）",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        return statistics.median(values)


@OperatorRegistry.register("mode")
class ModeOperator(BaseOperator):
    """众数
    计算逻辑：统计样本中出现频率最高的数值
    返回规则：若唯一众数则返回该值；若存在多个并列众数则返回列表
    数据来源：first_value 至 tenth_value（支持多字段合并统计）
    异常处理：未收集到任何数值时抛出 DATA_NOT_FOUND
    """
    name = "mode"
    config_schema = {"type": "object", "properties": {
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
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            raise OperatorException(
                "mode 未收集到任何数值（请检查 first_value/second_value/... 是否能取到值）",
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
    """极差
    计算逻辑：样本中的最大值减去最小值
    数学公式：max(values) - min(values)
    数据来源：first_value（样本集合）
    特殊处理：无数据时返回 0.0
    """
    name = "range"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        return max(values) - min(values)


@OperatorRegistry.register("std_dev")
class StdDevOperator(BaseOperator):
    """标准差
    计算逻辑：衡量样本数值的离散程度（基于样本标准差，使用贝塞尔校正）
    数学公式：sqrt(sum((x - mean)^2) / (n - 1))
    数据来源：first_value（样本集合）
    特殊处理：样本数 < 2 时返回 None（无法计算无偏估计）
    """
    name = "std_dev"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)


@OperatorRegistry.register("variance")
class VarianceOperator(BaseOperator):
    """方差
    计算逻辑：衡量样本数值的离散程度（基于样本方差，使用贝塞尔校正）
    数学公式：s² = Σ(xi - x̄)² / (n - 1)
    数据来源：first_value（样本集合）
    特殊处理：样本数 < 2 时返回 None（无法计算无偏估计）
    """
    name = "variance"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if len(values) < 2:
            return 0.0
        return statistics.variance(values)


@OperatorRegistry.register("growth_rate")
class GrowthRateOperator(BaseOperator):
    """增长率
    计算逻辑：(现期值 - 基期值) / 基期值，支持标量或向量批量计算
    参数兼容：old_value 兼容 first_value（基期）；new_value 兼容 second_value（现期）
    数据来源：old_value/first_value（基期）；new_value/second_value（现期）
    """
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
        c = dict(super()._resolve_config(config))
        if c.get("old_value") in (None, "") and c.get("first_value") not in (None, ""):
            c["old_value"] = c.get("first_value")
            c["first_value"] = None
        if c.get("new_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["new_value"] = c.get("second_value")
            c["second_value"] = None
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
    """下降率
    计算逻辑：(基期值 - 现期值) / 基期值，支持标量或向量批量计算
    参数兼容：old_value 兼容 first_value（基期）；new_value 兼容 second_value（现期）
    数据来源：old_value/first_value（基期）；new_value/second_value（现期）
    """
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
        c = dict(super()._resolve_config(config))
        if c.get("old_value") in (None, "") and c.get("first_value") not in (None, ""):
            c["old_value"] = c.get("first_value")
            c["first_value"] = None
        if c.get("new_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["new_value"] = c.get("second_value")
            c["second_value"] = None
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
    """百分比
        计算逻辑：部分数值除以总体数值，结果以百分比形式返回
        数学公式：(part / total) * 100
        数值处理：若输入为列表，则先求和再计算（基于 to_number 逻辑）
        数据来源：first_value（部分）；second_value（总体）
        异常处理：分母（总体数值）为 0 时抛出 CALC_LOGIC_ERROR
        """
    name = "percentage"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value", "second_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        part_val = to_number(get_value(data, config.get("first_value"), context))
        total_val = to_number(get_value(data, config.get("second_value"), context))
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
    """条数占比
    计算逻辑：部分条数除以总条数，结果以百分比形式返回
    数学公式：(count(part) / count(total)) * 100
    计数规则：非列表视为 1 条，None 视为 0 条
    数据来源：first_value（部分）；second_value（整体）
    异常处理：分母（整体条数）为 0 时抛出 CALC_LOGIC_ERROR
    """
    name = "percentage_by_count"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value", "second_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def execute(self, data, config, context: ExecutionContext):
        part_val = _list_element_count(get_value(data, config.get("first_value"), context))
        total_val = _list_element_count(get_value(data, config.get("second_value"), context))
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
    """加权平方和
    计算逻辑：(样本1² × 权重1) + (样本2² × 权重2) + ...
    数学公式：Sum(weights[i] * values[i]²)
    权重顺序：按位置一一对应（第1个权重对应第1个样本，第2个对应第2个...）
    数据来源：first_value（样本集合）；权重来源：weights（或兼容 second_value）
    """
    name = "weighted_sum_squares"
    config_schema = {"type": "object", "properties": {
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
        if not values:
            return 0.0
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        return sum(w * (v ** 2) for w, v in zip(weights, values))


@OperatorRegistry.register("sum_squares")
class SumSquaresOperator(BaseOperator):
    """平方和
    计算逻辑：遍历样本中的每个数值进行平方运算，最后求和
    数学公式：Sum(x^2)
    数据来源：first_value（样本集合）
    """
    name = "sum_squares"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        return sum(x ** 2 for x in values)




@OperatorRegistry.register("percentile")
class PercentileOperator(BaseOperator):
    """百分位数
    数据来源：first_value（样本集合）；分位点参数：second_value（兼容旧参数，映射为 percentile）
    计算逻辑：计算样本在指定分位点（0~100）上的数值
    手动模式：线性插值法（基于索引 k = (n-1) * p / 100）
    异常处理：分位点必须为数值且在 0~100 范围内
    """
    name = "percentile"
    config_schema = {
        "type": "object",
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}, "percentile": {}},
    }
    # 不在 default_config 里写 percentile：否则合并后始终为 50，second_value 无法写入 percentile
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        if merged.get("percentile") in (None, "") and merged.get("second_value") not in (None, ""):
            merged = dict(merged)
            merged["percentile"] = merged.get("second_value")
            merged["second_value"] = None
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _ensure_sample_from_first_value_only(data, config, context, operator=self.name)
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
    """变异系数
    计算逻辑：标准差除以平均值，结果以百分比形式返回
    标准差：statistics.stdev(values)（基于样本的标准差）
    平均值：statistics.mean(values)（算术平均值）
    特殊处理：样本数 < 2 或 平均值 == 0 时返回 0.0（避免除零或计算错误）
    """
    name = "cv"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
    """四分位数
    计算逻辑：返回包含第一四分位数、中位数、第三四分位数的列表 [Q1, Q2, Q3]
    Q1：np.percentile(values, 25) 或 _interp(0.25)（第 25 百分位数）
    Q2：statistics.median(s)（第 50 百分位数/中位数，NumPy 模式下为 percentile 50）
    Q3：np.percentile(values, 75) 或 _interp(0.75)（第 75 百分位数）
    """
    name = "quartiles"
    config_schema = {
        "type": "object",
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}},
    }
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "list"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _ensure_sample_from_first_value_only(data, config, context, operator=self.name)
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
    """四分位距
    计算逻辑：第三四分位数（Q3）减去第一四分位数（Q1）
    Q1：np.percentile(values, 25) （第 25 百分位数）
    Q3：np.percentile(values, 75) （第 75 百分位数）
    """
    name = "iqr"
    config_schema = {
        "type": "object",
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}},
    }
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _ensure_sample_from_first_value_only(data, config, context, operator=self.name)
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
    """加权方差 (weighted_variance)
    计算逻辑：各数值与加权均值差的平方，乘以权重后求和，再除以权重总和
    数学公式：σ² = Σ(wi * (xi - μ)²) / Σwi
    数据来源：first_value（样本集合）；权重来源：weights（或兼容 second_value）
    特殊处理：无数据或权重和为 0 时返回 0.0
    """
    name = "weighted_variance"
    config_schema = {"type": "object", "properties": {
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
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
    """加权标准差 (weighted_std)
    计算逻辑：加权方差的算术平方根
    数学公式：σ = √[Σ(wi * (xi - μ)²) / Σwi]
    数据来源：first_value（样本集合）；权重来源：weights（或兼容 second_value）
    特殊处理：无数据或权重和为 0 时返回 0.0
    """
    name = "weighted_std"
    config_schema = {"type": "object", "properties": {
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
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
    """截尾均值 (trimmed_mean)
    计算逻辑：排序后去除首尾指定比例（默认 10%）的极端值，再计算算术平均
    数学公式：μ = Σ(sorted[k:-k]) / (n-2k)
    数据来源：first_value（样本集合）
    特殊处理：无数据时返回 0.0
    """
    name = "trimmed_mean"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "trim_percent": {"type": "number"}}}
    default_config = {"trim_percent": 0.1}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
    """缩尾均值 (winsorized_mean)
    计算逻辑：将首尾指定比例（默认 10%）的极端值替换为临界值，再计算算术平均
    数据来源：first_value（样本集合）
    特殊处理：无数据时返回 0.0
    """
    name = "winsorized_mean"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "winsor_percent": {"type": "number"}}}
    default_config = {"winsor_percent": 0.1}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
    """频数 (frequency)
    计算逻辑：统计目标值在列表中出现的次数
    数学公式：Count(x == target)
    数据来源：first_value（列表）；target_value（目标值）
    参数兼容：target_value 兼容 second_value
    """
    name = "frequency"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "target_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        c = dict(super()._resolve_config(config))
        if c.get("target_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["target_value"] = c.get("second_value")
            c["second_value"] = None
        return c

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
        target = get_value(data, config.get("target_value"), context)
        if not isinstance(val, list):
            val = [val] if val is not None else []
        return val.count(target)


@OperatorRegistry.register("relative_frequency")
class RelativeFrequencyOperator(BaseOperator):
    """频率 (relative_frequency)
    计算逻辑：目标值出现次数除以列表总长度
    数学公式：P = Count(x == target) / N
    参数兼容：target_value 兼容 second_value
    数据来源：first_value（列表）；target_value（目标值）
    特殊处理：列表为空时返回 0.0
    """
    name = "relative_frequency"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "target_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        c = dict(super()._resolve_config(config))
        if c.get("target_value") in (None, "") and c.get("second_value") not in (None, ""):
            c["target_value"] = c.get("second_value")
            c["second_value"] = None
        return c

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
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
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return 0.0
        return sum(v ** 2 for v in values) / len(values)


@OperatorRegistry.register("root_mean_square")
class RootMeanSquareOperator(BaseOperator):
    """均方根：sqrt(Σ(xi²) / n)"""
    name = "root_mean_square"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
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
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

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
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
        if not values:
            return 0.0
        weights = _weights_flat(data, config.get("weights"), context, len(values), operator=self.name, config=config)
        return sum(w * math.exp(v) ** 2 for w, v in zip(weights, values))


@OperatorRegistry.register("weighted_log_mean")
class WeightedLogMeanOperator(BaseOperator):
    """加权对数均值：Σ(w * ln(xi)) / Σ(w)"""
    name = "weighted_log_mean"
    config_schema = {"type": "object", "properties": {
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
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
        "weights": {"type": "array"},
        "first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}, "fifth_value": {},
    }}
    default_config = {"weights": []}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        merged = _resolve_first_second_values_weights(merged)
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values_first_only(data, config, context, operator=self.name)
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
    输入：first_value（必填，列表数据来源）。
    """
    name = "count_items"
    config_schema = {"type": "object", "properties": {"first_value": {}}}
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "scalar"}

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
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
