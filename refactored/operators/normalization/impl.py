"""
归一化/标准化算子：
  normalize        — L2 归一化（向量各分量除以 L2 范数）
  standardize      — Z-score 标准化（均值=0，标准差=1）
  z_score          — Z 分数（standardize 的别名）
  min_max_normalize — Min-Max 归一化（映射到 [0,1]）
"""
import math
import statistics
from typing import Any

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...utils import safe_convert_to_number
from .._common import get_value, to_number
from ..statistics.stats import _collect_values


@OperatorRegistry.register("normalize")
class NormalizeOperator(BaseOperator):
    """
    L2 归一化：将向量各分量除以向量的 L2 范数（欧几里得长度），
    使结果向量的长度为 1。
    输入：数值列表；输出：归一化后的数值列表。
    """
    name = "normalize"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return []
        norm = math.sqrt(sum(x ** 2 for x in values))
        if norm == 0:
            return [0.0] * len(values)
        return [x / norm for x in values]


@OperatorRegistry.register("standardize")
class StandardizeOperator(BaseOperator):
    """
    Z-score 标准化（零均值单位方差）：(x - mean) / std。
    输入：数值列表；输出：z 分数列表（均值≈0，标准差≈1）。
    """
    name = "standardize"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if len(values) < 2:
            return values
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        if std_val == 0:
            return [0.0] * len(values)
        return [(x - mean_val) / std_val for x in values]



@OperatorRegistry.register("z_score")
class ZScoreOperator(BaseOperator):
    """
    Z 分数：计算单个值相对于均值和标准差的位置，
    等同于 standardize，返回同样的 z 分数列表。
    """
    name = "z_score"
    config_schema = {"type": "object", "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "fourth_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data, config, context: ExecutionContext):
        return StandardizeOperator().execute(data, config, context)


@OperatorRegistry.register("min_max_normalize")
class MinMaxNormalizeOperator(BaseOperator):
    """
    Min-Max 归一化：将数值线性映射到 [0, 1]（或指定的 [min_val, max_val]）。
    公式：(x - min) / (max - min)。
    可选 min_val / max_val 指定固定上下界，否则从数据中自动取。
    """
    name = "min_max_normalize"
    config_schema = {"type": "object", "properties": {
        "min_val": {}, "max_val": {},
        "first_value": {}, "second_value": {}, "third_value": {},
    }}
    default_config = {}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        if merged.get("min_val") in (None, "") and merged.get("second_value") not in (None, ""):
            merged = dict(merged)
            merged["min_val"] = merged.get("second_value")
        if merged.get("max_val") in (None, "") and merged.get("third_value") not in (None, ""):
            merged = dict(merged)
            merged["max_val"] = merged.get("third_value")
        return merged

    def execute(self, data, config, context: ExecutionContext):
        values = _collect_values(data, config, context)
        if not values:
            return []
        min_raw = config.get("min_val")
        max_raw = config.get("max_val")

        def _bound_number(raw: Any) -> float:
            n = safe_convert_to_number(raw)
            if n is not None:
                return float(n)
            return to_number(get_value(data, raw, context))

        if min_raw is not None and min_raw != "":
            min_val = _bound_number(min_raw)
        else:
            min_val = min(values)
        if max_raw is not None and max_raw != "":
            max_val = _bound_number(max_raw)
        else:
            max_val = max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(x - min_val) / (max_val - min_val) for x in values]
