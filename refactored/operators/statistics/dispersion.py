"""统计类：离散程度、离均差平方和（与 stats 同包，依赖 _collect_values）。"""
import statistics
from typing import Any, Dict

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from .._common import normalize_config_to_fields
from .stats import _collect_values, _resolve_first_second_values_weights, _weights_flat


@OperatorRegistry.register("dispersion")
class DispersionOperator(BaseOperator):
    """离散程度：标准差与均值之比（变异系数同类）。"""
    name = "dispersion"
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
        std_val = statistics.stdev(values)
        return std_val / abs(mean_val)


@OperatorRegistry.register("sum_of_squared_deviations")
class SumOfSquaredDeviationsOperator(BaseOperator):
    """离均差平方和：Σ(xi - x̄)²"""
    name = "sum_of_squared_deviations"
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
        mean_val = statistics.mean(values)
        return sum((v - mean_val) ** 2 for v in values)


@OperatorRegistry.register("weighted_sum_of_squared_deviations")
class WeightedSumOfSquaredDeviationsOperator(BaseOperator):
    """加权离均差平方和：Σ(w * (xi - x̄)²)"""
    name = "weighted_sum_of_squared_deviations"
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
        return sum(w * (v - w_mean) ** 2 for w, v in zip(weights, values))
