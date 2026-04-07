"""精度控制算子：中间高精度，结果统一约分到指定小数位。数据来源为 source/field；decimal_places 为前端配置。"""
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...utils import safe_convert_to_number
from .._common import get_value


FALLBACK_DECIMAL_PLACES = 10


def get_precision(context: ExecutionContext) -> int:
    val = context.get(context.PRECISION_KEY)
    if val is not None:
        return int(val)
    return FALLBACK_DECIMAL_PLACES


def round_to_precision(value, decimal_places: int):
    """按指定小数位四舍五入（ROUND_HALF_UP，符合常见「四舍五入」，非 Python 内置 round 的银行家舍入）。"""
    if value is None or (not isinstance(value, (int, float))):
        return value
    dp = int(decimal_places)
    if dp < 0:
        dp = 0
    try:
        d = Decimal(str(float(value)))
        q = Decimal(10) ** -dp
        return float(d.quantize(q, rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, TypeError):
        return float(value)


@OperatorRegistry.register("precision_round")
class PrecisionRoundOperator(BaseOperator):
    """精度控制算子：按配置小数位四舍五入（ROUND_HALF_UP）。"""
    name = "precision_round"
    config_schema = {
        "type": "object",
        "properties": {
            "decimal_places": {"type": "integer"},
            "source": {},
            "field": {},
            "expression": {},
            "first_value": {},
            "second_value": {},
            "third_value": {},
        },
    }
    default_config = {}

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("field") in (None, "") and c.get("source") in (None, "") and c.get("expression") in (None, ""):
            for k in ("first_value", "value", "input", "primary"):
                v = c.get(k)
                if v not in (None, ""):
                    c["field"] = v
                    break
        if c.get("decimal_places") in (None, ""):
            for k in ("second_value",):
                v = c.get(k)
                if v not in (None, ""):
                    c["decimal_places"] = v
                    break
        if c.get("expression") in (None, "") and c.get("third_value") not in (None, ""):
            c["expression"] = c.get("third_value")
        return c

    def execute(self, data, config, context: ExecutionContext):
        dp = config.get("decimal_places")
        decimal_places = dp if dp is not None else get_precision(context)
        field_or_expr = config.get("source") or config.get("field") or config.get("expression")
        if field_or_expr is None:
            return round_to_precision(0, decimal_places)
        value = get_value(data, field_or_expr, context)
        if isinstance(value, list):
            return [round_to_precision(safe_convert_to_number(v), decimal_places) for v in value]
        num = safe_convert_to_number(value)
        return round_to_precision(num if num is not None else 0, decimal_places)
