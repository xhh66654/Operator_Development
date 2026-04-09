"""精度控制算子：中间高精度，结果统一约分到指定小数位。数据来源为 first_value；decimal_places 为 second_value。"""
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
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
            "expression": {},
            "first_value": {},
            "second_value": {},
            "third_value": {},
        },
    }
    default_config = {}

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
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
        field_or_expr = config.get("expression")
        if field_or_expr in (None, ""):
            field_or_expr = config.get("first_value")
        if field_or_expr is None:
            raise OperatorException(
                "precision_round 缺少数据来源（first_value 或 expression）",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        value = get_value(data, field_or_expr, context)
        if isinstance(value, list):
            return [round_to_precision(safe_convert_to_number(v), decimal_places) for v in value]
        num = safe_convert_to_number(value)
        if num is None:
            raise OperatorException(
                f"precision_round 数据来源无法转为数字: {field_or_expr}",
                code=ErrorCode.DATA_NOT_FOUND,
                operator=self.name,
                config=config,
            )
        return round_to_precision(num, decimal_places)
