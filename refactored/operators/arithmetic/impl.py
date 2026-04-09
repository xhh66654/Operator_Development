"""
算术运算算子：
  基础运算   add / subtract / multiply / divide
  高级运算   power / absolute_value
  三角函数   sin / cos / tan
  对数开方   log / sqrt / factorial
  向量运算   cosine_similarity / matrix_multiply
  比率计算   ratio / proportion
"""
import json
import math
from typing import Any, Dict, List, Optional, Union

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from ...utils import safe_convert_to_number
from .._common import (
    get_value,
    to_number,
    to_number_or_none,
    normalize_primary_secondary,
)


def _arithmetic_operand_or_raise(val, role: str, operator: str, config: dict):
    """算术操作数必须为可转数值；若为 None、JSON 数组字符串或不可转数值则抛错，不补 0。"""
    if val is None:
        raise OperatorException(
            f"{role}不能为空或未找到",
            code=ErrorCode.DATA_NOT_FOUND,
            operator=operator,
            config=config,
        )
    if isinstance(val, str) and val.strip().startswith("["):
        raise OperatorException(
            f"{role}不能为 JSON 数组字符串（如 \"[1]\"），请传入数值或数值数组（如 1 或 [1,2]）",
            code=ErrorCode.TYPE_ERROR,
            operator=operator,
            config=config,
        )
    n = to_number_or_none(val)
    if n is None:
        raise OperatorException(
            f"{role}无法转为数值（当前类型或内容无法参与算术运算）",
            code=ErrorCode.TYPE_ERROR,
            operator=operator,
            config=config,
        )
    return n


_SEQ_VALUE_KEYS = [
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


def _seq_config_properties() -> Dict[str, Any]:
    return {k: {} for k in _SEQ_VALUE_KEYS}


def _get_seq_refs(config: Dict[str, Any]) -> List[Any]:
    return [config[k] for k in _SEQ_VALUE_KEYS if config.get(k) not in (None, "")]


def _missing_seq_refs_hint(config: Dict[str, Any]) -> str:
    """
    _get_seq_refs 为空时的补充说明：多见于推理树里 ${step}.列名 解析成 None
    （如 es_extract 0 条命中得到 [{}]、列名不存在、或 step 尚未写入上下文）。
    """
    bad = [k for k in _SEQ_VALUE_KEYS if k in config and config.get(k) in (None, "")]
    if bad:
        return (
            "（当前 " + ", ".join(f"{k} 为空/未解析" for k in bad[:6])
            + (f" 等共{len(bad)}项" if len(bad) > 6 else "")
            + "；若曾使用 ${step}.列名，请检查上一步是否产出该列、es_extract 是否有命中，以及 reasoningId→sRID 过滤是否过严）"
        )
    return "（请配置 first_value，必要时补充 second_value/third_value…）"


def _to_numeric_list_strict(raw: Any, role: str, operator: str, config: dict) -> List[float]:
    if isinstance(raw, str) and raw.strip().startswith("["):
        raise OperatorException(
            f"{role}不能为 JSON 数组字符串（如 \"[1,2]\"），请传数组或数组引用",
            code=ErrorCode.TYPE_ERROR,
            operator=operator,
            config=config,
        )
    if not isinstance(raw, list) or not raw:
        raise OperatorException(
            f"{role}必须为非空数组",
            code=ErrorCode.TYPE_ERROR,
            operator=operator,
            config=config,
        )
    out: List[float] = []
    for i, x in enumerate(raw):
        n = to_number_or_none(x)
        if n is None:
            raise OperatorException(
                f"{role}第{i + 1}项无法转为数值",
                code=ErrorCode.TYPE_ERROR,
                operator=operator,
                config=config,
            )
        out.append(float(n))
    return out


def _is_non_empty_list(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0


def _as_scalar_or_raise(raw: Any, role: str, operator: str, config: dict) -> float:
    if _is_non_empty_list(raw):
        raise OperatorException(
            f"{role}在标量模式下不能为数组，请拆成 first_value/second_value 或使用单独一个列表做折叠运算",
            code=ErrorCode.TYPE_ERROR,
            operator=operator,
            config=config,
        )
    return float(_arithmetic_operand_or_raise(raw, role, operator, config))


def _broadcast_ratio(
    num_raw: Any,
    den_raw: Any,
    *,
    operator: str,
    config: dict,
    zero_msg: str,
    num_role: str = "first_value",
    den_role: str = "second_value",
) -> Any:
    """比值/占比：标量/标量、标量↔数组广播、数组/数组等长逐项。"""
    if not _is_non_empty_list(num_raw) and not _is_non_empty_list(den_raw):
        num = float(to_number(num_raw))
        den = float(to_number(den_raw))
        if den == 0:
            raise OperatorException(
                zero_msg,
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=operator,
                config=config,
            )
        return num / den
    if _is_non_empty_list(num_raw) and _is_non_empty_list(den_raw):
        left = _to_numeric_list_strict(num_raw, num_role, operator, config)
        right = _to_numeric_list_strict(den_raw, den_role, operator, config)
        if len(left) != len(right):
            raise OperatorException(
                f"{operator} 数组逐项运算要求长度一致: {len(left)} != {len(right)}",
                code=ErrorCode.SCHEMA_MISMATCH,
                operator=operator,
                config=config,
            )
        for i, d in enumerate(right):
            if d == 0.0:
                raise OperatorException(
                    f"{zero_msg}（第{i + 1}项）",
                    code=ErrorCode.CALC_LOGIC_ERROR,
                    operator=operator,
                    config=config,
                )
        return [a / b for a, b in zip(left, right)]
    if _is_non_empty_list(num_raw):
        left = _to_numeric_list_strict(num_raw, num_role, operator, config)
        s = _as_scalar_or_raise(den_raw, den_role, operator, config)
        if s == 0.0:
            raise OperatorException(zero_msg, code=ErrorCode.CALC_LOGIC_ERROR, operator=operator, config=config)
        return [a / s for a in left]
    right = _to_numeric_list_strict(den_raw, den_role, operator, config)
    s = _as_scalar_or_raise(num_raw, num_role, operator, config)
    out: List[float] = []
    for i, d in enumerate(right):
        if d == 0.0:
            raise OperatorException(
                f"{zero_msg}（第{i + 1}项）",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=operator,
                config=config,
            )
        out.append(s / d)
    return out


def _count_operand_for_by_count(raw: Any) -> Union[float, List[float]]:
    """
    条数口径：顶层「数值」计 1 条；顶层 [] 计 0 条。
    顶层为列表且元素均非 list/tuple 时，用 len(列表) 作为单侧总条数（与旧版一致）。
    顶层列表中若存在 list/tuple，则按「分组」计数：每组 list/tuple 用其 len，标量元素计 1。
    """
    if raw is None:
        return 0.0
    if not isinstance(raw, list):
        return 1.0
    if len(raw) == 0:
        return 0.0
    if any(isinstance(x, (list, tuple)) for x in raw):
        out: List[float] = []
        for x in raw:
            if isinstance(x, (list, tuple)):
                out.append(float(len(x)))
            else:
                out.append(1.0)
        return out
    return float(len(raw))


def _broadcast_count_ratio(
    num_counts: Union[float, List[float]],
    den_counts: Union[float, List[float]],
    *,
    operator: str,
    config: dict,
    zero_msg: str,
) -> Union[float, List[float]]:
    """条数比值/占比：标量↔列表广播，等长列表逐项相除。"""

    def safe_div(n: float, d: float, idx: Optional[int]) -> float:
        if d == 0.0:
            suffix = f"（第 {idx + 1} 项）" if idx is not None else ""
            raise OperatorException(
                zero_msg + suffix,
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=operator,
                config=config,
            )
        return n / d

    if isinstance(num_counts, float) and isinstance(den_counts, float):
        return safe_div(num_counts, den_counts, None)
    if isinstance(num_counts, float) and isinstance(den_counts, list):
        return [safe_div(num_counts, d, i) for i, d in enumerate(den_counts)]
    if isinstance(num_counts, list) and isinstance(den_counts, float):
        return [safe_div(n, den_counts, i) for i, n in enumerate(num_counts)]
    if isinstance(num_counts, list) and isinstance(den_counts, list):
        if len(num_counts) != len(den_counts):
            raise OperatorException(
                f"{operator} 分组条数运算要求两侧分组数量一致: {len(num_counts)} != {len(den_counts)}",
                code=ErrorCode.SCHEMA_MISMATCH,
                operator=operator,
                config=config,
            )
        return [safe_div(n, d, i) for i, (n, d) in enumerate(zip(num_counts, den_counts))]
    raise OperatorException(
        f"{operator} 条数计数类型组合异常",
        code=ErrorCode.CALC_LOGIC_ERROR,
        operator=operator,
        config=config,
    )


@OperatorRegistry.register("add")
class AddOperator(BaseOperator):
    """
    加法算子：计算多个数值的和
    
    功能说明：
    - 实现加数 + 加数 + ... 的加法运算
    - 支持任意个加数，全部求和
    
    计算逻辑：
    result = sum(all_operands)
    
    配置参数：
    - first_value/second_value/third_value...：按顺序提供加数，每项可以是：
      * 接口字段名: "price"
      * 上步骤引用: "${step_key}"
      * 数值: 100
    
    输入数据格式：
    base_data: {
        "price": 10.5,
        "tax": 2.0,
        "fee": 1.5
    }
    
    配置示例：
    {
        "operator": "add",
        "config": {
            "first_value": "price",
            "second_value": "tax",
            "third_value": "fee"
        }
    }
    
    输出格式：
    14.0 (数值)
    
    使用场景：
    - 计算总金额、总成本
    - 多个指标求和
    - 金融计算中的费用累加
    """
    name = "add"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": _seq_config_properties(),
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        refs = _get_seq_refs(config)
        if not refs:
            raise OperatorException(
                "加法请至少提供 first_value（可继续提供 second_value/third_value...）"
                + _missing_seq_refs_hint(config),
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        vals = [get_value(data, r, context) for r in refs]
        any_list = any(_is_non_empty_list(v) for v in vals)

        if any_list and len(vals) > 2:
            raise OperatorException(
                "加法在涉及数组时仅支持两个操作数（first_value 与 second_value），多标量请全部为数值",
                code=ErrorCode.CONFIG_FORMAT_ERROR,
                operator=self.name,
                config=config,
            )

        if len(vals) == 1:
            v0 = vals[0]
            if _is_non_empty_list(v0):
                nums = _to_numeric_list_strict(v0, "first_value", self.name, config)
                return float(sum(nums))
            return _arithmetic_operand_or_raise(v0, "first_value", self.name, config)

        if not any_list:
            total = 0.0
            for i, v in enumerate(vals):
                total += _arithmetic_operand_or_raise(v, f"加数（第{i + 1}项）", self.name, config)
            return total

        a, b = vals[0], vals[1]
        if _is_non_empty_list(a) and _is_non_empty_list(b):
            left = _to_numeric_list_strict(a, "first_value", self.name, config)
            right = _to_numeric_list_strict(b, "second_value", self.name, config)
            if len(left) != len(right):
                raise OperatorException(
                    f"数组逐项相加要求长度一致: {len(left)} != {len(right)}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    operator=self.name,
                    config=config,
                )
            return [x + y for x, y in zip(left, right)]
        if _is_non_empty_list(a):
            left = _to_numeric_list_strict(a, "first_value", self.name, config)
            s = _as_scalar_or_raise(b, "second_value", self.name, config)
            return [x + s for x in left]
        right = _to_numeric_list_strict(b, "second_value", self.name, config)
        s = _as_scalar_or_raise(a, "first_value", self.name, config)
        return [s + y for y in right]


@OperatorRegistry.register("subtract")
class SubtractOperator(BaseOperator):
    """
    减法算子（统一参数）：
    - 仅接受 first_value / second_value
    - 支持以下规则：
      1) first_value 为列表且 second_value 为空：首位依次减去后续项
      2) first_value 和 second_value 都为列表：逐项相减，返回列表
      3) 两者均为数值：直接相减
      4) first_value 为数值、second_value 为列表：first_value 减去列表全部值
    """
    name = "subtract"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {
            "first_value": {},
            "second_value": {},
        },
    }
    default_config = {}
    # 既支持标量结果，也支持列表逐项相减结果。
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        first_raw = get_value(data, config.get("first_value"), context)
        second_cfg = config.get("second_value")
        second_raw = get_value(data, second_cfg, context) if second_cfg not in (None, "") else None

        def _to_numeric_list(raw: Any, role: str) -> List[float]:
            if isinstance(raw, str) and raw.strip().startswith("["):
                raise OperatorException(
                    f"{role}不能为 JSON 数组字符串（如 \"[1,2]\"），请传数组或数组引用",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            if not isinstance(raw, list) or not raw:
                raise OperatorException(
                    f"{role}必须为非空数组",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            out: List[float] = []
            for i, x in enumerate(raw):
                n = to_number_or_none(x)
                if n is None:
                    raise OperatorException(
                        f"{role}第{i + 1}项无法转为数值",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(float(n))
            return out

        # 1) 单独一个列表：首位依次减去后续项（折叠）
        if isinstance(first_raw, list) and second_raw is None:
            nums = _to_numeric_list(first_raw, "first_value")
            ans = nums[0]
            for x in nums[1:]:
                ans -= x
            return ans

        # 多个标量：链式减 first - second - third ...
        extra_refs = [config[k] for k in _SEQ_VALUE_KEYS[2:] if config.get(k) not in (None, "")]
        extra_vals = [get_value(data, r, context) for r in extra_refs]
        if extra_vals and any(_is_non_empty_list(v) for v in [first_raw, second_raw, *extra_vals]):
            raise OperatorException(
                "减法在涉及数组时仅支持 first_value 与 second_value",
                code=ErrorCode.CONFIG_FORMAT_ERROR,
                operator=self.name,
                config=config,
            )

        if extra_vals and second_raw is not None:
            a = _arithmetic_operand_or_raise(first_raw, "first_value", self.name, config)
            b = _arithmetic_operand_or_raise(second_raw, "second_value", self.name, config)
            res = a - b
            for i, v in enumerate(extra_vals):
                res -= _arithmetic_operand_or_raise(v, f"减数（第{i + 3}项）", self.name, config)
            return res

        # 2) 两个列表：逐项相减 → 列表
        if isinstance(first_raw, list) and isinstance(second_raw, list):
            left = _to_numeric_list(first_raw, "first_value")
            right = _to_numeric_list(second_raw, "second_value")
            if len(left) != len(right):
                raise OperatorException(
                    f"两个列表逐项相减要求长度一致: {len(left)} != {len(right)}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    operator=self.name,
                    config=config,
                )
            return [a - b for a, b in zip(left, right)]

        # 3) 两个数值：直接相减
        if not isinstance(first_raw, list) and not isinstance(second_raw, list) and second_raw is not None:
            a = _arithmetic_operand_or_raise(first_raw, "first_value", self.name, config)
            b = _arithmetic_operand_or_raise(second_raw, "second_value", self.name, config)
            return a - b

        # 4) 标量 与 数组：逐项相减 → 列表
        if not isinstance(first_raw, list) and isinstance(second_raw, list):
            a = float(_arithmetic_operand_or_raise(first_raw, "first_value", self.name, config))
            right = _to_numeric_list(second_raw, "second_value")
            return [a - x for x in right]

        if isinstance(first_raw, list) and not isinstance(second_raw, list) and second_raw is not None:
            left = _to_numeric_list(first_raw, "first_value")
            b = float(_arithmetic_operand_or_raise(second_raw, "second_value", self.name, config))
            return [x - b for x in left]

        raise OperatorException(
            "subtract 参数组合不支持：请使用 first_value/second_value，并确保类型为数值或数值列表",
            code=ErrorCode.CONFIG_FORMAT_ERROR,
            operator=self.name,
            config=config,
        )


@OperatorRegistry.register("multiply")
class MultiplyOperator(BaseOperator):
    """
    乘法算子：计算多个数值的乘积
    
    功能说明：
    - 实现乘数1 * 乘数2 * ... 的乘法运算
    - 支持任意个乘数，全部相乘
    
    计算逻辑：
    result = product(all_operands)
    
    配置参数：
    - first_value/second_value/third_value...：按顺序提供乘数，每项可以是字段名、${step_key} 或数值
    
    输入数据格式：
    base_data: {
        "quantity": 5,
        "unit_price": 10.5,
        "quantity_weight": 2
    }
    
    配置示例：
    {
        "operator": "multiply",
        "config": {
            "first_value": "quantity",
            "second_value": "unit_price",
            "third_value": "quantity_weight"
        }
    }
    
    输出格式：
    105.0 (数值)
    
    使用场景：
    - 计算商品总价（数量*单价）
    - 计算复利
    - 计算面积、体积
    """
    name = "multiply"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": _seq_config_properties(),
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        refs = _get_seq_refs(config)
        if not refs:
            raise OperatorException(
                "乘法请至少提供 first_value（以及 second_value 等乘数）"
                + _missing_seq_refs_hint(config),
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        vals = [get_value(data, r, context) for r in refs]
        any_list = any(_is_non_empty_list(v) for v in vals)

        if any_list and len(vals) > 2:
            raise OperatorException(
                "乘法在涉及数组时仅支持两个操作数（first_value 与 second_value），多标量请全部为数值",
                code=ErrorCode.CONFIG_FORMAT_ERROR,
                operator=self.name,
                config=config,
            )

        if len(vals) == 1:
            v0 = vals[0]
            if _is_non_empty_list(v0):
                nums = _to_numeric_list_strict(v0, "first_value", self.name, config)
                p = 1.0
                for x in nums:
                    p *= x
                return p
            return _arithmetic_operand_or_raise(v0, "first_value", self.name, config)

        if not any_list:
            result = 1.0
            for i, v in enumerate(vals):
                result *= _arithmetic_operand_or_raise(v, f"乘数（第{i + 1}项）", self.name, config)
            return result

        a, b = vals[0], vals[1]
        if _is_non_empty_list(a) and _is_non_empty_list(b):
            left = _to_numeric_list_strict(a, "first_value", self.name, config)
            right = _to_numeric_list_strict(b, "second_value", self.name, config)
            if len(left) != len(right):
                raise OperatorException(
                    f"数组逐项相乘要求长度一致: {len(left)} != {len(right)}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    operator=self.name,
                    config=config,
                )
            return [x * y for x, y in zip(left, right)]
        if _is_non_empty_list(a):
            left = _to_numeric_list_strict(a, "first_value", self.name, config)
            s = _as_scalar_or_raise(b, "second_value", self.name, config)
            return [x * s for x in left]
        right = _to_numeric_list_strict(b, "second_value", self.name, config)
        s = _as_scalar_or_raise(a, "first_value", self.name, config)
        return [s * y for y in right]


@OperatorRegistry.register("divide")
class DivideOperator(BaseOperator):
    """
    除法算子：计算被除数除以多个除数的结果
    
    功能说明：
    - 实现被除数 ÷ 除数1 ÷ 除数2 ÷ ... 的除法运算
    - 支持多个除数，逐个相除
    - 完整浮点数精度，无默认截断
    
    计算逻辑：
    result = dividend / divisor1 / divisor2 / ...
    例: 1/3 = 0.3333333333333333 (完整精度)
    
    配置参数：
    - first_value (required): 被除数，可以是：
      * 接口字段名: "total"
      * 上步骤引用: "${step_key}"
      * 数值: 100
    - second_value: 除数（标量）或除数数组（list）
    
    输入数据格式：
    base_data: {
        "total_score": 90,
        "scale1": 10,
        "scale2": 3
    }
    
    配置示例：
    {
        "operator": "divide",
        "config": {
            "first_value": "total_score",
            "second_value": ["scale1", "scale2"]
        }
    }
    
    输出格式：
    3.0 (数值，完整精度)
    
    精度说明：
    - 返回完整浮点数精度，如需截断使用 precision_round 算子
    - 1/3 返回 0.3333333333333333，不是 0.333
    
    使用场景：
    - 计算平均值（总和/数量）
    - 计算比率（部分/整体）
    - 金融计算（本息和/时期数）
    - 科学计算（含小数计算）
    
    异常处理：
    - 除数为0时抛出错误
    - 必须提供 first_value 与 second_value
    """
    name = "divide"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": _seq_config_properties(),
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        first_raw = get_value(data, config.get("first_value"), context)
        second_cfg = config.get("second_value")
        second_raw = get_value(data, second_cfg, context) if second_cfg not in (None, "") else None

        # 不支持：单独一个数组（first_value=[a,b,c] 这种链式除）
        if second_raw is None:
            raise OperatorException(
                "divide 必须提供 second_value（除数或除数数组）；不支持仅用 first_value 数组做链式除",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )

        first_is_list = _is_non_empty_list(first_raw)
        second_is_list = _is_non_empty_list(second_raw)

        # 支持：数值/数值
        if not first_is_list and not second_is_list:
            a = float(_arithmetic_operand_or_raise(first_raw, "first_value", self.name, config))
            b = float(_arithmetic_operand_or_raise(second_raw, "second_value", self.name, config))
            if b == 0.0:
                raise OperatorException(
                    "second_value 不能为0",
                    code=ErrorCode.CALC_LOGIC_ERROR,
                    operator=self.name,
                    config=config,
                )
            return a / b

        # 支持：数值/数组（链式除：a / d1 / d2 / ... → 标量）
        if not first_is_list and second_is_list:
            a = float(_arithmetic_operand_or_raise(first_raw, "first_value", self.name, config))
            divs = _to_numeric_list_strict(second_raw, "second_value", self.name, config)
            res = a
            for i, d in enumerate(divs):
                if d == 0.0:
                    raise OperatorException(
                        f"second_value 第{i + 1}项不能为0",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                res /= d
            return res

        # 支持：数组/数组（逐项）
        if first_is_list and second_is_list:
            left = _to_numeric_list_strict(first_raw, "first_value", self.name, config)
            right = _to_numeric_list_strict(second_raw, "second_value", self.name, config)
            if len(left) != len(right):
                raise OperatorException(
                    f"数组逐项相除要求长度一致: {len(left)} != {len(right)}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    operator=self.name,
                    config=config,
                )
            for i, d in enumerate(right):
                if d == 0.0:
                    raise OperatorException(
                        f"second_value 第{i + 1}项不能为0",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
            return [a / b for a, b in zip(left, right)]

        # 支持：数组/数值（逐项：每个元素除以同一个除数 → 列表）
        if first_is_list and not second_is_list:
            left = _to_numeric_list_strict(first_raw, "first_value", self.name, config)
            b = float(_arithmetic_operand_or_raise(second_raw, "second_value", self.name, config))
            if b == 0.0:
                raise OperatorException(
                    "second_value 不能为0",
                    code=ErrorCode.CALC_LOGIC_ERROR,
                    operator=self.name,
                    config=config,
                )
            return [a / b for a in left]

        raise OperatorException(
            "divide 参数组合不支持：请使用 first_value/second_value，并确保类型为数值或数值列表",
            code=ErrorCode.CONFIG_FORMAT_ERROR,
            operator=self.name,
            config=config,
        )


@OperatorRegistry.register("power")
class PowerOperator(BaseOperator):
    """
    乘方算子：计算底数的指数次幂
    
    功能说明：
    - 实现底数^指数的幂运算
    - 底数和指数均为单值（不支持列表）
    
    计算逻辑：
    result = base ** exponent
    例: 2^3 = 8, 2^0.5 = 1.414...
    
    配置参数：
    - base (required): 底数，可以是：
      * 接口字段名: "original_value"
      * 上步骤引用: "${step_key}"
      * 数值: 2
    - exponent (required): 指数，格式同base
    
    输入数据格式：
    base_data: {
        "base_num": 2,
        "exp_num": 3
    }
    
    配置示例：
    {
        "operator": "power",
        "config": {
            "base": "base_num",
            "exponent": "exp_num"
        }
    }
    
    输出格式：
    8.0 (数值)
    
    支持的指数类型：
    - 整数: 2^3 = 8
    - 小数: 4^0.5 = 2.0（开方）
    - 负数: 2^-1 = 0.5（倒数）
    
    使用场景：
    - 复利计算（本利^期数）
    - 平方、立方等几何计算
    - 指数增长/衰减
    - 开方运算（指数=0.5）
    """
    name = "power"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value", "second_value"],
        "properties": {
            "first_value": {},
            "second_value": {},
        },
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        base_val = get_value(data, config.get("first_value"), context)
        exp_val = get_value(data, config.get("second_value"), context)

        if _is_non_empty_list(base_val) and _is_non_empty_list(exp_val):
            left = _to_numeric_list_strict(base_val, "first_value", self.name, config)
            right = _to_numeric_list_strict(exp_val, "second_value", self.name, config)
            if len(left) != len(right):
                raise OperatorException(
                    f"乘方数组逐项运算要求长度一致: {len(left)} != {len(right)}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    operator=self.name,
                    config=config,
                )
            return [pow(b, e) for b, e in zip(left, right)]

        if _is_non_empty_list(base_val):
            left = _to_numeric_list_strict(base_val, "first_value", self.name, config)
            ee = float(_arithmetic_operand_or_raise(exp_val, "second_value", self.name, config))
            return [pow(b, ee) for b in left]

        if _is_non_empty_list(exp_val):
            bb = float(_arithmetic_operand_or_raise(base_val, "first_value", self.name, config))
            right = _to_numeric_list_strict(exp_val, "second_value", self.name, config)
            return [pow(bb, e) for e in right]

        base = _arithmetic_operand_or_raise(base_val, "first_value", self.name, config)
        exp = _arithmetic_operand_or_raise(exp_val, "second_value", self.name, config)
        return pow(base, exp)


def _ensure_vector(val):
    """
    将标量、单元素或 JSON 数组字符串转为列表，便于与「向量」统一处理。
    仅用于余弦相似度：请求/前端常把向量写成 base_data 字符串 \"[1,2]\"，若不做解析则
    isinstance(vec, list) 为 False，原逻辑会直接 return 0.0，导致结果为 0 且无报错。
    """
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        try:
            return [float(val)]
        except (TypeError, ValueError):
            return None
    try:
        return [float(val)]
    except (TypeError, ValueError):
        return None


def _to_numeric_vector(val):
    """
    将取值结果转为数值向量（一维浮点列表）；非法返回 None。
    支持元素中再嵌套列表（如 [6,5,[2,7]]），按深度优先展平为 [6,5,2,7]，便于与「一一对应」多向量配置对齐。
    """
    vec = _ensure_vector(val)
    if vec is None:
        return None
    out: List[float] = []

    def collect(x: Any) -> bool:
        if isinstance(x, list):
            for y in x:
                if not collect(y):
                    return False
            return True
        n = safe_convert_to_number(x)
        if n is None:
            return False
        out.append(float(n))
        return True

    if not collect(vec):
        return None
    return out if out else None


@OperatorRegistry.register("absolute_value")
class AbsoluteOperator(BaseOperator):
    """计算数值的绝对值"""
    name = "absolute_value"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": _seq_config_properties(),
    }
    default_config = {}

    def _resolve_config(self, config):
        return super()._resolve_config(config)

    def execute(self, data: Dict[str, Any], config: Dict[str, Any], context: ExecutionContext) -> Any:
        fields = [config.get(k) for k in ("first_value", "second_value", "third_value", "fourth_value", "fifth_value") if config.get(k) not in (None, "")]
        if not fields:
            raise OperatorException(
                "请指定至少一个值,进行绝对值转换",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        out = []
        for i, ref in enumerate(fields):
            val = get_value(data, ref, context)
            if isinstance(val, list):
                for v in val:
                    n = to_number_or_none(v)
                    if n is None:
                        raise OperatorException(
                            f"绝对值转换失败：第{i + 1}项列表中存在无法转为数值的元素",
                            code=ErrorCode.TYPE_ERROR,
                            operator=self.name,
                            config=config,
                        )
                    out.append(abs(float(n)))
                continue
            n = to_number_or_none(val)
            if n is None:
                raise OperatorException(
                    f"绝对值转换失败：第{i + 1}项无法转为数值",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            out.append(abs(float(n)))
        return out[0] if len(out) == 1 else out



@OperatorRegistry.register("sin")
class SinOperator(BaseOperator):
    """
    正弦函数算子：计算角度的正弦值
    
    功能说明：
    - 计算输入角度（弧度制）的正弦值
    
    计算逻辑：
    result = sin(angle_in_radians)
    取值范围: [-1, 1]
    
    配置参数：
    - value (required): 角度（弧度制），可以是：
      * 接口字段名: "angle"
      * 上步骤引用: "${step_key}"
      * 数值: 1.57 (π/2)
    
    输入数据格式：
    base_data: {
        "angle_rad": 1.5707963267948966  // π/2
    }
    
    配置示例：
    {
        "operator": "sin",
        "config": {
            "value": "angle_rad"
        }
    }
    
    输出格式：
    1.0 (数值，范围[-1, 1])
    
    角度转换：
    - π/2 弧度 ≈ 1.5708 rad → sin = 1.0
    - π 弧度 ≈ 3.1416 rad → sin = 0.0
    - π/4 弧度 ≈ 0.7854 rad → sin ≈ 0.7071
    
    使用场景：
    - 波形分析
    - 三角形计算
    - 信号处理
    - 物理模拟
    """
    name = "sin"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
        if isinstance(val, list):
            out = []
            for v in val:
                num = safe_convert_to_number(v)
                if num is None:
                    raise OperatorException(
                        "sin 算子要求数值输入（列表中存在无法转数值的元素）",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                r = math.sin(float(num))
                if math.isnan(r) or math.isinf(r):
                    raise OperatorException(
                        "sin 计算结果无效（NaN 或 Inf）",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(r)
            return out
        num = safe_convert_to_number(val)
        if num is None:
            raise OperatorException(
                "sin 算子要求数值输入",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        result = math.sin(float(num))
        if math.isnan(result) or math.isinf(result):
            raise OperatorException(
                "sin 计算结果无效（NaN 或 Inf）",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return result


@OperatorRegistry.register("cos")
class CosOperator(BaseOperator):
    """
    余弦函数算子：计算角度的余弦值
    
    功能说明：
    - 计算输入角度（弧度制）的余弦值
    
    计算逻辑：
    result = cos(angle_in_radians)
    取值范围: [-1, 1]
    
    配置参数：
    - value (required): 角度（弧度制）
    
    输入数据格式：
    base_data: {
        "angle_rad": 0.0
    }
    
    配置示例：
    {
        "operator": "cos",
        "config": {
            "value": "angle_rad"
        }
    }
    
    输出格式：
    1.0 (数值，范围[-1, 1])
    
    使用场景：
    - 波形分析
    - 向量计算
    - 物理模拟
    """
    name = "cos"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
        if isinstance(val, list):
            out = []
            for v in val:
                num = safe_convert_to_number(v)
                if num is None:
                    raise OperatorException(
                        "cos 算子要求数值输入（列表中存在无法转数值的元素）",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                r = math.cos(float(num))
                if math.isnan(r) or math.isinf(r):
                    raise OperatorException(
                        "cos 计算结果无效（NaN 或 Inf）",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(r)
            return out
        num = safe_convert_to_number(val)
        if num is None:
            raise OperatorException(
                "cos 算子要求数值输入",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        result = math.cos(float(num))
        if math.isnan(result) or math.isinf(result):
            raise OperatorException(
                "cos 计算结果无效（NaN 或 Inf）",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return result


@OperatorRegistry.register("tan")
class TanOperator(BaseOperator):
    """
    正切函数算子：计算角度的正切值
    
    功能说明：
    - 计算输入角度（弧度制）的正切值
    
    计算逻辑：
    result = tan(angle_in_radians) = sin(angle) / cos(angle)
    
    配置参数：
    - value (required): 角度（弧度制）
    
    使用场景：
    - 斜率计算
    - 角度分析
    """
    name = "tan"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
        if isinstance(val, list):
            out = []
            for v in val:
                num = safe_convert_to_number(v)
                if num is None:
                    raise OperatorException(
                        "tan 算子要求数值输入（列表中存在无法转数值的元素）",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                r = math.tan(float(num))
                if math.isnan(r) or math.isinf(r):
                    raise OperatorException(
                        "tan 计算结果无效（NaN 或 Inf）",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(r)
            return out
        num = safe_convert_to_number(val)
        if num is None:
            raise OperatorException(
                "tan 算子要求数值输入",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        result = math.tan(float(num))
        if math.isnan(result) or math.isinf(result):
            raise OperatorException(
                "tan 计算结果无效（NaN 或 Inf）",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return result


@OperatorRegistry.register("log")
class LogOperator(BaseOperator):
    """
    对数算子：计算对数值
    
    功能说明：
    - 计算任意进制的对数
    - 不指定base时计算自然对数（ln）
    
    计算逻辑：
    result = log(value, base)
    若base为空: result = ln(value) = log_e(value)
    
    配置参数：
    - value (required): 对数的真数，必须为正数
    - base (optional): 对数的底数，默认为e
      * 不指定或为空: 计算自然对数
      * 指定数值: 计算该进制对数
    
    输入数据格式：
    base_data: {
        "number": 100,
        "base_num": 10
    }
    
    配置示例：
    {
        "operator": "log",
        "config": {
            "value": "number",
            "base": "base_num"
        }
    }
    
    输出格式：
    2.0 (数值)
    
    常见对数值：
    - log(100, 10) = 2.0
    - log(8, 2) = 3.0
    - ln(e) = 1.0
    
    使用场景：
    - 数据分析（取对数缩放）
    - 信息论（熵计算）
    - 复利计算（时间计算）
    
    异常处理：
    - value ≤ 0 时抛出错误
    - base ≤ 0 或 base = 1 时抛出错误
    """
    name = "log"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
        sec = config.get("second_value")
        base_val = get_value(data, sec, context) if sec not in (None, "") else None

        def _log_one(x, b):
            num = safe_convert_to_number(x)
            if num is None or num <= 0:
                raise OperatorException(
                    "log 算子要求真数为正数",
                    code=ErrorCode.CALC_LOGIC_ERROR,
                    operator=self.name,
                    config=config,
                )
            if b is not None:
                base_num = safe_convert_to_number(b)
                if base_num is None or base_num <= 0 or base_num == 1:
                    raise OperatorException(
                        "log 算子底数必须为正数且不等于1",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                return math.log(float(num), float(base_num))
            return math.log(float(num))

        if base_val is None:
            if _is_non_empty_list(val):
                return [_log_one(x, None) for x in val]
            return _log_one(val, None)

        if not _is_non_empty_list(val) and not _is_non_empty_list(base_val):
            return _log_one(val, base_val)

        if _is_non_empty_list(val) and _is_non_empty_list(base_val):
            left = _to_numeric_list_strict(val, "first_value", self.name, config)
            right = _to_numeric_list_strict(base_val, "second_value", self.name, config)
            if len(left) != len(right):
                raise OperatorException(
                    f"log 数组逐项运算要求长度一致: {len(left)} != {len(right)}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    operator=self.name,
                    config=config,
                )
            return [_log_one(x, b) for x, b in zip(left, right)]

        if _is_non_empty_list(val):
            left = _to_numeric_list_strict(val, "first_value", self.name, config)
            return [_log_one(x, base_val) for x in left]

        right = _to_numeric_list_strict(base_val, "second_value", self.name, config)
        return [_log_one(val, b) for b in right]


@OperatorRegistry.register("sqrt")
class SqrtOperator(BaseOperator):
    """
    平方根算子：计算数值的平方根
    
    功能说明：
    - 计算非负数的平方根
    
    计算逻辑：
    result = √value
    例: √16 = 4.0, √2 ≈ 1.414
    
    配置参数：
    - value (required): 被开方数，必须 ≥ 0
    
    输入数据格式：
    base_data: {
        "area": 16
    }
    
    配置示例：
    {
        "operator": "sqrt",
        "config": {
            "value": "area"
        }
    }
    
    输出格式：
    4.0 (数值)
    
    使用场景：
    - 几何计算（边长计算）
    - 统计学（标准差到方差的转换）
    - 距离计算
    
    异常处理：
    - value < 0 时抛出错误
    """
    name = "sqrt"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
        if isinstance(val, list):
            out = []
            for v in val:
                num = safe_convert_to_number(v)
                if num is None or num < 0:
                    raise OperatorException(
                        "sqrt 算子要求非负数输入（列表中存在非法元素）",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(math.sqrt(float(num)))
            return out
        num = safe_convert_to_number(val)
        if num is None or num < 0:
            raise OperatorException(
                "sqrt 算子要求非负数输入",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return math.sqrt(float(num))


@OperatorRegistry.register("factorial")
class FactorialOperator(BaseOperator):
    """
    阶乘算子：计算非负整数的阶乘
    
    功能说明：
    - 计算 n! = n × (n-1) × (n-2) × ... × 1
    
    计算逻辑：
    result = n!
    例: 5! = 120, 0! = 1
    
    配置参数：
    - value (required): 非负整数
    
    输入数据格式：
    base_data: {
        "n": 5
    }
    
    配置示例：
    {
        "operator": "factorial",
        "config": {
            "value": "n"
        }
    }
    
    输出格式：
    120 (整数)
    
    使用场景：
    - 组合数学（排列组合）
    - 概率计算
    - 统计学
    
    异常处理：
    - value < 0 或不是整数时抛出错误
    """
    name = "factorial"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    input_spec = {"type": "table"}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        val = get_value(data, config.get("first_value"), context)
        if isinstance(val, list):
            out = []
            for v in val:
                num = safe_convert_to_number(v)
                if num is None or num < 0 or num != int(num):
                    raise OperatorException(
                        "factorial 算子要求非负整数输入（列表中存在非法元素）",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(float(math.factorial(int(num))))
            return out
        num = safe_convert_to_number(val)
        if num is None or num < 0 or num != int(num):
            raise OperatorException(
                "factorial 算子要求非负整数输入",
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
            )
        return float(math.factorial(int(num)))


@OperatorRegistry.register("ratio")
class RatioOperator(BaseOperator):
    """
    比率算子：计算两个数的比值
    
    功能说明：
    - 计算分子/分母的比值
    
    计算逻辑：
    result = numerator / denominator
    
    配置参数：
    - numerator (required): 分子
    - denominator (required): 分母，不能为0
    
    输入数据格式：
    base_data: {
        "part": 1,
        "total": 3
    }
    
    配置示例：
    {
        "operator": "ratio",
        "config": {
            "numerator": "part",
            "denominator": "total"
        }
    }
    
    输出格式：
    0.3333333333333333 (完整精度)
    
    使用场景：
    - 计算占比
    - 财务比率分析
    - 概率计算
    """
    name = "ratio"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value", "second_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        num_raw = get_value(data, config.get("first_value"), context)
        den_raw = get_value(data, config.get("second_value"), context)
        return _broadcast_ratio(
            num_raw,
            den_raw,
            operator=self.name,
            config=config,
            zero_msg="比率算子分母不能为0",
            num_role="first_value",
            den_role="second_value",
        )


@OperatorRegistry.register("proportion")
class ProportionOperator(BaseOperator):
    """
    比例算子：计算部分占总体的比例
    
    功能说明：
    - 计算 部分 / 总体 的比例值
    - 与percentage相同，但返回小数而非百分比
    
    计算逻辑：
    result = part / total
    
    配置参数：
    - part (required): 部分值
    - total (required): 总体值，不能为0
    
    输入数据格式：
    base_data: {
        "passed": 85,
        "total": 100
    }
    
    配置示例：
    {
        "operator": "proportion",
        "config": {
            "part": "passed",
            "total": "total"
        }
    }
    
    输出格式：
    0.85 (小数)
    
    vs percentage:
    - proportion 返回 0.85
    - percentage 返回 85.0
    
    使用场景：
    - 通过率计算
    - 市场份额分析
    - 概率计算
    """
    name = "proportion"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value", "second_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        part_raw = get_value(data, config.get("first_value"), context)
        total_raw = get_value(data, config.get("second_value"), context)
        return _broadcast_ratio(
            part_raw,
            total_raw,
            operator=self.name,
            config=config,
            zero_msg="比例算子总体（分母）不能为0",
            num_role="first_value",
            den_role="second_value",
        )


@OperatorRegistry.register("ratio_by_count")
class RatioByCountOperator(BaseOperator):
    """比率（按个数）：count(A)/count(B)。支持数值↔数组广播与分组数组逐项（数值计 1 条，见 _count_operand_for_by_count）。"""
    name = "ratio_by_count"
    config_schema = {
        "type": "object",
        # 算子仅使用 first_value/second_value；上游可能夹带其它键，若 additionalProperties=False 会直接校验失败
        "additionalProperties": True,
        "required": ["first_value", "second_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        num_raw = get_value(data, config.get("first_value"), context)
        den_raw = get_value(data, config.get("second_value"), context)
        num = _count_operand_for_by_count(num_raw)
        den = _count_operand_for_by_count(den_raw)
        return _broadcast_count_ratio(
            num,
            den,
            operator=self.name,
            config=config,
            zero_msg="ratio_by_count 分母个数不能为0",
        )


@OperatorRegistry.register("proportion_by_count")
class ProportionByCountOperator(BaseOperator):
    """比例（按个数）：count(part)/count(total)。广播规则同 ratio_by_count。"""
    name = "proportion_by_count"
    config_schema = {
        "type": "object",
        "additionalProperties": True,
        "required": ["first_value", "second_value"],
        "properties": {"first_value": {}, "second_value": {}},
    }
    default_config = {}
    output_spec = None

    def execute(self, data, config, context: ExecutionContext):
        part_raw = get_value(data, config.get("first_value"), context)
        total_raw = get_value(data, config.get("second_value"), context)
        part = _count_operand_for_by_count(part_raw)
        total = _count_operand_for_by_count(total_raw)
        return _broadcast_count_ratio(
            part,
            total,
            operator=self.name,
            config=config,
            zero_msg="proportion_by_count 总体个数不能为0",
        )


@OperatorRegistry.register("matrix_multiply")
class MatrixMultiplyOperator(BaseOperator):
    """矩阵乘法：支持矩阵×矩阵 或 矩阵×标量"""
    name = "matrix_multiply"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["first_value"],
        "properties": {"first_value": {}, "second_value": {}, "third_value": {}, "mode": {}},
    }
    default_config = {}
    output_spec = {"type": "list"}

    def execute(self, data, config, context: ExecutionContext):
        # first_value -> matrix1, second_value -> matrix2(optional), third_value -> scalar(optional)
        m1_raw = get_value(data, config.get("first_value"), context)
        m2_field = config.get("second_value")
        scalar_raw = config.get("third_value")

        if m2_field:
            m2 = get_value(data, m2_field, context)
            if not isinstance(m1_raw, list) or not isinstance(m2, list):
                raise OperatorException(
                    "矩阵乘法要求二维列表输入",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            result = self._matrix_times_matrix(m1_raw, m2)
        elif scalar_raw is not None:
            scalar = safe_convert_to_number(scalar_raw)
            if scalar is None:
                raise OperatorException(
                    "标量必须为数值",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            result = self._matrix_times_scalar(m1_raw, float(scalar))
        else:
            raise OperatorException(
                "需提供 second_value（matrix2）或 third_value（scalar）",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        return result

    @staticmethod
    def _matrix_times_scalar(matrix, scalar):
        if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
            raise OperatorException("矩阵乘标量要求二维列表输入", code=ErrorCode.TYPE_ERROR)
        return [[v * scalar if isinstance(v, (int, float)) else v for v in row] if isinstance(row, list) else row for row in matrix]

    @staticmethod
    def _matrix_times_matrix(a, b):
        if not all(isinstance(row, list) for row in a) or not all(isinstance(row, list) for row in b):
            raise OperatorException("矩阵必须为二维列表")
        rows_a, cols_a = len(a), len(a[0]) if a else 0
        rows_b, cols_b = len(b), len(b[0]) if b else 0
        if cols_a != rows_b:
            raise OperatorException(f"矩阵维度不匹配: ({rows_a}×{cols_a}) × ({rows_b}×{cols_b})")
        result = []
        for i in range(rows_a):
            row = []
            for j in range(cols_b):
                val = sum(a[i][k] * b[k][j] for k in range(cols_a) if isinstance(a[i][k], (int, float)) and isinstance(b[k][j], (int, float)))
                row.append(val)
            result.append(row)
        return result


@OperatorRegistry.register("cosine_similarity")
class CosineSimilarityOperator(BaseOperator):
    """
    余弦相似度算子：计算向量间的相似度
    
    功能说明：
    - 计算两个或多个向量的余弦相似度
    - 2个向量返回标量，3+个向量返回相似度矩阵
    
    计算逻辑：
    cos_sim(a, b) = (a·b) / (|a| × |b|)
    其中 a·b 是点积，|a| 是向量模长
    
    取值范围: [-1, 1]
    - 1: 完全相同方向
    - 0: 正交
    - -1: 完全相反方向
    
    配置参数：
    - vectors (array): 向量列表（推荐）
      每个元素是数值列表：[1.0, 2.0, 3.0]
    - first_value, second_value: 两个向量（字段名、${step_key} 或字面量）
    
    输入数据格式：
    base_data: {
        "vec1": [1, 2, 3],
        "vec2": [4, 5, 6]
    }
    
    配置示例：
    {
        "operator": "cosine_similarity",
        "config": {
            "vectors": ["vec1", "vec2"]
        }
    }
    
    输出格式：
    - 2个向量: 0.9746 (标量)
    - 3+个向量: [[1.0, 0.97, ...], [0.97, 1.0, ...], ...] (矩阵)
    
    向量要求：
    - 所有向量长度必须相同
    - 不能有零向量（模长为0）
    - 只接受数值，不支持JSON数组字符串
    
    使用场景：
    - 文本相似度（词向量）
    - 推荐系统（用户相似度）
    - 图像检索
    - 聚类分析
    
    异常处理：
    - 向量长度不一致时报错
    - 零向量时报错
    """
    name = "cosine_similarity"
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "vectors": {"type": "array"},
            "first_value": {},
            "second_value": {},
        },
    }
    default_config = {}

    def _resolve_config(self, config):
        merged = super()._resolve_config(config)
        if merged.get("vectors") and isinstance(merged["vectors"], list):
            return merged
        return merged

    def execute(self, data, config, context: ExecutionContext):
        vectors_config = config.get("vectors")
        if vectors_config is not None and len(vectors_config) > 0:
            # N 个向量
            vecs = []
            for key in vectors_config:
                raw = get_value(data, key, context)
                v = _to_numeric_vector(raw)
                if v is None:
                    raise OperatorException(
                        f"向量数据无效或非数值: {key}",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                vecs.append(v)
        else:
            v1 = get_value(data, config.get("first_value"), context)
            v2 = get_value(data, config.get("second_value"), context)

            def _is_vector_list(x) -> bool:
                # 外层为「向量组」：每一项是一条向量（数值列表，可含嵌套以便展平），不要求向量内部元素也是 list
                return isinstance(x, list) and len(x) > 0 and all(isinstance(row, list) for row in x)

            # 新增：支持两组“向量列表”一一对应计算，返回相似度列表
            if _is_vector_list(v1) or _is_vector_list(v2):
                v1_list = v1 if _is_vector_list(v1) else None
                v2_list = v2 if _is_vector_list(v2) else None
                if v1_list is not None and v2_list is not None:
                    if len(v1_list) != len(v2_list):
                        raise OperatorException(
                            f"first_value 与 second_value 为二维数组时，要求外层长度一致: {len(v1_list)} != {len(v2_list)}",
                            code=ErrorCode.SCHEMA_MISMATCH,
                            operator=self.name,
                            config=config,
                        )
                    pairs = list(zip(v1_list, v2_list))
                elif v1_list is not None:
                    base_v2 = _to_numeric_vector(v2)
                    if base_v2 is None:
                        raise OperatorException(
                            "second_value 无效或非数值",
                            code=ErrorCode.TYPE_ERROR,
                            operator=self.name,
                            config=config,
                        )
                    pairs = [(a, base_v2) for a in v1_list]
                else:
                    base_v1 = _to_numeric_vector(v1)
                    if base_v1 is None:
                        raise OperatorException(
                            "first_value 无效或非数值",
                            code=ErrorCode.TYPE_ERROR,
                            operator=self.name,
                            config=config,
                        )
                    pairs = [(base_v1, b) for b in v2_list]  # type: ignore[list-item]

                def _cos_pair(a, b):
                    aa = _to_numeric_vector(a)
                    bb = _to_numeric_vector(b)
                    if aa is None or bb is None:
                        raise OperatorException(
                            "向量数据无效或非数值",
                            code=ErrorCode.TYPE_ERROR,
                            operator=self.name,
                            config=config,
                        )
                    if len(aa) != len(bb):
                        raise OperatorException(
                            f"各向量长度须一致，当前长度: {[len(aa), len(bb)]}",
                            code=ErrorCode.CONFIG_INVALID,
                            operator=self.name,
                            config=config,
                        )
                    dot = sum(x * y for x, y in zip(aa, bb))
                    norm_a = math.sqrt(sum(x * x for x in aa))
                    norm_b = math.sqrt(sum(y * y for y in bb))
                    if norm_a == 0 or norm_b == 0:
                        raise OperatorException(
                            "存在模长为 0 的向量，无法计算余弦相似度",
                            code=ErrorCode.CALC_LOGIC_ERROR,
                            operator=self.name,
                            config=config,
                        )
                    return dot / (norm_a * norm_b)

                return [_cos_pair(a, b) for a, b in pairs]

            vec1 = _to_numeric_vector(v1)
            vec2 = _to_numeric_vector(v2)
            if vec1 is None:
                raise OperatorException(
                    "first_value 无效或非数值",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            if vec2 is None:
                raise OperatorException(
                    "second_value 无效或非数值",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                    config=config,
                )
            vecs = [vec1, vec2]

        if len(vecs) == 1:
            return 1.0

        n = len(vecs)
        lens = [len(v) for v in vecs]
        if len(set(lens)) != 1:
            raise OperatorException(
                f"各向量长度须一致，当前长度: {lens}",
                code=ErrorCode.CONFIG_INVALID,
                operator=self.name,
                config=config,
            )

        def cos_pair(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                raise OperatorException(
                    "存在模长为 0 的向量，无法计算余弦相似度",
                    code=ErrorCode.CALC_LOGIC_ERROR,
                    operator=self.name,
                    config=config,
                )
            return dot / (norm_a * norm_b)

        if n == 2:
            return cos_pair(vecs[0], vecs[1])

        # N≥3: 返回完整对称相似度矩阵（二维列表）
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                val = cos_pair(vecs[i], vecs[j])
                matrix[i][j] = val
                matrix[j][i] = val
        return matrix
