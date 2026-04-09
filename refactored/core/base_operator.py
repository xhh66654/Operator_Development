"""算子基类：统一异常处理、上下文、配置校验"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .context import ExecutionContext
from .exceptions import ErrorCode, OperatorException
from .config_schema import validate_operator_config
from .data_model import DataValue


def _build_seq_value_keys(max_items: int = 50) -> list[str]:
    """
    生成 first_value/second_value/... 的序号槽位 key 列表（支持到 fiftieth_value）。
    用于后端统一从 config 里读出“按位置”的多段输入。
    """
    unit_words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
    }
    teen_words = {
        11: "eleventh",
        12: "twelfth",
        13: "thirteenth",
        14: "fourteenth",
        15: "fifteenth",
        16: "sixteenth",
        17: "seventeenth",
        18: "eighteenth",
        19: "nineteenth",
    }
    tens_words = {
        10: "tenth",
        20: "twentieth",
        30: "thirtieth",
        40: "fortieth",
        50: "fiftieth",
    }

    keys: list[str] = []
    for n in range(1, int(max_items) + 1):
        if n in unit_words:
            word = unit_words[n]
        elif n in teen_words:
            word = teen_words[n]
        elif n in tens_words:
            word = tens_words[n]
        elif 21 <= n <= 29:
            word = "twenty_" + unit_words[n - 20]
        elif 31 <= n <= 39:
            word = "thirty_" + unit_words[n - 30]
        elif 41 <= n <= 49:
            word = "forty_" + unit_words[n - 40]
        else:
            # 目前仅要求到 50；超出范围明确失败，避免静默生成错误 key
            raise ValueError(f"unsupported seq value slot: {n}")
        keys.append(f"{word}_value")
    return keys


_LEGACY_INPUT_KEYS = {
    # 多来源/字段列表风格
    "fields",
    "field",
    "operands",
    "inputs",
    "input",
    # 常见“数据来源”旧键
    "source",
    "source_field",
    # 二元/多元算术旧键（顺序参数应替代它们）
    "primary",
    "secondary",
    "minuend",
    "subtrahends",
    "dividend",
    "divisors",
    "numerator",
    "denominator",
    "part",
    "total",
    "base",
    "exponent",
    "vectors",
    "matrix1",
    "matrix2",
    "value",
}


def _reject_legacy_input_keys(operator: str, raw_config: Dict[str, Any]) -> None:
    """
    强约束：禁止使用旧“输入指定”键。
    统一只用 first_value/second_value/third_value... 作为输入来源表达。

    注意：这里不限制算子的“业务参数键”（如 size/page_size/query/decimal_places 等），
    仅拦截旧输入键，避免影响提取类算子等的其它配置项。
    """
    if not isinstance(raw_config, dict) or not raw_config:
        return
    bad = [k for k in raw_config.keys() if str(k) in _LEGACY_INPUT_KEYS]
    if bad:
        bad_sorted = ", ".join(sorted(set(map(str, bad)))[:20])
        raise OperatorException(
            f"{operator} 配置包含不允许的输入键；请仅使用 first_value/second_value/...。发现非法键: {bad_sorted}",
            code=ErrorCode.CONFIG_FORMAT_ERROR,
            operator=operator,
            config=raw_config,
        )


class BaseOperator(ABC):
    """算子基类：统一异常处理、上下文注入、配置校验"""

    name: str = "base"
    config_schema: Optional[dict] = None
    default_config: Optional[dict] = None
    input_spec: Optional[dict] = None
    output_spec: Optional[dict] = None

    _SEQ_KEYS = _build_seq_value_keys(50)


    def _is_extract_operator(self) -> bool:
        n = (self.name or "").strip()
        return n.endswith("_extract") or n.startswith("extract_") or n in {"extract_only", "json_extract"}

    def _is_weight_operator(self) -> bool:
        return "weight" in (self.name or "").strip()

    def _normalize_sequential_values(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一参数风格：first_value/first_value... -> 各算子常用参数。
        仅用于非提取、非权重算子。
        """
        if self._is_extract_operator() or self._is_weight_operator():
            return merged

        # 已完成全量算子内部改造：输入只允许顺序槽位 first_value/second_value/...
        # 因此不再将顺序参数回填到任何旧命名，避免旧键在内部“复活”。
        return dict(merged)

    def run(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> Any:
        """统一入口：合并/规范化配置 -> 校验 -> 执行 -> 异常转 OperatorException"""
        ctx = context or ExecutionContext()
        if isinstance(config, dict):
            _reject_legacy_input_keys(self.name, config)
            config = self._resolve_config(config)
        try:
            validate_operator_config(self.name, config, self.config_schema)
            self.validate_input(data)
            raw_result = self.execute(data, config, ctx)
            data_value = DataValue.from_python(raw_result)
            self.validate_output(data_value)
            return data_value
        except OperatorException:
            raise
        except Exception as e:
            raise OperatorException(
                message=str(e),
                code=ErrorCode.CALC_LOGIC_ERROR,
                operator=self.name,
                config=config,
                cause=e,
            )

    def validate_input(self, data: Dict[str, Any]) -> None:
        """Optional input contract check."""
        if not self.input_spec:
            return
        expected = str(self.input_spec.get("type") or "").strip()
        if not expected:
            return
        actual = DataValue.from_python(data)
        if expected == "table":
            if actual.type not in {"table", "scalar"}:
                raise OperatorException(
                    f"输入类型不匹配: 期望 {expected}, 实际 {actual.type}",
                    code=ErrorCode.TYPE_ERROR,
                    operator=self.name,
                )
        elif actual.type != expected:
            raise OperatorException(
                f"输入类型不匹配: 期望 {expected}, 实际 {actual.type}",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
            )

    def validate_output(self, data_value: DataValue) -> None:
        """Optional output contract check."""
        if not self.output_spec:
            return
        expected = str(self.output_spec.get("type") or "").strip()
        if expected and data_value.type != expected:
            raise OperatorException(
                f"输出类型不匹配: 期望 {expected}, 实际 {data_value.type}",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
            )

    def _resolve_config(self, config: Dict) -> Dict:
        """合并默认配置，并将 config 中所有字符串值的中文标点统一转为英文标点"""
        from ..operators._common import normalize_config_punct
        base = self.default_config or {}
        merged = {**base, **config}
        merged = normalize_config_punct(merged)
        return self._normalize_sequential_values(merged)

    @abstractmethod
    def execute(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        """子类实现：纯计算逻辑"""
        pass
