"""算子基类：统一异常处理、上下文、配置校验"""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .context import ExecutionContext
from .exceptions import ErrorCode, OperatorException
from .config_schema import validate_operator_config
from .data_model import DataValue

_DEBUG_STEP_CTX_KEY = "_debug_step_id"


def operator_console_debug_enabled() -> bool:
    """
    为 True 时，每次算子 run() 成功后在控制台打印算子名与 python 结果。
    环境变量（任一为 1/true/on）：CALC_DEBUG_OPERATOR_RESULT、CALC_DEBUG_STEP_RESULT。
    """
    v = os.environ.get("CALC_DEBUG_OPERATOR_RESULT", os.environ.get("CALC_DEBUG_STEP_RESULT", "")).strip().lower()
    return v in ("1", "true", "yes", "on")


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
        统一参数风格：first_calue/first_value... -> 各算子常用参数。
        仅用于非提取、非权重算子。
        """
        if self._is_extract_operator() or self._is_weight_operator():
            return merged

        out = dict(merged)
        #修改冗余，这里除了阈值算子，其他的都采用first_value,second_value....
        if self.name in {
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "ratio",
            "proportion",
            "log",
            "sin",
            "cos",
            "tan",
            "sqrt",
            "factorial",
            "max",
            "min",
            "ratio_by_count",
            "proportion_by_count",
            "percentile",
            "weighted_average",
            "weighted_sum_squares",
            "weighted_variance",
            "weighted_std",
            "weighted_log_sum_square",
            "weighted_exp_sum_square",
            "weighted_log_mean",
            "weighted_exp_mean",
            "weighted_sum_of_squared_deviations",
            "min_max_normalize",
            "cosine_similarity",
            "euclidean_distance",
            "angle_between",
            "vector_angle",
            "time_add",
            "time_subtract",
            "average_time",
            "precision_round",
        }:
            return out

        seq_vals = [out[k] for k in self._SEQ_KEYS if out.get(k) not in (None, "")]
        if not seq_vals:
            return out

        if out.get("fields") in (None, ""):
            out["fields"] = list(seq_vals)
        if out.get("inputs") in (None, ""):
            out["inputs"] = list(seq_vals)
        if out.get("input") in (None, "") and seq_vals:
            out["input"] = seq_vals[0]
        if out.get("value") in (None, "") and seq_vals:
            out["value"] = seq_vals[0]
        if out.get("primary") in (None, "") and seq_vals:
            out["primary"] = seq_vals[0]
        if out.get("secondary") in (None, "") and len(seq_vals) >= 2:
            out["secondary"] = seq_vals[1] if len(seq_vals) == 2 else seq_vals[1:]
        if out.get("minuend") in (None, "") and seq_vals:
            out["minuend"] = seq_vals[0]
        if out.get("subtrahends") in (None, "") and len(seq_vals) >= 2:
            out["subtrahends"] = seq_vals[1:] if len(seq_vals) > 2 else [seq_vals[1]]
        if out.get("dividend") in (None, "") and seq_vals:
            out["dividend"] = seq_vals[0]
        if out.get("divisors") in (None, "") and len(seq_vals) >= 2:
            out["divisors"] = seq_vals[1:] if len(seq_vals) > 2 else [seq_vals[1]]
        if out.get("numerator") in (None, "") and seq_vals:
            out["numerator"] = seq_vals[0]
        if out.get("denominator") in (None, "") and len(seq_vals) >= 2:
            out["denominator"] = seq_vals[1]
        if out.get("part") in (None, "") and seq_vals:
            out["part"] = seq_vals[0]
        if out.get("total") in (None, "") and len(seq_vals) >= 2:
            out["total"] = seq_vals[1]
        if out.get("base") in (None, "") and seq_vals:
            out["base"] = seq_vals[0]
        if out.get("exponent") in (None, "") and len(seq_vals) >= 2:
            out["exponent"] = seq_vals[1]
        if out.get("vectors") in (None, "") and len(seq_vals) >= 2:
            out["vectors"] = [seq_vals[0], seq_vals[1]]
        if out.get("matrix1") in (None, "") and seq_vals:
            out["matrix1"] = seq_vals[0]
        if out.get("matrix2") in (None, "") and len(seq_vals) >= 2:
            out["matrix2"] = seq_vals[1]
        if out.get("scalar") in (None, "") and len(seq_vals) >= 3:
            out["scalar"] = seq_vals[2]
        return out

    def run(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> Any:
        """统一入口：合并/规范化配置 -> 校验 -> 执行 -> 异常转 OperatorException"""
        ctx = context or ExecutionContext()
        if isinstance(config, dict):
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
