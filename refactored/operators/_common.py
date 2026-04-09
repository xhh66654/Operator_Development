"""
算子公共：字段取值、数值转换、精度常量、config 规范化、标点标准化。

统一 config 规范（与接口/前端约定一致）：
- 输入数据来源：统一使用顺序参数 first_value/second_value/third_value/...（可为字段名或 ${step_key} 引用）。
- 接口字段名指定取数位置，get_value(data, ref, context) 解析：record 中的字段名、${step_key}、字面量。
- 前端直接传的配置（如权重、阈值）：weights、threshold、ranges、decimal_places 等，可为字面量或数据源引用。
"""
from typing import Any, Dict, List, Optional, Tuple

from ..core.exceptions import OperatorException, ErrorCode
from ..utils import extract_field_value, safe_convert_to_number

# 精度策略：计算结果不做默认截断，保留浮点数完整精度
# 需要控制精度时，由用户显式使用 precision_round 算子或前端配置 decimal_places

# 中文→英文标点映射（仅标点符号，不影响中文字符/字段名）
_PUNCT_TABLE = str.maketrans({
    '\uff0c': ',',   # ，全角逗号
    '\u3001': ',',   # 、顿号
    '\u201c': '"',   # " 左双引号
    '\u201d': '"',   # " 右双引号
    '\u2018': "'",   # ' 左单引号
    '\u2019': "'",   # ' 右单引号
    '\uff1b': ';',   # ；全角分号
    '\uff1a': ':',   # ：全角冒号
    '\u3002': '.',   # 。句号
    '\uff08': '(',   # （全角左括号
    '\uff09': ')',   # ）全角右括号
    '\u300a': '"',   # 《
    '\u300b': '"',   # 》
})

_DEPRECATED_PARAM_WARNINGS_KEY = "_deprecated_param_warnings"


def normalize_punct(s: str) -> str:
    """将字符串中的中文标点转为对应英文标点，中文字符本身不受影响。"""
    if not isinstance(s, str):
        return s
    return s.translate(_PUNCT_TABLE)


def normalize_config_punct(config: Any) -> Any:
    """
    递归将 config 中所有字符串值的中文标点转为英文标点。
    仅处理 dict/list/str，不改变数字/布尔。
    """
    if isinstance(config, dict):
        return {k: normalize_config_punct(v) for k, v in config.items()}
    if isinstance(config, list):
        return [normalize_config_punct(v) for v in config]
    if isinstance(config, str):
        return normalize_punct(config)
    return config


def normalize_primary_secondary(
    config: Dict[str, Any],
    primary_key: str = "primary",
    secondary_key: str = "secondary",
    legacy_primary: str = "minuend",
    legacy_secondary: str = "subtrahends",
) -> Dict[str, Any]:
    """
    历史兼容函数：在两组“语义等价”的键之间做互相回填。

    说明：
    - 本项目的**输入取值**已统一为顺序槽位 first_value/second_value/...；该函数仅用于存量代码在内部仍使用
      primary/secondary（或其它 legacy_* 命名）时的过渡兼容，不建议在新算子中继续扩散使用。
    """
    if not config:
        return dict(config)
    out = dict(config)
    if out.get(primary_key) is not None and out.get(legacy_primary) is None:
        out = {**out, legacy_primary: out[primary_key]}
    if out.get(secondary_key) is not None and out.get(legacy_secondary) is None:
        out = {**out, legacy_secondary: out[secondary_key]}
    if out.get(legacy_primary) is not None and out.get(primary_key) is None:
        out = {**out, primary_key: out[legacy_primary]}
    if out.get(legacy_secondary) is not None and out.get(secondary_key) is None:
        out = {**out, secondary_key: out[legacy_secondary]}
    return out


def normalize_config_input(
    config: Dict[str, Any],
    single_key: str = "input",
    multi_key: str = "inputs",
    legacy_single_keys: Tuple[str, ...] = ("field", "source", "source_field", "list_field"),
    legacy_multi_keys: Tuple[str, ...] = ("fields", "operands"),
) -> Dict[str, Any]:
    """
    历史兼容函数：已不再做旧键映射/回填，保留函数名仅为避免调用方报错。
    当前只返回 config 的浅拷贝。
    """
    return dict(config) if isinstance(config, dict) else {"value": config}


def _ctx(context) -> Dict[str, Any]:
    """从 ExecutionContext 抽取，给 extract_field_value 使用"""
    return context._store if context else {}


def rows_to_field_list_dict(rows: List[Dict[str, Any]]) -> List[Dict[str, List[Any]]]:
    """
    提取类算子统一输出：单行字典，键为字段名、值为该列在行方向上的取值列表，
    再包一层列表 → [{字段名: [值, ...], ...}]。
    无数据时返回 [{}]，与「列表套字典」约定一致。
    """
    if not rows:
        return [{}]
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    out: Dict[str, List[Any]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            out[k].append(r.get(k))
    return [out]


def _looks_like_context_ref(s: str) -> bool:
    t = s.strip()
    return t.startswith("${") and "}" in t


def get_value(data: Dict, field_or_expr: Any, context) -> Any:
    """从 data 或上下文中取值；支持字段名、${step_key}、字面量、或多来源列表。"""
    if field_or_expr is None:
        return None
    if isinstance(field_or_expr, (list, tuple)):
        if any(isinstance(x, (list, dict, tuple)) for x in field_or_expr):
            return list(field_or_expr)
        merged = []
        for item in field_or_expr:
            if isinstance(item, str):
                v = extract_field_value(data, item, _ctx(context))
                # 严格模式：上下文引用取不到值直接报错（避免静默丢失导致算子“看起来还能算”）。
                if v is None and _looks_like_context_ref(item):
                    raise OperatorException(
                        f"上下文引用未取到值: {item}",
                        code=ErrorCode.DATA_NOT_FOUND,
                    )
                # data 中无此字段且非上下文引用时，视为字面量（如集合算子的 ["苹果","香蕉"]）
                if v is None and not _looks_like_context_ref(item):
                    v = item
            else:
                v = item
            if v is None:
                continue
            if isinstance(v, list):
                merged.extend(v)
            else:
                merged.append(v)
        return merged
    if isinstance(field_or_expr, str):
        v = extract_field_value(data, field_or_expr, _ctx(context))
        if v is None and _looks_like_context_ref(field_or_expr):
            raise OperatorException(
                f"上下文引用未取到值: {field_or_expr}",
                code=ErrorCode.DATA_NOT_FOUND,
            )
        return v
    return extract_field_value(data, field_or_expr, _ctx(context))


def to_number(val: Any) -> float:
    """
    将单值或列表转为用于计算的数值（列表则求和）。
    关键约束：
    - 字符串：若 safe_convert_to_number 失败，则抛出异常。
    - 列表：任一元素无法转为数字则抛出异常。
    """
    if val is None:
        raise OperatorException(
            "数据来源为空或未找到，无法转为数字",
            code=ErrorCode.DATA_NOT_FOUND,
        )
    # 列表：逐个元素严格检查
    if isinstance(val, list):
        total: float = 0.0
        for v in val:
            n = safe_convert_to_number(v)
            if n is None:
                raise OperatorException(
                    f"列表中存在无法转为数字的元素: {v}",
                    code=ErrorCode.FORMAT_ERROR,
                )
            total += float(n)
        return total
    # 字符串：必须能转为数字，否则报错
    if isinstance(val, str):
        n = safe_convert_to_number(val)
        if n is None:
            raise OperatorException(
                f"无法将字符串转为数字: {val}",
                code=ErrorCode.FORMAT_ERROR,
            )
        return float(n)
    # 其它类型：仍尝试数值化，不可转则报错
    n = safe_convert_to_number(val)
    if n is None:
        raise OperatorException(
            f"值无法转为数字: {val!r}",
            code=ErrorCode.FORMAT_ERROR,
        )
    return float(n)


def to_number_or_none(val: Any) -> Optional[float]:
    """
    严格转为数值：仅当可明确转为数字时返回 float，否则返回 None。
    字符串形如 \"[1,2]\"（JSON 数组字符串）返回 None，供算术算子报错用。
    """
    if val is None:
        return None
    if isinstance(val, str) and val.strip().startswith("["):
        return None
    if isinstance(val, list):
        if not val:
            return None
        for v in val:
            if safe_convert_to_number(v) is None:
                return None
        return sum(safe_convert_to_number(v) for v in val)
    n = safe_convert_to_number(val)
    return float(n) if n is not None else None
