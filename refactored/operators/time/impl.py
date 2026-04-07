"""时间类算子：时间加、时间减、平均时间"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from ...utils import safe_convert_to_number
from .._common import get_value


def _looks_like_context_ref(s: Any) -> bool:
    return isinstance(s, str) and s.strip().startswith("${") and "}" in s.strip()


def _get_time_ref(data: Dict, ref: Any, context: ExecutionContext) -> Any:
    """
    时间入参解析：${step}/字段名/列引用与「非引用字符串字面量」（ISO 时间、时间戳字符串等）。
    extract_field_value 对字面量 ISO 字符串会返回 None，此处与列表分支行为对齐。
    """
    if ref is None:
        return None
    if isinstance(ref, (int, float)):
        return ref
    if not isinstance(ref, str):
        return get_value(data, ref, context)
    v = get_value(data, ref, context)
    if v is not None:
        return v
    if _looks_like_context_ref(ref):
        return None
    return ref


def _resolve_numeric_config(data: Dict, ref: Any, context: ExecutionContext) -> Optional[float]:
    """duration_seconds 等：支持数字、${step} 引用、数字字符串字面量。"""
    if ref is None:
        return None
    if isinstance(ref, (int, float)):
        return float(ref)
    v = get_value(data, ref, context)
    if v is not None:
        n = safe_convert_to_number(v) if not isinstance(v, (int, float)) else float(v)
        return float(n) if n is not None else None
    if isinstance(ref, str):
        if _looks_like_context_ref(ref):
            return None
        n = safe_convert_to_number(ref.strip())
        return float(n) if n is not None else None
    return None


def _get_duration_dict(data: Dict, raw: Any, context: ExecutionContext) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and _looks_like_context_ref(raw):
        v = get_value(data, raw, context)
        return v if isinstance(v, dict) else {}
    return {}

# 常见时间字符串格式，按优先级尝试
_DATETIME_FMT = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
]


def _parse_datetime(s: Any) -> datetime:
    if isinstance(s, (int, float)):
        return datetime.utcfromtimestamp(float(s))
    s = str(s).strip()
    for fmt in _DATETIME_FMT:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise OperatorException(f"无法解析时间: {s}", code=ErrorCode.FORMAT_ERROR)


def _to_timestamp(dt: datetime) -> float:
    return dt.timestamp()


def _from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts)


@OperatorRegistry.register("time_add")
class TimeAddOperator(BaseOperator):
    """
    时间加：基准时间 + 时间间隔
    config: base_time（字段或 "${step_key}"）, duration_seconds（秒数）或 duration（{"days":0,"hours":0,"minutes":0,"seconds":0}）
    支持单值或列表输入，列表时返回结果列表。
    """
    name = "time_add"
    config_schema = {
        "type": "object",
        "properties": {
            "base_time": {},
            "duration_seconds": {},
            "duration": {},
            "first_value": {},
            "second_value": {},
            "third_value": {},
        },
    }
    default_config = {}

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("base_time") in (None, ""):
            for k in ("first_value", "value", "input", "primary"):
                v = c.get(k)
                if v not in (None, ""):
                    c["base_time"] = v
                    break
        if c.get("duration_seconds") in (None, ""):
            for k in ("second_value", "secondary"):
                v = c.get(k)
                if v not in (None, ""):
                    c["duration_seconds"] = v
                    break
        if c.get("duration") in (None, "") and c.get("third_value") not in (None, ""):
            c["duration"] = c.get("third_value")
        return c

    def execute(self, data, config, context: ExecutionContext):
        base = _get_time_ref(data, config.get("base_time") or config.get("first_value"), context)
        if base is None:
            raise OperatorException(
                "first_value（基准时间）不能为空或未解析",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        ds = config.get("duration_seconds")
        dconf = config.get("duration")
        if ds is not None:
            ds_n = _resolve_numeric_config(data, ds, context)
            if ds_n is None:
                raise OperatorException(
                    "second_value（增量秒数）无效或未解析",
                    code=ErrorCode.CONFIG_MISSING,
                    operator=self.name,
                    config=config,
                )
            delta = timedelta(seconds=ds_n)
        else:
            d = _get_duration_dict(data, dconf, context)
            if not d:
                raise OperatorException(
                    "请提供 second_value（秒数）或 third_value（分量时长对象）",
                    code=ErrorCode.CONFIG_MISSING,
                    operator=self.name,
                    config=config,
                )
            delta = timedelta(
                days=float(d.get("days", 0)),
                hours=float(d.get("hours", 0)),
                minutes=float(d.get("minutes", 0)),
                seconds=float(d.get("seconds", 0)),
            )
        if isinstance(base, list):
            return [((_parse_datetime(b) + delta).strftime("%Y-%m-%dT%H:%M:%S")) for b in base]
        dt = _parse_datetime(base)
        result = dt + delta
        return result.strftime("%Y-%m-%dT%H:%M:%S")


@OperatorRegistry.register("time_subtract")
class TimeSubtractOperator(BaseOperator):
    """
    时间减：结束时间 - 开始时间，返回间隔秒数（浮点）
    或：基准时间 - 时间间隔，返回新时间字符串（当提供 duration_seconds 时）
    config: end_time, start_time 或 base_time, duration_seconds
    """
    name = "time_subtract"
    config_schema = {
        "type": "object",
        "properties": {
            "end_time": {},
            "start_time": {},
            "base_time": {},
            "duration_seconds": {},
            "first_value": {},
            "second_value": {},
            "third_value": {},
            "fourth_value": {},
        },
    }
    default_config = {}

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("end_time") in (None, ""):
            for k in ("first_value", "value", "input", "primary"):
                v = c.get(k)
                if v not in (None, ""):
                    c["end_time"] = v
                    break
        if c.get("start_time") in (None, ""):
            for k in ("second_value", "secondary"):
                v = c.get(k)
                if v not in (None, ""):
                    c["start_time"] = v
                    break
        if c.get("base_time") in (None, ""):
            for k in ("third_value",):
                v = c.get(k)
                if v not in (None, ""):
                    c["base_time"] = v
                    break
        if c.get("duration_seconds") in (None, ""):
            for k in ("fourth_value",):
                v = c.get(k)
                if v not in (None, ""):
                    c["duration_seconds"] = v
                    break
        return c

    def execute(self, data, config, context: ExecutionContext):
        end = _get_time_ref(data, config.get("end_time") or config.get("first_value"), context)
        start = _get_time_ref(data, config.get("start_time") or config.get("second_value"), context)
        if end is not None and start is not None:
            if isinstance(end, list) or isinstance(start, list):
                end_list = end if isinstance(end, list) else [end]
                start_list = start if isinstance(start, list) else [start]
                if len(end_list) == 0 or len(start_list) == 0:
                    raise OperatorException(
                        "时间列表为空，无法相减",
                        code=ErrorCode.DATA_NOT_FOUND,
                        operator=self.name,
                        config=config,
                    )
                if len(end_list) == 1 and len(start_list) > 1:
                    end_list = end_list * len(start_list)
                elif len(start_list) == 1 and len(end_list) > 1:
                    start_list = start_list * len(end_list)
                elif len(end_list) != len(start_list):
                    raise OperatorException(
                        "time_subtract 列表长度不一致，无法按位相减",
                        code=ErrorCode.CALC_LOGIC_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out = []
                for e, s in zip(end_list, start_list):
                    dt_end = _parse_datetime(e)
                    dt_start = _parse_datetime(s)
                    out.append((dt_end - dt_start).total_seconds())
                return out
            dt_end = _parse_datetime(end)
            dt_start = _parse_datetime(start)
            return (dt_end - dt_start).total_seconds()
        base = _get_time_ref(data, config.get("base_time") or config.get("third_value"), context)
        ds = config.get("duration_seconds")
        if base is not None and ds is not None:
            ds_n = _resolve_numeric_config(data, ds, context)
            if ds_n is None:
                raise OperatorException(
                    "fourth_value（要减去的秒数）无效或未解析",
                    code=ErrorCode.CONFIG_MISSING,
                    operator=self.name,
                    config=config,
                )
            dt = _parse_datetime(base)
            delta = timedelta(seconds=ds_n)
            result = dt - delta
            return result.strftime("%Y-%m-%dT%H:%M:%S")
        raise OperatorException(
            "请提供 (first_value, second_value) 求间隔秒数，或 (third_value, fourth_value) 回推时间",
            code=ErrorCode.CONFIG_MISSING,
            operator=self.name,
            config=config,
        )


@OperatorRegistry.register("average_time")
class AverageTimeOperator(BaseOperator):
    """
    平均时间：对一组时间取平均，返回 ISO 时间字符串
    config: field（存时间列表的字段名或 "${step_key}"）
    """
    name = "average_time"
    config_schema = {"type": "object", "properties": {"field": {}, "source": {}, "first_value": {}}}
    default_config = {}

    def _resolve_config(self, config):
        c = super()._resolve_config(config)
        if c.get("field") in (None, "") and c.get("source") in (None, ""):
            for k in ("first_value", "value", "input", "primary"):
                v = c.get(k)
                if v not in (None, ""):
                    c["field"] = v
                    break
        return c

    def execute(self, data, config, context: ExecutionContext):
        ref = config.get("field") or config.get("source") or config.get("first_value")
        raw = _get_time_ref(data, ref, context)
        if raw is None:
            raise OperatorException(
                "first_value（时间序列）不能为空或未解析",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        if not isinstance(raw, list):
            raw = [raw]
        if not raw:
            raise OperatorException("时间列表为空", code=ErrorCode.DATA_NOT_FOUND, operator=self.name, config=config)
        timestamps = [_to_timestamp(_parse_datetime(t)) for t in raw]
        avg_ts = sum(timestamps) / len(timestamps)
        result = _from_timestamp(avg_ts)
        return result.strftime("%Y-%m-%dT%H:%M:%S")
