"""对外入口：体系计算 / 结果修改计算（顶层 ``steps`` 协议，见 SYSTEM_EVALUATION_API_IMPLEMENTATION_SPEC.md）。"""
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, TypeVar

# 触发算子自动注册（必须先 import）
from . import operators  # noqa: F401

from .pipeline.system_protocol import execute_steps_tree_calculation

T = TypeVar("T")


def _heartbeat_interval_sec() -> float:
    """
    环境变量 CALC_HEARTBEAT_SEC：长耗时 tree 计算期间打 INFO 的间隔（秒），默认 1。
    0 或负数表示关闭。
    """
    raw = os.environ.get("CALC_HEARTBEAT_SEC", "1").strip()
    if not raw:
        return 1.0
    try:
        v = float(raw)
        return v if v > 0 else 0.0
    except ValueError:
        return 1.0


def _run_with_calc_heartbeat(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """在 fn 执行期间按间隔打 INFO 日志（同步/异步任务共用）。"""
    interval = _heartbeat_interval_sec()
    if interval <= 0:
        return fn(*args, **kwargs)

    stop = threading.Event()
    t0 = time.monotonic()

    def _tick() -> None:
        tick_idx = 0
        while not stop.wait(timeout=interval):
            tick_idx += 1
            elapsed = time.monotonic() - t0
            logging.info(
                "计算心跳 #%d：tree 仍在执行（非超时、不中断），已耗时 %.1f s；"
                "CALC_HEARTBEAT_SEC=%s，大数据可调大间隔或置 0 关闭",
                tick_idx,
                elapsed,
                interval,
            )

    th = threading.Thread(target=_tick, daemon=True, name="calc-heartbeat")
    th.start()
    try:
        return fn(*args, **kwargs)
    finally:
        stop.set()
        th.join(timeout=min(2.0, interval + 0.5))


def calculate_system(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """体系计算：``POST …/system/result`` 请求体。"""
    logging.info("体系计算开始（taskId=%s）", request_data.get("taskId"))
    return _run_with_calc_heartbeat(execute_steps_tree_calculation, request_data)


def calculate_evaluation(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    结果修改计算：``POST …/evaluation/result`` 请求体。

    执行策略细节待定前，与体系计算 **共用** ``execute_steps_tree_calculation``；
    与体系的差异由 **异步回调 URL**（8091 path）区分。
    """
    logging.info("结果修改计算开始（taskId=%s）", request_data.get("taskId"))
    return _run_with_calc_heartbeat(execute_steps_tree_calculation, request_data)


__all__ = [
    "calculate_system",
    "calculate_evaluation",
    "execute_steps_tree_calculation",
]
