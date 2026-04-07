"""按 runId 控制单次计算生命周期：并发去重与运行结束登记清理。

业务约定：
- 同一时刻同一 ``runId`` 只允许一个请求在执行；重复请求不参与计算。
- 无 ``runId`` 的旧式请求跳过去重，保持兼容。
- 运行结束后由调用方 ``release_run`` 释放占位；流水线内上下文由
  ``ExecutionContext.dispose()`` 另行清空。
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Set

_lock = threading.Lock()
_in_flight: Set[str] = set()
_run_resources: Dict[str, Dict[str, Any]] = {}


def normalize_run_id(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def try_acquire_run(run_id: Optional[str]) -> bool:
    """尝试占用 runId。无 runId 视为 True；已在执行中则 False。"""
    if not run_id:
        return True
    with _lock:
        if run_id in _in_flight:
            return False
        _in_flight.add(run_id)
        if run_id not in _run_resources:
            _run_resources[run_id] = {}
        return True


def release_run(run_id: Optional[str]) -> None:
    """释放 runId 占位并丢弃为该 run 登记的临时资源引用。"""
    if not run_id:
        return
    with _lock:
        _in_flight.discard(run_id)
        _run_resources.pop(run_id, None)


def register_run_resource(run_id: str, key: str, value: Any) -> None:
    """可选：将仅在本 run 内使用的缓存对象登记，便于后续扩展统一清理。"""
    if not run_id or not key:
        return
    with _lock:
        bucket = _run_resources.setdefault(run_id, {})
        bucket[key] = value
