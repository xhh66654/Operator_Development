"""上下文生命周期、快照/回滚、复杂表达式解析。"""
import json
import threading
from typing import Any, Dict, List, Set


class ExecutionContext:
    """上下文：生命周期清晰，支持快照/回滚与复杂表达式"""

    PRECISION_KEY = "_decimal_places"
    LATEST_COLUMNS_KEY = "_latest_columns"
    LATEST_STEP_KEY = "_latest_extract_step_key"
    NON_SNAPSHOT_KEYS = {"_es_client", LATEST_COLUMNS_KEY, LATEST_STEP_KEY}
    #数据缓存最大占有量，超过自动截至
    MAX_MEMORY_BYTES = 50 * 1024 * 1024

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._lock = threading.RLock()
        # 快照记录“步骤前已存在 key 集合”，回滚时只删除新 key，避免整仓深拷贝。
        self._snapshot_stack: List[Set[str]] = []

    def _estimate_memory_bytes(self) -> int:
        try:
            payload = json.dumps(self._store, ensure_ascii=False, default=str)
            return len(payload.encode("utf-8"))
        except Exception:
            # 兜底估算，避免对象不可序列化导致中断。
            return sum(len(str(k)) + len(str(v)) for k, v in self._store.items())
    #对数据超过self.MAX_MEMORY_BYTES缓存量的数据进行终止运算报错
    # def _enforce_memory_limit(self) -> None:
    #     if self._estimate_memory_bytes() > self.MAX_MEMORY_BYTES:
    #         raise MemoryError(f"ExecutionContext memory exceeds {self.MAX_MEMORY_BYTES} bytes")

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            # self._enforce_memory_limit()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def remove(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear_unused(self, keep_keys: List[str]) -> None:
        with self._lock:
            keep = set(keep_keys) | self.NON_SNAPSHOT_KEYS
            for k in list(self._store.keys()):
                if k not in keep:
                    self._store.pop(k, None)

    def snapshot(self) -> None:
        """步骤执行前保存快照"""
        with self._lock:
            existing = {k for k in self._store.keys() if k not in self.NON_SNAPSHOT_KEYS}
            self._snapshot_stack.append(existing)

    def rollback(self) -> None:
        """步骤失败时恢复到上一快照"""
        with self._lock:
            if self._snapshot_stack:
                existed = self._snapshot_stack.pop()
                for k in list(self._store.keys()):
                    if k in self.NON_SNAPSHOT_KEYS:
                        continue
                    if k not in existed:
                        self._store.pop(k, None)

    def dispose(self) -> None:
        """单次计算结束后清空上下文，释放大对象引用，便于等待下一次计算。"""
        with self._lock:
            self._store.clear()
            self._snapshot_stack.clear()

    def resolve_expression(self, expr: str) -> Any:
        """解析复杂表达式，如 ${step1} + ${step2} * 0.8"""
        from ..utils.expr_parser import resolve_expression as _resolve
        return _resolve(expr, self._store)
