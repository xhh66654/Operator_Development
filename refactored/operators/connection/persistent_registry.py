"""
可选：在**同一 OS 线程**上跨多次 HTTP 请求复用 Elasticsearch 客户端。

* 环境变量 ``PERSIST_DB_CONNECTIONS`` 为 ``1`` / ``true`` / ``yes`` / ``on`` 时启用（名称沿用历史配置）。
* 使用 **threading.local**，避免多线程 Flask worker 下多请求共用一个 ES 客户端。
* 修改连接参数会得到新的缓存键；旧客户端随线程空闲，直至进程退出或由服务端回收。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Dict

_tls = threading.local()


def persist_db_connections_enabled() -> bool:
    v = os.environ.get("PERSIST_DB_CONNECTIONS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _thread_store() -> Dict[str, Any]:
    d = getattr(_tls, "db_conns", None)
    if d is None:
        d = {}
        _tls.db_conns = d
    return d


def connection_fingerprint(kind: str, norm_config: Dict[str, Any]) -> str:
    return json.dumps(
        {"kind": kind, "config": norm_config},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _es_healthy(client: Any) -> bool:
    try:
        return bool(client.ping())
    except Exception:
        return False


def reuse_or_open(
    kind: str,
    norm_config: Dict[str, Any],
    factory: Callable[[], Any],
) -> Any:
    if not persist_db_connections_enabled():
        return factory()
    if kind != "elasticsearch":
        return factory()

    fp = connection_fingerprint(kind, norm_config)
    store = _thread_store()
    prev = store.get(fp)
    if prev is not None and _es_healthy(prev):
        return prev

    conn = factory()
    store[fp] = conn
    return conn
