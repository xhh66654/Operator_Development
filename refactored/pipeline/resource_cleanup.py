"""流水线结束后释放连接等资源，避免长驻 Worker 泄漏 ES 客户端。"""
from __future__ import annotations

from typing import Any

from ..core import ExecutionContext
from ..operators.connection.persistent_registry import persist_db_connections_enabled


def release_pipeline_resources(ctx: ExecutionContext | None) -> None:
    """在回传响应前调用：尽量关闭 ES 客户端。

    若设置环境变量 ``PERSIST_DB_CONNECTIONS``（1/true/yes/on），则只从上下文中摘除 ES 客户端引用，
    不关闭，以便同一 Worker 线程上后续请求复用（见 ``persistent_registry``）。
    """
    if ctx is None:
        return
    from ..operators.connection.elasticsearch import ES_CLIENT_CONTEXT_KEY

    if persist_db_connections_enabled():
        try:
            ctx.remove(ES_CLIENT_CONTEXT_KEY)
        except Exception:
            pass
        return
    try:
        client: Any = ctx.get(ES_CLIENT_CONTEXT_KEY)
        if client is not None:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            ctx.remove(ES_CLIENT_CONTEXT_KEY)
    except Exception:
        pass
