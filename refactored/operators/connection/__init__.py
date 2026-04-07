"""连接：服务侧通过环境变量/配置直连 ES；此处仅 ES 客户端与可选复用逻辑。"""
from . import elasticsearch  # noqa: F401

__all__ = ["elasticsearch"]
