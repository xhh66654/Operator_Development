"""配置前置校验：解决在运行中才能发现问题"""
import functools
from typing import Optional

from .exceptions import OperatorException, ErrorCode


def validate_operator_config(operator_name: str, config: dict, schema: Optional[dict]) -> None:
    """使用 JSON Schema 做前置校验"""
    if not schema:
        return
    try:
        import jsonschema
    except ImportError:
        return
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.ValidationError as e:
        msg = e.message if hasattr(e, "message") else str(e)
        raise OperatorException(
            f"配置校验失败: {msg}",
            code=ErrorCode.CONFIG_TYPE_ERROR,
            operator=operator_name,
            config=config,
        )


def require_config(*required_keys: str):
    """装饰器：校验必填配置项"""

    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, data, config, context):
            missing = [k for k in required_keys if config.get(k) is None or config.get(k) == ""]
            if missing:
                raise OperatorException(
                    f"缺少必填参数: {missing}",
                    code=ErrorCode.CONFIG_MISSING,
                    operator=getattr(self, "name", "?"),
                    config=config,
                )
            return f(self, data, config, context)

        return wrapper

    return decorator
