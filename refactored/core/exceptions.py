"""统一异常类与错误码体系"""
from enum import IntEnum
from typing import Any, Optional


class ErrorCode(IntEnum):
    """统一错误码：便于调用方区分异常类型"""
    # 1xxx 数据/配置
    DATA_NOT_FOUND = 1001
    CONFIG_MISSING = 1002
    CONFIG_TYPE_ERROR = 1003  # JSON Schema / 类型校验失败（如类型、required、additionalProperties）
    CONFIG_FORMAT_ERROR = 1008  # 非法旧输入键
    CONFIG_INVALID = 1004
    DEPENDENCY_ERROR = 1005
    VERSION_UNSUPPORTED = 1006
    DUPLICATE_RUN_ID = 1007
    # 2xxx 类型/格式
    TYPE_ERROR = 2001
    FORMAT_ERROR = 2002
    SCHEMA_MISMATCH = 2003
    # 3xxx 计算逻辑
    CALC_LOGIC_ERROR = 3001
    OUT_OF_RANGE = 3002
    # 4xxx 资源/性能
    OOM = 4001
    TIMEOUT = 4002
    RESOURCE_LIMIT_EXCEEDED = 4003
    # 5xxx 运行时/未知
    RUNTIME_ERROR = 5001
    EXTERNAL_SERVICE_ERROR = 5002  # 如 ES 连接/查询失败
    UNKNOWN = 5999


# 调用方依赖「JSON Schema 失败」与「非法旧输入键」可用不同数值区分；若合并为同值须同步改契约与全部测试。
if int(ErrorCode.CONFIG_TYPE_ERROR) == int(ErrorCode.CONFIG_FORMAT_ERROR):
    raise RuntimeError(
        "ErrorCode.CONFIG_TYPE_ERROR and CONFIG_FORMAT_ERROR must not share the same integer"
    )


class OperatorException(Exception):
    """算子统一异常"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        operator: Optional[str] = None,
        config: Optional[dict] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.code = code
        self.operator = operator
        self.config = config
        self.cause = cause
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "success": False,
            "error": self.message,
            "error_code": int(self.code),
            "operator": self.operator,
            "config": self.config,
        }
