"""Unified error payload and error-code taxonomy."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .exceptions import ErrorCode

ERROR_CODE_TAXONOMY: Dict[int, str] = {
    int(ErrorCode.DATA_NOT_FOUND): "data_not_found",
    int(ErrorCode.CONFIG_MISSING): "config_missing",
    int(ErrorCode.CONFIG_TYPE_ERROR): "config_type_error",
    int(ErrorCode.CONFIG_FORMAT_ERROR): "config_format_error",
    int(ErrorCode.CONFIG_INVALID): "config_invalid",
    int(ErrorCode.DEPENDENCY_ERROR): "dependency_error",
    int(ErrorCode.VERSION_UNSUPPORTED): "version_unsupported",
    int(ErrorCode.DUPLICATE_RUN_ID): "duplicate_run_id",
    int(ErrorCode.TYPE_ERROR): "type_error",
    int(ErrorCode.FORMAT_ERROR): "format_error",
    int(ErrorCode.SCHEMA_MISMATCH): "schema_mismatch",
    int(ErrorCode.CALC_LOGIC_ERROR): "calc_logic_error",
    int(ErrorCode.OUT_OF_RANGE): "out_of_range",
    int(ErrorCode.OOM): "oom",
    int(ErrorCode.TIMEOUT): "timeout",
    int(ErrorCode.RESOURCE_LIMIT_EXCEEDED): "resource_limit_exceeded",
    int(ErrorCode.RUNTIME_ERROR): "runtime_error",
    int(ErrorCode.EXTERNAL_SERVICE_ERROR): "external_service_error",
    int(ErrorCode.UNKNOWN): "unknown_error",
}


def error_payload(
    *,
    indicator_name: Optional[str],
    message: str,
    error_code: int,
    step_key: Optional[str] = None,
    steps: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """精简错误响应结构。"""
    payload: Dict[str, Any] = {
        "success": False,
        "indicator_name": indicator_name,
        "error_code": int(error_code),
        "message": message,
        "result": {"type": "null", "data": None},
        "steps": steps or [],
    }
    if extra:
        payload.update(extra)
    return payload

