"""
基础接口配置
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional
from urllib.parse import urlsplit


# Elasticsearch

ES_HOSTS: List[str] = ["http://127.0.0.1:9200"]
ES_USER: Optional[str] = None
ES_PASSWORD: Optional[str] = None
ES_API_KEY: Optional[str] = None
ES_CLOUD_ID: Optional[str] = None
ES_TIMEOUT: int = 200



# 1. Java → Python：体系计算 POST 目标
INTERFACE_URL_PYTHON_SYSTEM: str = (
    os.environ.get("INTERFACE_URL_PYTHON_SYSTEM", "").strip()
    or "http://127.0.0.1:19080/system/result"
)
# 2. Java → Python：结果修改计算 POST 目标
INTERFACE_URL_PYTHON_EVALUATION: str = (
    os.environ.get("INTERFACE_URL_PYTHON_EVALUATION", "").strip()
    or "http://127.0.0.1:19080/evaluation/result"
)
# 3. Python → Java：体系计算结果 POST（与 INTERFACE_URL_PYTHON_SYSTEM 一一对应）
INTERFACE_URL_JAVA_SYSTEM_RESULT: str = (
    os.environ.get("INTERFACE_URL_JAVA_SYSTEM_RESULT", "").strip()
    or os.environ.get("JAVA_RESULT_CALLBACK_URL_SYSTEM", "").strip()
    or "http://127.0.0.1:8091/system/result"
)
# 4. Python → Java：结果修改计算结果 POST（与 INTERFACE_URL_PYTHON_EVALUATION 一一对应）
INTERFACE_URL_JAVA_EVALUATION_RESULT: str = (
    os.environ.get("INTERFACE_URL_JAVA_EVALUATION_RESULT", "").strip()
    or os.environ.get("JAVA_RESULT_CALLBACK_URL_EVALUATION", "").strip()
    or "http://127.0.0.1:8091/evaluation/result"
)


def _path_from_url(url: str, fallback: str) -> str:
    u = (url or "").strip()
    if not u:
        return fallback
    path = urlsplit(u).path or fallback
    return path if path.startswith("/") else fallback


# Flask 入站路径（由上面两条 Python 侧 URL 解析，无需再手写一遍路径）
API_RECEIVE_PATH_SYSTEM_RESULT: str = _path_from_url(
    INTERFACE_URL_PYTHON_SYSTEM,
    "/system/result",
)
API_RECEIVE_PATH_EVALUATION_RESULT: str = _path_from_url(
    INTERFACE_URL_PYTHON_EVALUATION,
    "/evaluation/result",
)

# Python → Java 结果回调（与上两条 INTERFACE_URL_JAVA_* 一一对应）
JAVA_RESULT_CALLBACK_URL_SYSTEM: str = INTERFACE_URL_JAVA_SYSTEM_RESULT
JAVA_RESULT_CALLBACK_URL_EVALUATION: str = INTERFACE_URL_JAVA_EVALUATION_RESULT


def java_status_callback_url_for(compute_kind: Literal["system", "evaluation"]) -> str:
    """
    状态回调地址：与 Java 入站成对（体系 / 结果修改），由对应 **结果** 回调 URL 派生，
    将路径末尾 ``result`` 替换为 ``status``（例如 ``…/system/result`` → ``…/system/status``）。

    若结果 URL 不以 ``result`` 结尾则返回空字符串（跳过状态 POST）。
    """
    url = (
        JAVA_RESULT_CALLBACK_URL_SYSTEM.strip()
        if compute_kind == "system"
        else JAVA_RESULT_CALLBACK_URL_EVALUATION.strip()
    )
    if not url:
        return ""
    su = url.rstrip("/")
    if su.endswith("result"):
        return su[: -len("result")] + "status"
    return ""
