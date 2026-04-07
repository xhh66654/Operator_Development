"""
ES连接配置
"""

from __future__ import annotations

from typing import List, Optional


# Elasticsearch连接地址
# 例: ["http://127.0.0.1:9200"]
ES_HOSTS: List[str] = ["http://120.48.140.1:9200"]
# ES_HOSTS: List[str] = ["http://127.0.0.1:9200"]
ES_USER: Optional[str] = None
ES_PASSWORD: Optional[str] = None
ES_API_KEY: Optional[str] = None
ES_CLOUD_ID: Optional[str] = None
ES_TIMEOUT: int = 30

# 回调地址默认值（服务内置兜底；若环境变量 RESULT_CALLBACK_URL/STATUS_CALLBACK_URL 有配置则优先用环境变量）
# Java 侧通常监听该地址接收 Python 计算结果。
# RESULT_CALLBACK_URL_DEFAULT: str = "http://127.0.0.1:8091/api/callback/result"
RESULT_CALLBACK_URL_DEFAULT: str = "http://10.60.184.245:8091/api/callback/result"

