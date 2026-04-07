"""
ES连接配置
"""

from __future__ import annotations

from typing import List, Optional


# Elasticsearch
# 例: ["http://127.0.0.1:9200"]
ES_HOSTS: List[str] = ["http://127.0.0.1:9200"]
ES_USER: Optional[str] = None
ES_PASSWORD: Optional[str] = None
ES_API_KEY: Optional[str] = None
ES_CLOUD_ID: Optional[str] = None
ES_TIMEOUT: int = 30

