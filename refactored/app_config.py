"""
ES连接配置
"""

from __future__ import annotations

from typing import List, Optional


# Elasticsearch连接地址
# 例: ["http://127.0.0.1:9200"]
# ES_HOSTS: List[str] = ["http://120.48.140.1:9200"]
ES_HOSTS: List[str] = ["http://127.0.0.1:9200"]
ES_USER: Optional[str] = None
ES_PASSWORD: Optional[str] = None
ES_API_KEY: Optional[str] = None
ES_CLOUD_ID: Optional[str] = None
ES_TIMEOUT: int = 200

# 回调地址默认值（服务内置兜底；若环境变量 RESULT_CALLBACK_URL/STATUS_CALLBACK_URL 有配置则优先用环境变量）
# Java 侧通常监听该地址接收 Python 计算结果。
RESULT_CALLBACK_URL_DEFAULT: str = "http://127.0.0.1:8091/api/callback/result"
# RESULT_CALLBACK_URL_DEFAULT: str = "http://10.60.184.245:8091/api/callback/result"


#接口功能分析：
"""
1.http://127.0.0.1:9200：es数据库链接地址，可能会有用户名，密码等验证信息，填写在上方es配置里面

2.http://127.0.0.1:8091/api/callback/result：结果回调接口，在python端执行计算产生的结果，通过此接口回传，包括正确结果，错误结果，
    python调java网络波动情况报错等
      
3.http://127.0.0.1:18080/calculate:接收java数据回传给java正确接收，报错情况如下：空body，非json格式，不完整json，合法 JSON 但业务校验失败，
    缺少某些重要id，网关 413/截断等（请求体太大，主动截断），请求正常，入队正常

"""
