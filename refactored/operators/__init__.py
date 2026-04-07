"""
算子按业务分目录注册（import 子包即完成 OperatorRegistry 注册）：
  connection   连接
  extract      提取
  io           流式数据源等基础设施
  arithmetic   算术
  compare      比较
  cleaning     清洗
  statistics   统计（含离散/离均差平方和）
  precision    精度
  time         时间
  normalization 归一化
  correlation  相关性/距离/集合
  convert      转换
"""
from . import _common
from . import connection
from . import extract
from . import io
from . import arithmetic
from . import compare
from . import cleaning
from . import statistics
from . import precision
from . import time
from . import normalization
from . import correlation
from . import convert

__all__ = [
    "_common",
    "connection",
    "extract",
    "io",
    "arithmetic",
    "compare",
    "cleaning",
    "statistics",
    "precision",
    "time",
    "normalization",
    "correlation",
    "convert",
]
