"""算子自动注册（装饰器）"""
from typing import Dict, Type

from .base_operator import BaseOperator


class OperatorRegistry:
    _operators: Dict[str, "BaseOperator"] = {}

    @classmethod
    def register(cls, name: str = None):
        """装饰器：自动注册算子"""

        def decorator(op_class: Type[BaseOperator]):
            key = name or getattr(op_class, "name", op_class.__name__)
            if not issubclass(op_class, BaseOperator):
                raise TypeError(f"{op_class} must inherit BaseOperator")
            cls._operators[key] = op_class()
            return op_class

        return decorator

    @classmethod
    def get(cls, name: str) -> BaseOperator:
        if name not in cls._operators:
            raise KeyError(f"未知算子: {name}")
        return cls._operators[name]

    @classmethod
    def all_names(cls) -> list:
        return list(cls._operators.keys())
