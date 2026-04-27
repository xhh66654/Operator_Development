"""Strong typed data model for operator IO."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataValue:
    type: str
    schema: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("DataValue.type must be non-empty string")
        if not isinstance(self.schema, dict):
            raise ValueError("DataValue.schema must be dict")

    def to_dict(self) -> Any:
        raise NotImplementedError

    def to_python(self) -> Any:
        return self.to_dict()

    @staticmethod
    def from_python(obj: Any) -> "DataValue":
        if isinstance(obj, DataValue):
            obj.validate()
            return obj
        if obj is None:
            return ScalarValue(value=None, dtype="null")
        if isinstance(obj, bool):
            return ScalarValue(value=obj, dtype="bool")
        if isinstance(obj, int):
            return ScalarValue(value=obj, dtype="int")
        if isinstance(obj, float):
            return ScalarValue(value=obj, dtype="float")
        if isinstance(obj, str):
            return ScalarValue(value=obj, dtype="string")
        if isinstance(obj, list):
            if len(obj) > 0 and all(isinstance(x, dict) for x in obj):
                return TableValue.from_rows(obj)
            return ListValue.from_items(obj)
        if isinstance(obj, dict):
            # Keep object as scalar-like object payload.
            return ScalarValue(value=obj, dtype="object")
        return ScalarValue(value=obj, dtype="object")


@dataclass
class ScalarValue(DataValue):
    value: Any = None
    dtype: str = "object"

    def __init__(self, value: Any, dtype: str):
        self.value = value
        self.dtype = dtype
        super().__init__(type="scalar", schema={"dtype": dtype, "format": "scalar"})
        self.validate()

    def to_dict(self) -> Any:
        return self.value


@dataclass
class ListValue(DataValue):
    items: List[Any] = field(default_factory=list)
    dtype: str = "mixed"

    @staticmethod
    def _infer_dtype(items: List[Any]) -> str:
        if not items:
            return "unknown"
        kinds = {type(x).__name__ for x in items}
        return kinds.pop() if len(kinds) == 1 else "mixed"

    @classmethod
    def from_items(cls, items: List[Any]) -> "ListValue":
        dtype = cls._infer_dtype(items)
        return cls(items=items, dtype=dtype)

    def __init__(self, items: List[Any], dtype: str):
        self.items = items
        self.dtype = dtype
        super().__init__(type="list", schema={"dtype": dtype, "length": len(items), "format": "list"})
        self.validate()

    def validate(self) -> None:
        super().validate()
        if self.dtype != "mixed":
            for it in self.items:
                if type(it).__name__ != self.dtype:
                    raise ValueError("ListValue items must share same type")

    def to_dict(self) -> Any:
        return self.items


@dataclass
class TableValue(DataValue):
    columns: List[Dict[str, str]] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _dtype(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "string"
        if isinstance(v, list):
            return "list"
        if isinstance(v, dict):
            return "object"
        return "object"

    @staticmethod
    def _numeric_column_types_compatible(actual_dtype: str, cell: Any, column_dtype: str) -> bool:
        """
        列类型由首行推断后，允许数值列 int/float 互通（常见于 ES 等来源：整数字面量为 int、带小数或
        双精度为 float），避免误报 ``expects float`` / ``expects int``。
        - 列声明为 float：允许后续行为 int。
        - 列声明为 int：仅当 cell 为整数值的 float（如 200.0）时允许。
        """
        if actual_dtype == column_dtype:
            return True
        if column_dtype == "float" and actual_dtype == "int":
            return True
        if column_dtype == "int" and actual_dtype == "float":
            return (
                isinstance(cell, float)
                and math.isfinite(cell)
                and cell.is_integer()
            )
        return False

    @classmethod
    def from_rows(cls, rows: List[Dict[str, Any]]) -> "TableValue":
        if not rows:
            return cls(columns=[], rows=[])
        first = rows[0]
        cols = [{"name": k, "type": cls._dtype(v)} for k, v in first.items()]
        return cls(columns=cols, rows=rows)

    def __init__(self, columns: List[Dict[str, str]], rows: List[Dict[str, Any]]):
        self.columns = columns
        self.rows = rows
        super().__init__(
            type="table",
            schema={
                "columns": columns,
                "row_count": len(rows),
                "format": "rows",
            },
        )
        self.validate()

    def validate(self) -> None:
        super().validate()
        names = [c.get("name") for c in self.columns]
        name_set = set(names)
        for r in self.rows:
            if set(r.keys()) != name_set:
                raise ValueError("TableValue row fields must match columns")
            for c in self.columns:
                n = c["name"]
                t = c["type"]
                if t == "null":
                    continue
                v = r.get(n)
                if v is None:
                    continue
                actual = self._dtype(v)
                if actual != t and not self._numeric_column_types_compatible(actual, v, t):
                    raise ValueError(f"TableValue column type mismatch: {n} expects {t}")

    def to_dict(self) -> Any:
        return self.rows

