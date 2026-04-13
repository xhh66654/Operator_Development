"""CSV 流式读取与全量读取算子。"""
import math
from typing import Any, Dict, Iterator, List, Optional

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from .._common import normalize_config_input, rows_to_field_list_dict


def _clean_nan(records: List[Dict]) -> List[Dict]:
    def _fix(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return [{k: _fix(v) for k, v in row.items()} for row in records]


def extract_from_csv_stream(
    file_path: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
    chunksize: int = 50000,
    selected_columns: Optional[List[str]] = None,
    **kwargs
) -> Iterator[List[Dict[str, Any]]]:
    """流式读取 CSV，按块 yield，避免 OOM"""
    try:
        import pandas as pd
    except ImportError:
        raise OperatorException("需要安装 pandas", code=ErrorCode.RUNTIME_ERROR)
    for chunk in pd.read_csv(
        file_path,
        encoding=encoding,
        delimiter=delimiter,
        chunksize=chunksize,
        dtype=str,
        usecols=selected_columns,
        **kwargs
    ):
        yield chunk.to_dict("records")


@OperatorRegistry.register("extract_from_csv")
class ExtractFromCsvOperator(BaseOperator):
    """从 CSV 读取（全量，小文件）；大文件请用流式 API"""
    name = "extract_from_csv"
    config_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "encoding": {"type": "string"},
            "delimiter": {"type": "string"},
            "selected_columns": {"type": "array"},
            "skip_rows": {"type": "integer"},
            "max_rows": {"type": "integer"},
            "return_columns_only": {},
            "filter_conditions": {"type": "object"},
        },
    }
    default_config = {"encoding": "utf-8", "delimiter": ",", "skip_rows": 0}
    input_spec = {"type": "table"}
    output_spec = {"type": "table"}

    def _resolve_config(self, config):
        c = normalize_config_input(super()._resolve_config(config))
        # 兼容顺序参数（文档/前端早期使用 first_value/second_value...）
        # first_value -> file_path
        # second_value -> encoding
        # third_value -> delimiter
        # fourth_value -> selected_columns
        # fifth_value -> skip_rows
        # sixth_value -> max_rows
        if c.get("file_path") in (None, "") and c.get("first_value") not in (None, ""):
            c["file_path"] = c.get("first_value")
            c["first_value"] = None
        if c.get("encoding") in (None, "") and c.get("second_value") not in (None, ""):
            c["encoding"] = c.get("second_value")
            c["second_value"] = None
        if c.get("delimiter") in (None, "") and c.get("third_value") not in (None, ""):
            c["delimiter"] = c.get("third_value")
            c["third_value"] = None
        if c.get("selected_columns") in (None, "") and c.get("fourth_value") not in (None, ""):
            c["selected_columns"] = c.get("fourth_value")
            c["fourth_value"] = None
        if c.get("skip_rows") in (None, "") and c.get("fifth_value") not in (None, ""):
            c["skip_rows"] = c.get("fifth_value")
            c["fifth_value"] = None
        if c.get("max_rows") in (None, "") and c.get("sixth_value") not in (None, ""):
            c["max_rows"] = c.get("sixth_value")
            c["sixth_value"] = None
        return c

    def execute(self, data, config, context: ExecutionContext):
        file_path = config.get("file_path")
        if not file_path:
            raise OperatorException("file_path 不能为空", code=ErrorCode.CONFIG_MISSING, operator=self.name, config=config)
        try:
            import pandas as pd
        except ImportError:
            raise OperatorException("需要安装 pandas", code=ErrorCode.RUNTIME_ERROR)
        encoding = config.get("encoding", "utf-8")
        delimiter = config.get("delimiter", ",")
        skip_rows = config.get("skip_rows", 0)
        max_rows = config.get("max_rows")
        selected_columns = config.get("selected_columns")
        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, skiprows=skip_rows, nrows=max_rows)
        if selected_columns:
            available = [c for c in selected_columns if c in df.columns]
            if available:
                df = df[available]
        if config.get("return_columns_only"):
            return list(df.columns.tolist())
        return rows_to_field_list_dict(_clean_nan(df.to_dict("records")))
