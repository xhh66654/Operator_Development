"""Excel 文件提取。"""
from typing import Dict, List

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from .._common import normalize_config_input, rows_to_field_list_dict
from .fields import clean_nan_records


@OperatorRegistry.register("extract_from_excel")
class ExtractFromExcelOperator(BaseOperator):
    """从 Excel 读取并按列聚合为 [{列名: 取值列表}]。"""
    name = "extract_from_excel"
    config_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "sheet_name": {},
            "skip_rows": {"type": "integer"},
            "header_row": {"type": "integer"},
            "max_rows": {"type": "integer"},
            "selected_columns": {"type": "array"},
            "return_columns_only": {},
        },
    }
    default_config = {"sheet_name": 0, "skip_rows": 0, "header_row": 0}
    input_spec = {"type": "table"}
    output_spec = {"type": "table"}

    def _resolve_config(self, config):
        c = normalize_config_input(super()._resolve_config(config))
        # 兼容顺序参数：first_value -> file_path
        if c.get("file_path") in (None, "") and c.get("first_value") not in (None, ""):
            c["file_path"] = c.get("first_value")
        return c

    def execute(self, data: Dict, config: Dict, context: ExecutionContext) -> List[Dict]:
        file_path = config.get("file_path")
        if not file_path:
            raise OperatorException("file_path 不能为空", code=ErrorCode.CONFIG_MISSING, operator=self.name, config=config)
        try:
            import pandas as pd
        except ImportError:
            raise OperatorException("需要安装 pandas", code=ErrorCode.RUNTIME_ERROR)
        sheet_name = config.get("sheet_name", 0)
        skip_rows = config.get("skip_rows", 0)
        max_rows = config.get("max_rows")
        header_row = config.get("header_row", 0)
        selected_columns = config.get("selected_columns")
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows, nrows=max_rows, header=header_row)
        if selected_columns:
            available = [c for c in selected_columns if c in df.columns]
            if available:
                df = df[available]
        if config.get("return_columns_only"):
            return list(df.columns.tolist())
        return rows_to_field_list_dict(clean_nan_records(df.to_dict("records")))
