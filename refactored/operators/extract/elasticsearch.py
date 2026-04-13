"""Elasticsearch：从已连接客户端查询并提取。"""
from typing import Any, Dict, List, Optional

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode
from .._common import normalize_config_input, rows_to_field_list_dict
from ..connection.elasticsearch import ES_CLIENT_CONTEXT_KEY, ensure_service_es_client

ES_EXTRACT_CACHE_CONTEXT_KEY = "_es_extract_cache"


def _normalize_table_fields(table: Any) -> Optional[List[str]]:
    """规范化 table（字段名）：None 表示读取全部字段。"""
    if table is None:
        return None
    if isinstance(table, str):
        fields = [f.strip() for f in table.split(",") if f.strip()]
        return fields or None
    if isinstance(table, list):
        fields = [str(f).strip() for f in table if str(f).strip()]
        return fields or None
    return None


@OperatorRegistry.register("es_extract")
class ESExtractOperator(BaseOperator):
    """从 ES 提取数据。须先 es_connect。

    默认行为：不传 `size` 时，使用 scroll 方式把本次查询命中的全部文档取回（不做静默截断）。
    若显式传 `size`，则按 `size` 作为单次 search 的命中上限。
    """
    name = "es_extract"
    config_schema = {
        "type": "object",
        "properties": {
            "index": {"type": "string"},
            "table": {"type": ["string", "array"], "items": {"type": "string"}},
            "query": {"type": "object"},
            "size": {"type": "integer"},
            "page_size": {"type": "integer"},
            "scroll_timeout": {"type": "string"},
            "max_hits": {"type": "integer"},
        },
    }
    default_config: Dict[str, Any] = {}
    input_spec = {"type": "table"}
    output_spec = {"type": "table"}

    def _resolve_config(self, config):
        c = normalize_config_input(super()._resolve_config(config))
        # 提取类算子使用具名参数（index/table/query），同时支持顺序槽位：
        # first_value -> index, second_value -> table(字段列表), third_value -> query
        if c.get("index") in (None, "") and c.get("first_value") not in (None, ""):
            c["index"] = c.get("first_value")
            c["first_value"] = None
        if c.get("table") in (None, "") and c.get("second_value") not in (None, ""):
            c["table"] = c.get("second_value")
            c["second_value"] = None
        if c.get("query") in (None, "") and isinstance(c.get("third_value"), dict):
            c["query"] = c.get("third_value")
            c["third_value"] = None
        return c

    def execute(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        index = config.get("index")
        source_fields = _normalize_table_fields(config.get("table"))
        query = config.get("query") or {"match_all": {}}
        # 根据当前推理分支 reasoningId（存于 context 的 _active_reasoning_id）
        # 强制过滤 ES 文档字段 sRID，只取属于该 reasoningId 的全部记录。
        active_rid = context.get("_active_reasoning_id")
        if active_rid is not None and active_rid != "":
            try:
                # reasoningId 为数字时，尽量以 int 参与 term 精确匹配（避免类型不一致导致不命中）。
                active_rid_term: Any = int(active_rid)
            except (TypeError, ValueError):
                active_rid_term = active_rid
            query = {
                "bool": {
                    "must": [query],
                    "filter": [{"term": {"sRID": active_rid_term}}],
                }
            }
        if not index:
            raise OperatorException(
                "从ES提取缺少必需参数: index（索引名）",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        client = context.get(ES_CLIENT_CONTEXT_KEY)
        if client is None:
            client = ensure_service_es_client(context)
        try:
            explicit_size = config.get("size", None)
            if explicit_size is not None:
                size = int(explicit_size)
                search_kwargs = {"index": index, "query": query, "size": size}
                if source_fields is not None:
                    search_kwargs["_source"] = source_fields
                resp = client.search(**search_kwargs)
                hits = resp.get("hits", {}).get("hits", []) or []
                rows = [hit.get("_source", {}) for hit in hits]
                column_bundle = rows_to_field_list_dict(rows)
                context.set(ES_EXTRACT_CACHE_CONTEXT_KEY, column_bundle)
                context.set(context.LATEST_COLUMNS_KEY, column_bundle[0] if column_bundle else {})
                return column_bundle

            page_size = int(config.get("page_size", 1000))
            if page_size <= 0:
                page_size = 1000
            scroll_timeout = config.get("scroll_timeout", "2m")
            max_hits = config.get("max_hits")
            max_hits_n = int(max_hits) if max_hits is not None and max_hits != "" else None

            hits_sources: list = []
            search_kwargs = {
                "index": index,
                "query": query,
                "size": page_size,
                "scroll": scroll_timeout,
            }
            if source_fields is not None:
                search_kwargs["_source"] = source_fields
            resp = client.search(**search_kwargs)
            scroll_id = resp.get("_scroll_id") or resp.get("scroll_id")
            hits = (resp.get("hits", {}) or {}).get("hits", []) or []
            hits_sources.extend([hit.get("_source", {}) for hit in hits])
            total = len(hits_sources)

            if total > 0 and not scroll_id:
                raise OperatorException(
                    "ES scroll 未返回 scroll_id，无法保证全量取回；请显式传入 config.size。",
                    code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                    operator=self.name,
                    config=config,
                )

            try:
                while hits:
                    if max_hits_n is not None and total >= max_hits_n:
                        raise OperatorException(
                            f"ES 提取命中数超过 max_hits：{total} >= {max_hits_n}；已为避免 OOM/超时中断。"
                            f"请增大 max_hits 或调整查询条件。",
                            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
                            operator=self.name,
                            config=config,
                        )
                    resp = client.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
                    hits = (resp.get("hits", {}) or {}).get("hits", []) or []
                    if not hits:
                        break
                    hits_sources.extend([hit.get("_source", {}) for hit in hits])
                    total = len(hits_sources)
            finally:
                if scroll_id:
                    try:
                        client.clear_scroll(scroll_id=scroll_id)
                    except Exception:
                        pass

            column_bundle = rows_to_field_list_dict(hits_sources)
            context.set(ES_EXTRACT_CACHE_CONTEXT_KEY, column_bundle)
            context.set(context.LATEST_COLUMNS_KEY, column_bundle[0] if column_bundle else {})
            return column_bundle
        except Exception as e:
            raise OperatorException(
                f"从ES提取失败: {e}",
                code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                operator=self.name,
                config=config,
                cause=e,
            )

