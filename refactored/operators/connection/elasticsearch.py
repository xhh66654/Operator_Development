"""Elasticsearch：仅连接，客户端写入上下文供提取算子复用。
"""
import os
from typing import Any, Dict, List, Optional

from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from ...core.exceptions import OperatorException, ErrorCode

from .persistent_registry import reuse_or_open

ES_CLIENT_CONTEXT_KEY = "_es_client"

def _load_service_es_defaults() -> Dict[str, Any]:
    """
    服务侧 ES 默认配置：
    - 环境变量优先
    - 若环境变量未配置，则回退到 refactored.app_config
    """
    env_hosts = os.environ.get("ES_HOSTS", os.environ.get("ELASTICSEARCH_URL", "")).strip()
    hosts: List[str] = []
    if env_hosts:
        hosts = [h.strip() for h in env_hosts.replace(",", " ").split() if h.strip()]
    if not hosts:
        try:
            from refactored import app_config  # 延迟 import，避免循环依赖

            cfg_hosts = getattr(app_config, "ES_HOSTS", None)
            if isinstance(cfg_hosts, str):
                hosts = [h.strip() for h in cfg_hosts.replace(",", " ").split() if h.strip()]
            elif isinstance(cfg_hosts, list):
                hosts = [str(h).strip() for h in cfg_hosts if str(h).strip()]
        except Exception:
            hosts = []

    def _env_or_cfg(env_key: str, cfg_key: str) -> str:
        v = str(os.environ.get(env_key, "")).strip()
        if v:
            return v
        try:
            from refactored import app_config

            cv = getattr(app_config, cfg_key, "")  # type: ignore[attr-defined]
            return "" if cv is None else str(cv).strip()
        except Exception:
            return ""

    timeout_raw = _env_or_cfg("ES_TIMEOUT", "ES_TIMEOUT") or "30"
    try:
        timeout = int(timeout_raw)
    except Exception:
        timeout = 30

    return {
        "hosts": hosts,
        "es_user": _env_or_cfg("ES_USER", "ES_USER"),
        "es_password": _env_or_cfg("ES_PASSWORD", "ES_PASSWORD"),
        "api_key": _env_or_cfg("ES_API_KEY", "ES_API_KEY"),
        "cloud_id": _env_or_cfg("ES_CLOUD_ID", "ES_CLOUD_ID"),
        "timeout": timeout,
    }


def _get_es_client(
    hosts: Optional[List[str]] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    cloud_id: Optional[str] = None,
    request_timeout: int = 30,
) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        raise OperatorException(
            "连接 ES 需要安装 elasticsearch 库，请执行: pip install elasticsearch",
            code=ErrorCode.RUNTIME_ERROR,
        )

    hosts = hosts or []
    if isinstance(hosts, str):
        hosts = [h.strip() for h in hosts.split(",") if h.strip()]
    if not hosts:
        env_hosts = os.environ.get("ES_HOSTS", os.environ.get("ELASTICSEARCH_URL", ""))
        if env_hosts:
            hosts = [h.strip() for h in env_hosts.replace(",", " ").split() if h.strip()]
    if not hosts:
        raise OperatorException(
            "ES 基础地址不能为空。请在 config 中提供 es_hosts（或列表形式的 hosts），或设置环境变量 ES_HOSTS",
            code=ErrorCode.CONFIG_MISSING,
        )

    kwargs: Dict[str, Any] = {"hosts": hosts, "request_timeout": int(request_timeout)}
    missing: List[str] = []

    if cloud_id:
        kwargs["cloud_id"] = cloud_id
    if api_key:
        kwargs["api_key"] = api_key
    elif user or password:
        if not user:
            missing.append("es_user（或 api_key）")
        if not password:
            missing.append("es_password")
        if missing:
            raise OperatorException(
                f"当前 ES 连接需要认证，缺少: {', '.join(missing)}",
                code=ErrorCode.CONFIG_MISSING,
            )
        kwargs["basic_auth"] = (user, password)
    else:
        env_user = os.environ.get("ES_USER", "")
        env_password = os.environ.get("ES_PASSWORD", "")
        if env_user or env_password:
            kwargs["basic_auth"] = (env_user or "", env_password or "")

    try:
        client = Elasticsearch(**kwargs)
        if not client.ping():
            raise OperatorException(
                f"无法连接到 ES 集群，请检查 es_hosts 与网络: {hosts}",
                code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            )
        return client
    except OperatorException:
        raise
    except Exception as e:
        raise OperatorException(
            f"ES 连接失败: {e}. 若需认证请提供 es_user/es_password 或 api_key；若需 TLS 请在环境或 ES 端配置。",
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            cause=e,
        )


def ensure_service_es_client(context: ExecutionContext) -> Any:
    """
    无 ``es_connect`` 步骤时，使用环境变量直连 ES（``ES_HOSTS`` / ``ELASTICSEARCH_URL`` 等），
    并与 ``persistent_registry`` 复用连接。
    """
    existing = context.get(ES_CLIENT_CONTEXT_KEY)
    if existing is not None:
        return existing

    cfg = _load_service_es_defaults()
    hosts = cfg.get("hosts") or []
    if not hosts:
        raise OperatorException(
            "ES 未配置。请在 refactored/app_config.py 配置 ES_HOSTS，或设置环境变量 ES_HOSTS / ELASTICSEARCH_URL。",
            code=ErrorCode.CONFIG_MISSING,
        )
    es_user = cfg.get("es_user") or ""
    es_password = cfg.get("es_password") or ""
    api_key = cfg.get("api_key") or ""
    cloud_id = cfg.get("cloud_id") or ""
    timeout = int(cfg.get("timeout") or 30)
    norm: Dict[str, Any] = {
        "es_hosts": list(hosts),
        "es_user": es_user,
        "es_password": es_password,
        "api_key": api_key,
        "cloud_id": cloud_id,
        "timeout": timeout,
    }

    def _open() -> Any:
        return _get_es_client(
            hosts=list(hosts),
            user=es_user or None,
            password=es_password or None,
            api_key=api_key or None,
            cloud_id=cloud_id or None,
            request_timeout=timeout,
        )

    client = reuse_or_open("elasticsearch", norm, _open)
    context.set(ES_CLIENT_CONTEXT_KEY, client)
    return client


def env_has_es_service_config() -> bool:
    """是否可通过环境变量做服务侧直连（不配连接算子）。"""
    cfg = _load_service_es_defaults()
    return bool(cfg.get("hosts"))


def _walk_steps_for_es_extract(steps: Any) -> bool:
    if not isinstance(steps, list):
        return False
    for s in steps:
        if not isinstance(s, dict):
            continue
        op = s.get("operator") or s.get("operator_key")
        if op == "es_extract":
            return True
        if _walk_steps_for_es_extract(s.get("steps")):
            return True
    return False


def _walk_metric_node_for_es_extract(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    op = node.get("operator_key") or node.get("operator")
    if op == "es_extract":
        return True
    ch = node.get("steps")
    if _walk_steps_for_es_extract(ch if isinstance(ch, list) else []):
        return True
    return False


def _reasoning_entry_roots(rd: Dict[str, Any]) -> List[Any]:
    """与 tree_calculation 一致：优先 ``steps``，否则 ``types``。"""
    if isinstance(rd.get("steps"), list):
        return list(rd["steps"])
    return list(rd.get("types") or [])


def _walk_reasoning_tree_for_es_extract(rdl: Any) -> bool:
    if not isinstance(rdl, list):
        return False
    for rd in rdl:
        if not isinstance(rd, dict):
            continue
        for t in _reasoning_entry_roots(rd):
            if _walk_metric_node_for_es_extract(t):
                return True
    return False


def request_uses_es_extract(request_data: Dict[str, Any]) -> bool:
    """DAG 推理树请求中是否出现 es_extract 节点。"""
    if not isinstance(request_data, dict):
        return False
    rdl = request_data.get("reasoningDataList")
    if isinstance(rdl, list) and len(rdl) > 0:
        return _walk_reasoning_tree_for_es_extract(rdl)
    return False


def warm_es_client_for_calculation(context: ExecutionContext, request_data: Dict[str, Any]) -> None:
    """
    单次计算开始时预热 ES：环境已配置且请求将执行 ``es_extract`` 时，
    立即建立客户端并挂在上下文中，与同次计算内后续步骤共用，直到本次计算结束释放。
    """
    if not env_has_es_service_config():
        return
    if not request_uses_es_extract(request_data):
        return
    ensure_service_es_client(context)


@OperatorRegistry.register("es_connect")
class ESConnectOperator(BaseOperator):
    """
    仅连接 ES，不查数据。连接会保持在上下文中，供后续「从ES提取」步骤复用。
    config：es_hosts（必填）, es_user, es_password, api_key, cloud_id 可选。
    """
    name = "es_connect"
    config_schema = {
        "type": "object",
        "properties": {
            "es_hosts": {},
            "es_user": {"type": "string"},
            "es_password": {"type": "string"},
            "api_key": {"type": "string"},
            "cloud_id": {"type": "string"},
        },
    }

    def execute(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        es_hosts = config.get("es_hosts")
        if isinstance(es_hosts, str):
            es_hosts = [h.strip() for h in es_hosts.split(",") if h.strip()]
        elif isinstance(es_hosts, list):
            es_hosts = [str(h).strip() for h in es_hosts if str(h).strip()]
        else:
            es_hosts = []
        # 与 _get_es_client 一致：config 未填时可退回环境变量 ES_HOSTS / ELASTICSEARCH_URL
        if not es_hosts:
            env_hosts = os.environ.get("ES_HOSTS", os.environ.get("ELASTICSEARCH_URL", ""))
            if env_hosts:
                es_hosts = [h.strip() for h in env_hosts.replace(",", " ").split() if h.strip()]
        if not es_hosts:
            raise OperatorException(
                "连接 ES 缺少 es_hosts（连接地址），或请设置环境变量 ES_HOSTS / ELASTICSEARCH_URL",
                code=ErrorCode.CONFIG_MISSING,
                operator=self.name,
                config=config,
            )
        es_user = config.get("es_user")
        es_password = config.get("es_password")
        api_key = config.get("api_key")
        cloud_id = config.get("cloud_id")
        norm = {
            "es_hosts": list(es_hosts),
            "es_user": es_user or "",
            "es_password": es_password or "",
            "api_key": api_key or "",
            "cloud_id": cloud_id or "",
        }

        def _open() -> Any:
            return _get_es_client(
                hosts=list(es_hosts),
                user=es_user,
                password=es_password,
                api_key=api_key,
                cloud_id=cloud_id,
                request_timeout=int(os.environ.get("ES_TIMEOUT", "30")),
            )

        client = reuse_or_open("elasticsearch", norm, _open)
        context.set(ES_CLIENT_CONTEXT_KEY, client)
        return True
