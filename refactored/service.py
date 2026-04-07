import math
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_LOG_FMT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _stderr_line(msg: str) -> None:
    """部分 IDE 运行配置下 logging 不显示时，stderr 直出仍可见。"""
    print(msg, file=sys.stderr, flush=True)


class _FlushStreamHandler(logging.StreamHandler):
    """非 TTY 环境下避免日志卡在缓冲区，计算线程里打的 INFO 也能尽快出现在控制台。"""

    def emit(self, record):
        super().emit(record)
        self.flush()


def _configure_service_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stderr_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and getattr(h, "stream", None) is sys.stderr
    ]
    if stderr_handlers:
        for h in stderr_handlers:
            if h.level == logging.NOTSET or h.level > logging.INFO:
                h.setLevel(logging.INFO)
            if not h.formatter:
                h.setFormatter(logging.Formatter(_LOG_FMT))
        return
    h = _FlushStreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter(_LOG_FMT))
    root.addHandler(h)


_configure_service_logging()


def _ensure_runtime_logging_visible() -> None:
    """
    Flask / werkzeug 加载后再次保证：
    - 根 logger 上必有指向 stderr 的 handler（业务代码里 logging.info/warning 都走 root）
    - werkzeug、flask.app 为 INFO，便于看到每条 HTTP 访问与路由内日志
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    has_stderr = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr
        for h in root.handlers
    )
    if not has_stderr:
        h = _FlushStreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(_LOG_FMT))
        root.addHandler(h)
    for name in ("werkzeug", "flask.app"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True

from flask import Flask, request, Response
from flask_cors import CORS

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from refactored.api import calculate_indicator  # noqa: E402
from refactored.core import error_payload  # noqa: E402
from refactored.core.exceptions import ErrorCode  # noqa: E402
from refactored.core.run_lifecycle import normalize_run_id, release_run, try_acquire_run  # noqa: E402
from refactored.integration.async_jobs import job_submit  # noqa: E402
from refactored.pipeline.tree_calculation import response_meta_triple  # noqa: E402

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)
_ensure_runtime_logging_visible()


@app.before_request
def _log_http_inbound():
    """
    每条进入本进程的 HTTP 都打一行（WARNING），不依赖 werkzeug 自带访问日志是否在终端显示。
    若启动后始终看不到本行，说明请求未打到本机该端口（地址/端口/Java 的 CALCULATE_URL 不对）。
    """
    logging.warning(
        "HTTP 入站 %s %s remote=%s Content-Length=%s",
        request.method,
        request.path,
        request.environ.get("REMOTE_ADDR", "?"),
        request.content_length,
    )
    _stderr_line(
        f"[refactored.service] HTTP 入站 {request.method} {request.path} "
        f"remote={request.environ.get('REMOTE_ADDR', '?')} clen={request.content_length}"
    )


class _SafeEncoder(json.JSONEncoder):
    """将 NaN / Infinity 转为 None，避免非法 JSON。"""
    def default(self, o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return super().default(o)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, o):
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: self._sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self._sanitize(v) for v in o]
        return o


def _sync_calculate_enabled() -> bool:
    v = os.environ.get("CALCULATE_SYNC", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _write_ack_if_configured(req_data: dict) -> None:
    """可选：落盘「已接收」确认，便于与 Java 侧存档对照。"""
    d = os.environ.get("CALC_ACK_SAVE_DIR", "").strip()
    if not d:
        return
    try:
        root = Path(d)
        root.mkdir(parents=True, exist_ok=True)
        tid = str(req_data.get("taskId", "na"))
        rid = str(req_data.get("runId") or req_data.get("run_id") or "na")
        path = root / f"received_{tid}_{rid}.json"
        body = {
            "status": "received",
            "receivedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "taskId": req_data.get("taskId"),
            "runId": req_data.get("runId") or req_data.get("run_id"),
            "systemId": req_data.get("systemId"),
        }
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("已写入接收确认文件: %s", path)
    except OSError as e:
        logging.warning("写入接收确认文件失败: %s", e)


def make_json_response(data, status_code: int = 200) -> Response:
    """
    手动构造 UTF-8 JSON 响应，确保中文不乱码。
    """
    json_str = json.dumps(data, ensure_ascii=False, indent=2, cls=_SafeEncoder)
    return Response(
        json_str,
        status=status_code,
        mimetype="application/json; charset=utf-8",
    )


@app.route("/calculate", methods=["POST"])
def calculate():
    """
    指标计算接口（单个请求）:

    - 仅接受 execution_mode=dag 与非空 reasoningDataList（每项含 steps 或 types）；不再接受顶层 steps
    - 请求体 JSON = refactored.api.calculate_indicator 的 request_data
    """
    try:
        req_data = request.get_json()
        if not req_data:
            return make_json_response(
                error_payload(indicator_name=None, message="请求体不能为空", error_code=1000),
                status_code=400,
            )

        # WARNING 保证在多数日志配置下也能看见（排查「Java 发了但 Python 没反应」）
        logging.warning(
            "POST /calculate 已收到 JSON taskId=%s runId=%s",
            req_data.get("taskId"),
            req_data.get("runId") or req_data.get("run_id"),
        )
        _stderr_line(
            f"[refactored.service] POST /calculate 已收到 taskId={req_data.get('taskId')} "
            f"runId={req_data.get('runId') or req_data.get('run_id')}"
        )
        logging.info("收到计算请求: %s", json.dumps(req_data, ensure_ascii=False))

        run_id = normalize_run_id(req_data.get("runId") or req_data.get("run_id"))
        tid, rid_out, sid = response_meta_triple(req_data)
        if run_id and not try_acquire_run(run_id):
            logging.warning(
                "runId 已在执行中，本次跳过计算（仍返回 success）taskId=%s runId=%s",
                tid,
                run_id,
            )
            return make_json_response(
                {
                    "taskId": tid,
                    "runId": rid_out,
                    "systemId": sid,
                    "status": "success",
                },
                status_code=200,
            )

        _write_ack_if_configured(req_data)

        if _sync_calculate_enabled():
            try:
                result = calculate_indicator(req_data)
                logging.info(
                    "同步计算完成 taskId=%s runId=%s success=%s",
                    req_data.get("taskId"),
                    run_id,
                    result.get("success"),
                )
                return make_json_response(result, status_code=200)
            finally:
                if run_id:
                    release_run(run_id)


        try:
            internal_job_id = job_submit(req_data)
        except Exception:
            # 已占用 runId 但入队失败时，必须释放，否则该 runId 永久卡死
            if run_id:
                release_run(run_id)
            raise
        logging.info(
            "已提交异步计算 internal_job_id=%s taskId=%s runId=%s",
            internal_job_id,
            req_data.get("taskId"),
            run_id,
        )
        _cb_url = os.environ.get("RESULT_CALLBACK_URL", "").strip() or (
            "http://127.0.0.1:8091/api/callback/result"
        )
        _stderr_line(
            f"[refactored.service] 已入队异步任务 job={internal_job_id}；"
            f"算完后将 POST 结果到 {_cb_url}（须另有进程监听，例如 PythonResultReceiver）"
        )
        return make_json_response(
            {
                "taskId": tid,
                "runId": rid_out,
                "systemId": sid,
                "status": "success",
            },
            status_code=200,
        )

    except Exception as e:
        logging.exception("计算过程发生未捕获异常")
        return make_json_response(
            error_payload(indicator_name=None, message=str(e), error_code=5001),
            status_code=500,
        )


@app.route("/", methods=["GET"])
def index():
    return make_json_response(
        {
            "status": "OK",
            "message": "指标计算服务正常",
            "sync": {"POST": ["/calculate"]},
        }
    )


def _listen_host() -> str:
    return os.environ.get("CALC_LISTEN_HOST", "0.0.0.0").strip() or "0.0.0.0"


def _listen_port() -> int:
    try:
        return int(os.environ.get("CALC_LISTEN_PORT", "19080"))
    except ValueError:
        return 19080


def _log_startup_banner(host: str, port: int, callback_default: str) -> None:
    """进程一启动就打出一段说明，不依赖先有 HTTP 请求（方便确认日志与配置）。"""
    sync = _sync_calculate_enabled()
    cb_eff = os.environ.get("RESULT_CALLBACK_URL", "").strip() or callback_default
    try:
        import refactored as _rf

        pkg_root = getattr(_rf, "__file__", "?")
    except Exception:
        pkg_root = "?"
    root = logging.getLogger()
    logging.info("======== refactored.service 启动自检 ========")
    logging.info("Python: %s", sys.version.split()[0])
    logging.info("解释器: %s", sys.executable)
    logging.info("工作目录: %s", os.getcwd())
    logging.info("refactored 包: %s", pkg_root)
    logging.info("监听: http://%s:%s  | 本机: http://127.0.0.1:%s/", host, port, port)
    logging.info("计算接口: POST http://127.0.0.1:%s/calculate", port)
    if sync:
        logging.info("模式: CALCULATE_SYNC=1 → 响应体为完整计算结果（一般无需 8091 回调）")
    else:
        logging.info("模式: 异步；算完后 POST 回调: %s", cb_eff)
    logging.info("请求日志: 任意 HTTP 先打「HTTP 入站 …」；POST /calculate 再打业务日志")
    logging.info(
        "根 logger: level=%s handlers=%s",
        logging.getLevelName(root.level),
        len(root.handlers),
    )
    logging.info("===========================================")
    logging.warning(
        "服务就绪: POST http://127.0.0.1:%s/calculate （Flask 将阻塞本终端，Ctrl+C 停止）",
        port,
    )


if __name__ == "__main__":
    _host, _port = _listen_host(), _listen_port()
    _cb = os.environ.get("RESULT_CALLBACK_URL", "").strip() or (
        "http://127.0.0.1:8091/api/callback/result"
    )
    _ensure_runtime_logging_visible()
    _log_startup_banner(_host, _port, _cb)
    app.run(
        host=_host,
        port=_port,
        debug=False,
        threaded=True,
        use_reloader=False,
        processes=1,
    )

