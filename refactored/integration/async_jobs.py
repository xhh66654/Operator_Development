"""异步计算任务：入队 + Worker，与 runId 释放、结果/状态回调配合。"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, Optional

from refactored.core.exceptions import ErrorCode
from refactored.core.run_lifecycle import normalize_run_id, release_run
from refactored.pipeline.tree_calculation import response_meta_triple
from refactored import app_config

from .callback_client import post_json

logger = logging.getLogger(__name__)


def _stderr_line(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

DEFAULT_RESULT_CALLBACK_URL = app_config.RESULT_CALLBACK_URL_DEFAULT

_executor: Optional[ThreadPoolExecutor] = None
_jobs_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        n = int(os.environ.get("CALC_WORKER_THREADS", "8"))
        _executor = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="calc")
    return _executor


def job_get(job_id: str) -> Dict[str, Any]:
    """任务结束后会从内存移除，若已清理则返回空 dict。"""
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


def job_submit(body: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "taskId": body.get("taskId"),
            "runId": body.get("runId"),
            "systemId": body.get("systemId"),
        }
    _pool().submit(_run_job, job_id, body)
    return job_id


def _run_job(job_id: str, body: Dict[str, Any]) -> None:
    from refactored.api import calculate_indicator

    run_id = normalize_run_id(body.get("runId") )
    status_url = os.environ.get("STATUS_CALLBACK_URL", "").strip()
    result_url = os.environ.get("RESULT_CALLBACK_URL", "").strip() or DEFAULT_RESULT_CALLBACK_URL
    retries = int(os.environ.get("CALLBACK_RETRIES", "3"))
    meta = {
        "taskId": body.get("taskId"),
        "runId": body.get("runId") ,
        "systemId": body.get("systemId"),
    }
    try:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id] = {**_jobs[job_id], "status": "running"}

        logger.warning(
            "异步任务将计算并在结束后 POST 到 %s（须另有进程监听，否则无落盘）",
            result_url,
        )
        logger.info(
            "异步任务开始计算 job_id=%s taskId=%s runId=%s",
            job_id,
            meta.get("taskId"),
            meta.get("runId"),
        )
        _stderr_line(
            f"[async_jobs] 后台开始计算 job_id={job_id} taskId={meta.get('taskId')} "
            f"（完成后回调 {result_url}）"
        )
        result = calculate_indicator(body)
        ok = bool(result.get("success", True))
        tid, rid, sid = response_meta_triple(body)
        callback_payload: Dict[str, Any] = {
            "success": ok,
            "taskId": result.get("taskId", tid),
            "runId": result.get("runId", rid),
            "systemId": result.get("systemId", sid),
            "reasoningDataList": result.get("reasoningDataList") or [],
        }
        logger.info(
            "异步任务完成 job_id=%s taskId=%s runId=%s success=%s nodes=%s",
            job_id,
            callback_payload.get("taskId"),
            callback_payload.get("runId"),
            ok,
            len(callback_payload.get("reasoningDataList") or []),
        )
        _stderr_line(
            f"[async_jobs] 后台计算结束 job_id={job_id} success={ok} "
            f"→ 即将 POST 到 {result_url}"
        )
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id] = {
                    **_jobs[job_id],
                    "status": "success" if ok else "failed",
                    "result": result,
                }

        if status_url:
            if not post_json(
                status_url,
                {
                    **meta,
                    "status": "success" if ok else "failed",
                    **({} if ok else {"message": result.get("message", "")}),
                },
                retries=retries,
            ):
                logger.error("状态回调未送达 %s", status_url)
        if not post_json(result_url, callback_payload, retries=retries):
            logger.error("结果回调未送达 %s（Java ResultCallbackServer 不会写文件）", result_url)
            _stderr_line(
                f"[async_jobs] 结果回调失败 → {result_url} "
                f"（请先启动 Java PythonResultReceiver 监听 8091，或检查防火墙）"
            )
        else:
            logger.info("结果回调已送达 %s", result_url)
            _stderr_line(f"[async_jobs] 结果回调已成功 POST → {result_url}")
    except Exception as e:
        logger.exception("异步任务失败 job_id=%s", job_id)
        err_body: Dict[str, Any] = {
            "success": False,
            "taskId": meta.get("taskId"),
            "runId": meta.get("runId"),
            "systemId": meta.get("systemId"),
            "error_code": int(ErrorCode.RUNTIME_ERROR),
            "message": str(e),
            "reasoningDataList": [],
        }
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id] = {
                    **_jobs[job_id],
                    "status": "failed",
                    "error": str(e),
                    "result": err_body,
                }
        if status_url:
            if not post_json(
                status_url, {**meta, "status": "failed", "message": str(e)}, retries=retries
            ):
                logger.error("状态回调未送达 %s", status_url)
        if not post_json(
            result_url,
            {
                "success": False,
                "taskId": meta.get("taskId"),
                "runId": meta.get("runId"),
                "systemId": meta.get("systemId"),
                "reasoningDataList": [],
            },
            retries=retries,
        ):
            logger.error("失败结果回调未送达 %s", result_url)
        else:
            logger.info("失败结果回调已送达 %s", result_url)
    finally:
        release_run(run_id)
        with _jobs_lock:
            _jobs.pop(job_id, None)
