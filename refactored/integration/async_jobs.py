"""异步计算任务：入队 + Worker，与 runId 释放、结果回调配合。"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, Literal, Optional

from refactored import app_config
from refactored.core.exceptions import ErrorCode
from refactored.core.run_lifecycle import normalize_run_id, release_run
from refactored.pipeline.tree_calculation import response_meta_triple

from .callback_client import post_json

logger = logging.getLogger(__name__)


def _stderr_line(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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


def job_submit(
    body: Dict[str, Any],
    *,
    result_callback_url: str,
    compute_kind: Literal["system", "evaluation"],
) -> str:
    """
    :param result_callback_url: 算完后 POST 的 Java 地址（如 ``JAVA_RESULT_CALLBACK_URL_SYSTEM``）。
    :param compute_kind: ``system`` → ``calculate_system``；``evaluation`` → ``calculate_evaluation``。
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "taskId": body.get("taskId"),
            "runId": body.get("runId"),
            "systemId": body.get("systemId"),
            "compute_kind": compute_kind,
        }
    _pool().submit(_run_job, job_id, body, result_callback_url, compute_kind)
    return job_id


def _run_job(
    job_id: str,
    body: Dict[str, Any],
    result_url: str,
    compute_kind: Literal["system", "evaluation"],
) -> None:
    if compute_kind == "system":
        from refactored.api import calculate_system as _run
    else:
        from refactored.api import calculate_evaluation as _run

    run_id = normalize_run_id(body.get("runId") or body.get("run_id"))
    status_url = app_config.java_status_callback_url_for(compute_kind).strip()
    retries = int(os.environ.get("CALLBACK_RETRIES", "3"))
    meta = {
        "taskId": body.get("taskId"),
        "runId": body.get("runId") or body.get("run_id"),
        "systemId": body.get("systemId"),
    }

    def _compact_error_message(msg: str) -> str:
        if not msg:
            return "unknown error"
        core = msg
        cut = core.find(" [taskId=")
        if cut >= 0:
            core = core[:cut].strip()
        step_id = ""
        try:
            import re

            m = re.search(r"step_id=([^\\s\\]]+)", msg)
            if m:
                step_id = (m.group(1) or "").strip()
        except Exception:
            step_id = ""
        return f"{step_id} {core}".strip() if step_id else core.strip()

    try:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id] = {**_jobs[job_id], "status": "running"}

        logger.warning(
            "异步任务将计算并在结束后 POST 到 %s（compute_kind=%s）",
            result_url,
            compute_kind,
        )
        logger.info(
            "异步任务开始计算 job_id=%s taskId=%s runId=%s",
            job_id,
            meta.get("taskId"),
            meta.get("runId"),
        )
        _stderr_line(
            f"[async_jobs] 后台开始计算 job_id={job_id} taskId={meta.get('taskId')} "
            f"→ 完成后回调 {result_url}"
        )
        result = _run(body)
        ok = bool(result.get("success", True))
        callback_payload: Dict[str, Any]
        if ok:
            callback_payload = dict(result)
        else:
            tid, rid, sid = response_meta_triple(body)
            callback_payload = {
                "success": False,
                "taskId": result.get("taskId", tid),
                "runId": result.get("runId", rid),
                "systemId": result.get("systemId", sid),
                "error_code": int(result.get("error_code") or ErrorCode.RUNTIME_ERROR),
                "message": _compact_error_message(str(result.get("message") or "")),
                "steps": result.get("steps") if isinstance(result.get("steps"), list) else [],
            }
        logger.info(
            "异步任务完成 job_id=%s taskId=%s runId=%s success=%s",
            job_id,
            callback_payload.get("taskId"),
            callback_payload.get("runId"),
            ok,
        )
        _stderr_line(
            f"[async_jobs] 后台计算结束 job_id={job_id} success={ok} → POST {result_url}"
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
            logger.error("结果回调未送达 %s", result_url)
            _stderr_line(f"[async_jobs] 结果回调失败 → {result_url}")
        else:
            logger.info("结果回调已送达 %s", result_url)
            _stderr_line(f"[async_jobs] 结果回调已成功 POST → {result_url}")
    except Exception as e:
        logger.exception("异步任务失败 job_id=%s", job_id)
        tid, rid, sid = response_meta_triple(body)
        err_body: Dict[str, Any] = {
            "success": False,
            "taskId": tid,
            "runId": rid,
            "systemId": sid,
            "error_code": int(ErrorCode.RUNTIME_ERROR),
            "message": str(e),
            "steps": [],
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
        if not post_json(result_url, err_body, retries=retries):
            logger.error("失败结果回调未送达 %s", result_url)
        else:
            logger.info("失败结果回调已送达 %s", result_url)
    finally:
        release_run(run_id)
        with _jobs_lock:
            _jobs.pop(job_id, None)
