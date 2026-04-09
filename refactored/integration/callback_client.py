"""HTTP JSON 回调：状态 / 结果推送（带简单重试）。"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict

logger = logging.getLogger(__name__)


def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 120.0,
    retries: int = 3,
) -> bool:
    if not url or not str(url).strip():
        return False
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(
                str(url).strip(),
                data=data,
                method="POST",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            return True
        except urllib.error.HTTPError as e:
            last_err = e
            # HTTPError 也属于可读出 status code 的响应
            try:
                body = e.read()
                body_preview = body[:200].decode("utf-8", errors="replace") if body else ""
            except Exception:
                body_preview = ""
            logger.warning(
                "回调失败 (%s/%s) %s: http_status=%s reason=%s body=%s",
                attempt + 1,
                retries,
                url,
                getattr(e, "code", None),
                getattr(e, "reason", None),
                body_preview,
            )
            time.sleep(min(2**attempt, 30))
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = e
            logger.warning("回调失败 (%s/%s) %s: %s", attempt + 1, retries, url, e)
            time.sleep(min(2**attempt, 30))
    if last_err:
        logger.error("回调最终失败 %s: %s", url, last_err)
    return False
