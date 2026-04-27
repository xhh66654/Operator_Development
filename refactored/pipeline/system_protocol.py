"""
体系 / 结果修改：顶层 ``steps`` 协议 ↔ 既有 ``reasoningDataList`` 执行引擎的适配与响应整形。

- 入站：``execution_mode=dag`` + 非空顶层 ``steps``（不再使用顶层 ``reasoningDataList``）。
- 执行：包装为单条 ``reasoningDataList`` 后调用 ``execute_tree_calculation``。
- 出站：``success`` / ``taskId`` / ``runId`` / ``systemId`` / （成功时）``execution_mode`` + ``steps``。

**``result`` 回包约定（与 ``tree_calculation.build_result_tree`` 一致）**

1. **是否出现 ``result``**：仅当**请求**里该节点带有 ``result`` 键时，响应中才保留该节点的 ``result``
   （见 ``_prune_result_if_not_requested``）；不要用 ``config`` 有无推断是否回传。
2. **数值从哪来**：优先取**该节点**在执行上下文中的输出（``id`` / ``node_id`` / ``indicatorId`` 写入的值）；
   若仍无值、但请求声明了 ``result`` 且已构建子片段，则用**子树回包**中的兜底规则（如末子片段的
   ``result``）。容器节点在执行阶段也可能已被写入与末子一致的值，二者语义一致，均视为「执行产出」，
   而非按 ``config`` 区分来源。
3. **顺序链展开**：若请求在**非最后一个**子 step 上也带了 ``result``，顺序算子链不在回包中折叠为
   仅最后一步，以保证每个声明了 ``result`` 的节点都能在 JSON 中出现。
4. **``reasoningIds``（非空）的叶子指标**：对每条 reasoning 各跑一遍子算子链；回包 ``result`` 为
   ``{ "rid字符串": 值, ... }``（单条也为单键对象）；上下文 ``id`` 上另写**算术平均**供 ``${id}`` 等内部引用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.exceptions import ErrorCode
from .tree_calculation import execute_tree_calculation, response_meta_triple


def _steps_error(req: Dict[str, Any], message: str, code: ErrorCode) -> Dict[str, Any]:
    tid, rid, sid = response_meta_triple(req if isinstance(req, dict) else {})
    return {
        "success": False,
        "taskId": tid,
        "runId": rid,
        "systemId": sid,
        "error_code": int(code),
        "message": message,
        "steps": [],
    }


def validate_steps_root_request(data: Any) -> Optional[Dict[str, Any]]:
    """
    校验顶层 ``steps`` 协议。失败时返回 **新格式** 错误体（可直接作 HTTP JSON）。
    """
    if not isinstance(data, dict):
        return _steps_error({}, "请求体必须为 JSON 对象", ErrorCode.CONFIG_INVALID)
    mode = str(data.get("execution_mode") or "").strip().lower()
    if mode != "dag":
        return _steps_error(
            data,
            '仅支持 execution_mode 为 "dag"',
            ErrorCode.CONFIG_INVALID,
        )
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return _steps_error(data, "steps 须为非空数组", ErrorCode.CONFIG_INVALID)
    rdl = data.get("reasoningDataList")
    if isinstance(rdl, list) and len(rdl) > 0:
        return _steps_error(
            data,
            "请仅使用顶层 steps；不再接受 reasoningDataList",
            ErrorCode.CONFIG_INVALID,
        )
    return None


def to_legacy_reasoning_request(system_req: Dict[str, Any]) -> Dict[str, Any]:
    """将顶层 steps 包成单条 reasoningDataList，供 ``execute_tree_calculation`` 使用。"""
    return {
        "taskId": system_req.get("taskId"),
        "runId": system_req.get("runId") or system_req.get("run_id"),
        "systemId": system_req.get("systemId"),
        "execution_mode": "dag",
        "reasoningDataList": [{"steps": list(system_req.get("steps") or [])}],
    }


def _prune_result_if_not_requested(resp_nodes: Any, req_nodes: Any) -> None:
    """请求未带 ``result`` 键的节点，从响应中删除 ``result``（就地修改）；与回包「有声明才保留」一致。"""
    if not isinstance(resp_nodes, list) or not isinstance(req_nodes, list):
        return
    n = min(len(resp_nodes), len(req_nodes))
    for i in range(n):
        rn, qn = resp_nodes[i], req_nodes[i]
        if not isinstance(rn, dict) or not isinstance(qn, dict):
            continue
        if "result" in rn and "result" not in qn:
            del rn["result"]
        rs, qs = rn.get("steps"), qn.get("steps")
        if isinstance(rs, list) and isinstance(qs, list):
            _prune_result_if_not_requested(rs, qs)


def steps_response_from_legacy(legacy: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    """将 ``execute_tree_calculation`` 的 legacy 回包转为顶层 ``steps`` 新格式。"""
    tid = legacy.get("taskId")
    rid = legacy.get("runId")
    sid = legacy.get("systemId")
    if not legacy.get("success", True):
        return {
            "success": False,
            "taskId": tid,
            "runId": rid,
            "systemId": sid,
            "error_code": int(legacy.get("error_code") or ErrorCode.RUNTIME_ERROR),
            "message": str(legacy.get("message") or ""),
            "steps": [],
        }
    rdl = legacy.get("reasoningDataList") or []
    steps_out: List[Any] = []
    if rdl and isinstance(rdl[0], dict):
        steps_out = list(rdl[0].get("steps") or [])
    req_steps = original.get("steps") or []
    _prune_result_if_not_requested(steps_out, req_steps if isinstance(req_steps, list) else [])
    return {
        "success": True,
        "taskId": tid,
        "runId": rid,
        "systemId": sid,
        "execution_mode": "dag",
        "steps": steps_out,
    }


def execute_steps_tree_calculation(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行顶层 ``steps`` 体系树（与结果修改入站形状一致时可直接复用）。
    """
    err = validate_steps_root_request(request_data)
    if err is not None:
        return err
    legacy = to_legacy_reasoning_request(request_data)
    legacy_result = execute_tree_calculation(legacy)
    return steps_response_from_legacy(legacy_result, request_data)
