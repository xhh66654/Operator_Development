"""推理树协议：递归解析、自底向上执行；回包结构与 Java 约定对齐。

- ``reasoningDataList`` 每项：``{ reasoningId, steps: [...] }``，``steps`` 对应该项下各指标类型节点。
- 仅输出含结果的子树；同级 ``steps`` 若为顺序结构（子项均有 result 且无嵌套 steps），只保留最后一步。
  例外：父节点若带 ``operator_key``（如 ``add``）且非叶子指标，则视为多依赖分支，**不**做同级折叠，避免误吞容器节点（如 ``610111``）。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from statistics import fmean
from typing import Any, Dict, List, Optional, Union

from ..core import ExecutionContext, OperatorRegistry
from ..core.data_model import DataValue
from ..core.exceptions import ErrorCode, OperatorException
from .resource_cleanup import release_pipeline_resources

logger = logging.getLogger(__name__)

_REF_FULL = re.compile(r"^\$\{([^}]+)\}\s*$")

_CTX_DEBUG_SUFFIX_MARK = " [taskId="


def _wrap_operator_exception_with_tree_context(
    ctx: ExecutionContext,
    exc: OperatorException,
    *,
    op_name: str,
    step_key: str,
) -> OperatorException:
    """在异常 message 末尾附加 taskId/runId/reasoningId/step，便于日志与回调定位（不改变成功回包结构）。"""
    if _CTX_DEBUG_SUFFIX_MARK in exc.message:
        return exc
    meta = ctx.get("_tree_request_meta")
    if not isinstance(meta, dict):
        meta = {}
    rid = ctx.get("_active_reasoning_id")
    suffix = (
        f"{_CTX_DEBUG_SUFFIX_MARK}{meta.get('taskId')} runId={meta.get('runId')} "
        f"systemId={meta.get('systemId')} reasoningId={rid} step_id={step_key or '-'} operator={op_name}]"
    )
    return OperatorException(
        exc.message + suffix,
        code=exc.code,
        operator=exc.operator or op_name,
        config=exc.config,
        cause=exc.cause,
    )


def reasoning_entry_child_nodes(rd: Dict[str, Any]) -> List[Any]:
    """
    每项 reasoning 下的根节点列表：与 Java 对齐时优先 ``steps``，兼容旧字段 ``types``。
    二者同时存在时以 ``steps`` 为准。
    """
    if isinstance(rd.get("steps"), list):
        return list(rd["steps"])
    if isinstance(rd.get("types"), list):
        return list(rd["types"])
    return []


def response_meta_triple(req: Dict[str, Any]) -> tuple[Any, Any, Any]:
    """回传 taskId / runId / systemId：与请求 JSON 原样一致，不做类型转换。"""
    return (
        req.get("taskId"),
        req.get("runId") or req.get("run_id"),
        req.get("systemId"),
    )


def _step_timing_enabled() -> bool:
    v = os.environ.get("CALC_LOG_STEP_TIMING", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _float_round_decimals() -> int:
    """
    浮点输出清理精度（仅影响回包 JSON，不改变算子内部计算逻辑）。
    典型现象：5.56455-2.454848 => 3.1097019999999995（IEEE-754），回包希望为 3.109702。
    环境变量：CALC_FLOAT_ROUND_DECIMALS
      - 未设置：默认 7（统一保留 7 位小数）
      - 设为 0：关闭清理
      - 设为正数：固定小数位 round
    """
    raw = os.environ.get("CALC_FLOAT_ROUND_DECIMALS", "").strip()
    if raw == "":
        return 7
    try:
        return int(raw)
    except ValueError:
        return 7


def _float_sig_digits() -> int:
    """
    浮点“有效数字”输出位数（仅影响回包 JSON）。
    - 默认 15：通常可消除尾巴且不强行固定小数位（例如 3.1097019999999995 -> 3.109702）。
    - 可用 CALC_FLOAT_SIG_DIGITS 覆盖；<=0 表示禁用（退回原值）。
    """
    raw = os.environ.get("CALC_FLOAT_SIG_DIGITS", "").strip()
    if raw == "":
        return 15
    try:
        v = int(raw)
        return v
    except ValueError:
        return 15


def _clean_float_for_output(x: float) -> float:
    d = _float_round_decimals()
    if d == 0:
        # 显式关闭清理
        return x
    if d > 0:
        # 固定小数位模式：使用 round() 清理尾数；避免 -0.0
        y = round(x, d)
        return 0.0 if y == 0 else y
    return x


def _log_operator_timing(
    *,
    operator: str,
    step_key: str,
    elapsed_s: float,
    extra: str = "",
) -> None:
    """与根 logger 格式一致，由 logging 统一加 asctime/level/name；消息内带耗时。"""
    ms = elapsed_s * 1000.0
    tail = f" {extra}" if extra else ""
    logger.info("推理树算子 operator=%s step_key=%s 耗时_ms=%.2f%s", operator, step_key or "-", ms, tail)


_NOT_A_SINGLE_CONTEXT_REF = object()


def _resolve_single_context_ref_string(s: str, ctx: ExecutionContext) -> Any:
    """
    若整段字符串为 ``${id}`` 或 ``${id}.列名``（与 field_extractor 语义一致），返回解析结果；
    否则返回哨兵 ``_NOT_A_SINGLE_CONTEXT_REF``。
    """
    s0 = s.strip()
    if not (isinstance(s0, str) and s0.startswith("${") and "}" in s0):
        return _NOT_A_SINGLE_CONTEXT_REF
    end = s0.index("}")
    tail = s0[end + 1 :].strip()
    if tail and not tail.startswith("."):
        return _NOT_A_SINGLE_CONTEXT_REF
    inner = s0[2:end].strip()
    if not inner:
        return _NOT_A_SINGLE_CONTEXT_REF
    subfield: Optional[str] = None
    if tail.startswith("."):
        sub = tail[1:].strip()
        if not sub or "." in sub:
            return _NOT_A_SINGLE_CONTEXT_REF
        subfield = sub
    raw = _ctx_lookup(ctx, inner)
    if subfield is None:
        return raw
    from ..utils.field_extractor import _column_array_from_row_list, _unwrap_column_bundle

    raw_u = _unwrap_column_bundle(raw)
    if isinstance(raw_u, dict) and subfield in raw_u:
        return raw_u[subfield]
    row_col = _column_array_from_row_list(raw, subfield)
    if row_col is not None:
        return row_col
    return None

# 常见中文 operator_key -> 注册名（仅使用 operator_key，忽略 operator_name）
_OPERATOR_KEY_ALIASES: Dict[str, str] = {
    "减法": "subtract",
    "加法": "add",
    "乘法": "multiply",
    "除法": "divide",
    "幂": "power",
    "阈值比较": "compare_threshold",
    "单阈值比较": "single_compare_threshold",
    "多阈值分段": "multi_compare_threshold",
}


def validate_dag_request(data: Any) -> Optional[Dict[str, Any]]:
    """
    Java /calculate 唯一合法体：`execution_mode`=\"dag\" + 非空 `reasoningDataList`；
    每一项须含 ``steps`` 或 ``types`` 数组（与 Java 对齐时推荐 ``steps``）。不再接受顶层 ``steps`` 流水线。
    若非法则返回与成功同 shape 的错误响应（success=false, reasoningDataList=[]），否则返回 None。
    """
    if not isinstance(data, dict):
        return _tree_error({}, "请求体必须为 JSON 对象", ErrorCode.CONFIG_INVALID)
    mode = str(data.get("execution_mode") or "").strip().lower()
    if mode != "dag":
        return _tree_error(
            data,
            '仅支持 execution_mode 为 "dag"，且须包含非空 reasoningDataList（每项含 steps 或 types 数组）',
            ErrorCode.CONFIG_INVALID,
        )
    rdl = data.get("reasoningDataList")
    if not isinstance(rdl, list) or len(rdl) == 0:
        return _tree_error(
            data,
            "reasoningDataList 须为非空数组",
            ErrorCode.CONFIG_INVALID,
        )
    steps_top = data.get("steps")
    if isinstance(steps_top, list) and len(steps_top) > 0:
        return _tree_error(
            data,
            "不再接受顶层 steps；请仅使用 execution_mode=dag 与 reasoningDataList",
            ErrorCode.CONFIG_INVALID,
        )
    for i, rd in enumerate(rdl):
        if not isinstance(rd, dict):
            return _tree_error(
                data,
                f"reasoningDataList[{i}] 必须为对象",
                ErrorCode.CONFIG_INVALID,
            )
        if not isinstance(rd.get("steps"), list) and not isinstance(rd.get("types"), list):
            return _tree_error(
                data,
                f"reasoningDataList[{i}] 须含 steps 或 types 数组",
                ErrorCode.CONFIG_INVALID,
            )
    return None


def is_tree_protocol_request(data: Any) -> bool:
    """兼容旧名：请求是否满足当前 DAG 推理树协议（与 validate_dag_request 通过等价）。"""
    return validate_dag_request(data) is None


def normalize_tree_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    return dict(node)


def normalize_operator_key(key: Union[str, Any]) -> str:
    if key is None or (isinstance(key, str) and not key.strip()):
        raise OperatorException(
            "节点缺少 operator_key",
            code=ErrorCode.CONFIG_MISSING,
        )
    s = str(key).strip()
    return _OPERATOR_KEY_ALIASES.get(s, s)


def _ctx_alias_keys(raw: Any) -> List[str]:
    """
    同一节点 id 在上下文中可能出现的多种字符串键（Java/Gson 等可能把大整数变成 float）。
    set/get 时写入/尝试多个别名，避免 ``2100000000000001`` 与 ``2.1e+15`` 对不上。
    """
    out: List[str] = []
    if raw is None:
        return out
    if isinstance(raw, bool):
        out.append(str(raw))
        return list(dict.fromkeys(out))
    if isinstance(raw, int):
        out.append(str(raw))
        return list(dict.fromkeys(out))
    if isinstance(raw, float):
        out.append(str(raw))
        try:
            if raw.is_integer():
                out.append(str(int(raw)))
        except (ValueError, OverflowError):
            pass
        return list(dict.fromkeys(out))
    s = str(raw).strip()
    if s:
        out.append(s)
    try:
        if "e" not in s.lower() and "." not in s and s.lstrip("-").isdigit():
            out.append(str(int(s)))
    except ValueError:
        pass
    return list(dict.fromkeys(out))


def _ctx_lookup(ctx: ExecutionContext, ref_key: str) -> Any:
    for k in _ctx_alias_keys(ref_key):
        if not k:
            continue
        val = ctx.get(k)
        if val is not None:
            return val
    return None


def resolve_config_refs(value: Any, ctx: ExecutionContext) -> Any:
    """解析配置中的 ${id} 引用（全串或嵌在字符串中）；支持 dict/list 递归。"""
    if isinstance(value, dict):
        return {k: resolve_config_refs(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_config_refs(v, ctx) for v in value]
    if isinstance(value, str):
        s = value.strip()
        single = _resolve_single_context_ref_string(s, ctx)
        if single is not _NOT_A_SINGLE_CONTEXT_REF:
            # 严格模式：若形如 "${610111}" 的上下文引用解析为 None，直接报错。
            # 常见原因：容器节点（无 operator_key 但有 steps）没有把“最后一步结果”写入 ctx，导致 ${容器id} 取不到值。
            if single is None:
                raise OperatorException(
                    f"上下文引用未取到值: {s}（可能是容器节点未写入 ctx，或该节点未执行/无最后结果）",
                    code=ErrorCode.DATA_NOT_FOUND,
                )
            return single
        m = _REF_FULL.match(s)
        if m:
            got = _ctx_lookup(ctx, m.group(1))
            if got is None:
                raise OperatorException(
                    f"上下文引用未取到值: {s}（可能是容器节点未写入 ctx，或该节点未执行/无最后结果）",
                    code=ErrorCode.DATA_NOT_FOUND,
                )
            return got
        if "${" not in s:
            return value

        def _sub(mo: re.Match) -> str:
            subkey = mo.group(1).strip()
            got = _ctx_lookup(ctx, subkey)
            if got is None:
                return ""
            if isinstance(got, (dict, list)):
                return json.dumps(got, ensure_ascii=False)
            return str(got)

        return re.sub(r"\$\{([^}]+)\}", _sub, s)
    return value


def _is_leaf_metric(node: Dict[str, Any]) -> bool:
    """
    叶子指标：无顶层 ``operator_key``、有非空 ``steps`` 算子链，且至少具备其一：

    - ``indicatorId``（引擎原约定）
    - ``reasoningIds`` / ``reasoning_ids`` 非空数组（Java 常见「基础指标」写法）

    仅有 ``id`` 而无上述字段时仍按容器节点处理，避免误判。
    """
    node = normalize_tree_node(node)
    if str(node.get("operator_key") or "").strip():
        return False
    if not isinstance(node.get("steps"), list) or len(node["steps"]) == 0:
        return False
    if node.get("indicatorId") is not None:
        return True
    rids = node.get("reasoningIds") or node.get("reasoning_ids")
    if isinstance(rids, str) and rids.strip():
        rids = [rids]
    if isinstance(rids, list) and any(x is not None and str(x).strip() for x in rids):
        return True
    return False


def _store_node_outputs(ctx: ExecutionContext, node: Dict[str, Any], val: Any) -> None:
    node = normalize_tree_node(node)
    for k in _ctx_alias_keys(node.get("node_id")):
        ctx.set(k, val)
    for k in _ctx_alias_keys(node.get("id")):
        ctx.set(k, val)
    for k in _ctx_alias_keys(node.get("indicatorId")):
        ctx.set(k, val)


def _primary_node_store_key(node: Dict[str, Any]) -> str:
    for k in _ctx_alias_keys(node.get("id")) + _ctx_alias_keys(node.get("node_id")):
        if k:
            return k
    return ""


def _collect_operator_chain_ctx_keys(steps: Any) -> List[str]:
    """叶子算子链里会写入 ctx 的 step id 别名列表，便于按 rid 重跑前清理。"""
    keys: List[str] = []
    if not isinstance(steps, list):
        return keys
    for item in steps:
        if not isinstance(item, dict):
            continue
        if item.get("steps") and not str(item.get("operator_key") or "").strip():
            keys.extend(_collect_operator_chain_ctx_keys(item["steps"]))
            continue
        if str(item.get("operator_key") or item.get("operator") or "").strip():
            for k in _ctx_alias_keys(item.get("id") or item.get("node_id")):
                if k:
                    keys.append(k)
    return list(dict.fromkeys(keys))


def _reasoning_display_lookup(ctx: ExecutionContext, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in _ctx_alias_keys(node.get("id")) + _ctx_alias_keys(node.get("node_id")):
        if not k:
            continue
        d = ctx.get_reasoning_display_result(k)
        if d is not None:
            return d
    return None


def _pythonize(dv: Any) -> Any:
    if isinstance(dv, DataValue):
        return dv.to_python()
    return dv


def run_operator_chain(chain_steps: List[Any], ctx: ExecutionContext) -> Any:
    """执行叶子内算子链，自上而下；每步可用 node_id 写入上下文供后续引用。"""
    last_dv: Any = None
    for raw in chain_steps:
        item = normalize_tree_node(raw)
        if not isinstance(item, dict):
            continue
        if item.get("steps") and not str(item.get("operator_key") or "").strip():
            # 容器节点（无 operator_key，仅包一层 steps）：自身结果取最后一步子节点结果，
            # 需要写入 ctx 以支持 ${container_id} 引用。
            last_dv = run_operator_chain(item["steps"], ctx)
            raw_key = item.get("id") or item.get("node_id") or item.get("step_key")
            for step_key in _ctx_alias_keys(raw_key):
                ctx.set(step_key, _pythonize(last_dv))
            continue
        op_key = item.get("operator_key") or item.get("operator")
        if not op_key:
            continue
        op_name = normalize_operator_key(op_key)
        raw_key = item.get("id") or item.get("node_id") or item.get("step_key")
        step_keys = _ctx_alias_keys(raw_key)
        step_key = step_keys[0] if step_keys else ""
        cfg = resolve_config_refs(copy.deepcopy(item.get("config") or {}), ctx)
        op = OperatorRegistry.get(op_name)
        t0 = time.perf_counter()
        try:
            last_dv = op.run({}, cfg, ctx)
        except OperatorException as e:
            raise _wrap_operator_exception_with_tree_context(
                ctx, e, op_name=op_name, step_key=step_key
            ) from e
        if _step_timing_enabled():
            _log_operator_timing(
                operator=op_name,
                step_key=step_key,
                elapsed_s=time.perf_counter() - t0,
            )
        py = _pythonize(last_dv)
        for sk in step_keys:
            ctx.set(sk, py)
    return last_dv


def exec_metric_subtree(node: Any, ctx: ExecutionContext) -> None:
    """自底向上：先子节点，再当前节点算子。"""
    node = normalize_tree_node(node)
    if not isinstance(node, dict):
        return

    if _is_leaf_metric(node):
        rids_raw = node.get("reasoningIds") or node.get("reasoning_ids")
        if isinstance(rids_raw, str) and rids_raw.strip():
            rids_raw = [rids_raw]
        if isinstance(rids_raw, list) and any(x is not None and str(x).strip() for x in rids_raw):
            chain_keys = _collect_operator_chain_ctx_keys(node.get("steps"))
            per: Dict[str, Any] = {}
            prev_outer: Any = ctx.get("_active_reasoning_id")
            try:
                for rid in rids_raw:
                    if rid is None or not str(rid).strip():
                        continue
                    rk = str(rid).strip()
                    ctx.set("_active_reasoning_id", rid)
                    last = run_operator_chain(node["steps"], ctx)
                    per[rk] = _pythonize(last)
                    for ck in chain_keys:
                        ctx.remove(ck)
            finally:
                if prev_outer is not None:
                    ctx.set("_active_reasoning_id", prev_outer)
                else:
                    ctx.remove("_active_reasoning_id")
            nums: List[float] = []
            for v in per.values():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    if isinstance(v, float) and (v != v or abs(v) == float("inf")):
                        continue
                    nums.append(float(v))
            agg: Any = fmean(nums) if nums else None
            _store_node_outputs(ctx, node, agg)
            pk = _primary_node_store_key(node)
            if pk:
                ctx.set_reasoning_display_result(pk, per)
            return
        last = run_operator_chain(node["steps"], ctx)
        val = _pythonize(last)
        _store_node_outputs(ctx, node, val)
        return

    for ch in node.get("steps") or []:
        if isinstance(ch, dict):
            exec_metric_subtree(ch, ctx)

    op_raw = node.get("operator_key") or node.get("operator")
    if str(op_raw or "").strip():
        op_name = normalize_operator_key(op_raw)
        cfg = resolve_config_refs(copy.deepcopy(node.get("config") or {}), ctx)
        op = OperatorRegistry.get(op_name)
        sk = str(node.get("id") or node.get("node_id") or "").strip() or "-"
        t0 = time.perf_counter()
        try:
            dv = op.run({}, cfg, ctx)
        except OperatorException as e:
            raise _wrap_operator_exception_with_tree_context(
                ctx, e, op_name=op_name, step_key=sk
            ) from e
        if _step_timing_enabled():
            _log_operator_timing(
                operator=op_name,
                step_key=sk,
                elapsed_s=time.perf_counter() - t0,
            )
        val = _pythonize(dv)
        _store_node_outputs(ctx, node, val)
        return

    # 容器节点（无 operator_key 但有 steps）：将最后一个子节点结果写入自身 id，供 ${id} 引用。
    steps = node.get("steps") or []
    if isinstance(steps, list) and steps:
        last_child = steps[-1] if isinstance(steps[-1], dict) else None
        if isinstance(last_child, dict):
            last_val = _lookup_node_result(last_child, ctx)
            if last_val is not None:
                _store_node_outputs(ctx, node, _unwrap_single_scalar_list(last_val))


def _lookup_node_result(node: Dict[str, Any], ctx: ExecutionContext) -> Any:
    node = normalize_tree_node(node)
    for k in _ctx_alias_keys(node.get("node_id")):
        v = ctx.get(k)
        if v is not None:
            return v
    for k in _ctx_alias_keys(node.get("id")):
        v = ctx.get(k)
        if v is not None:
            return v
    for k in _ctx_alias_keys(node.get("indicatorId")):
        v = ctx.get(k)
        if v is not None:
            return v
    return None


def _jsonable(x: Any) -> Any:
    if isinstance(x, DataValue):
        return x.to_python()
    if isinstance(x, float):
        return _clean_float_for_output(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    try:
        json.dumps(x, default=str)
        return x
    except TypeError:
        return str(x)


def _should_collapse_sequential_sibling_frags(parent: Dict[str, Any]) -> bool:
    """
    顺序链折叠仅适用于「无算子、仅包一层线性子 step」的容器；**不**用于叶子指标内部的算子管道
    （否则在部分 ``result`` 声明组合下会误缩掉中间步，或与回包期望不一致）。
    """
    parent = normalize_tree_node(parent)
    if _is_leaf_metric(parent):
        return False
    if not str(parent.get("operator_key") or parent.get("operator") or "").strip():
        return True
    return False


def _any_non_last_step_requested_result(parent: Dict[str, Any]) -> bool:
    """
    请求里若在「非最后一个」子 step 上声明了 ``result``，则回包须保留整条顺序链，
    不得折叠为仅最后一步（否则带 ``result`` 的中间算子会从 JSON 中消失）。
    """
    parent = normalize_tree_node(parent)
    steps = parent.get("steps") or []
    if not isinstance(steps, list) or len(steps) <= 1:
        return False
    for ch in steps[:-1]:
        if isinstance(ch, dict) and "result" in ch:
            return True
    return False


def _collapse_sequential_step_results(
    frags: List[Dict[str, Any]], *, parent: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    同一父节点下 ``steps`` 子项若为顺序结构：每项均有 ``result`` 且无嵌套 ``steps``，
    则只保留最后一步（避免算子链每层全量展开）。

    若请求在非末子节点上也带了 ``result``，则不折叠，与「有 result 则回传该节点结果」一致。
    """
    if len(frags) <= 1:
        return frags
    if not _should_collapse_sequential_sibling_frags(parent):
        return frags
    if _any_non_last_step_requested_result(parent):
        return frags
    if all("result" in f and not f.get("steps") for f in frags):
        return [frags[-1]]
    return frags


def _final_result_for_output(x: Any) -> Any:
    """
    回传结果做一次收敛：
    - 若结果是“多个候选结果”的 list（常见于逐步产出的列表），则只取最后一个作为最终结果输出
    - 其它情况保持原样，避免破坏本应返回数组的算子输出
    """
    # 单元素标量数组：解包为标量（用户期望一个值不要用列表包裹）
    if isinstance(x, list) and len(x) == 1 and not isinstance(x[0], (dict, list, tuple)):
        return x[0]
    return x


def _unwrap_single_scalar_list(x: Any) -> Any:
    """
    用于“容器节点 steps 简化输出”：若结果是单元素标量数组（如 [15955.49]），解包为 15955.49。
    仅在 steps 被简化为值时使用，避免破坏本应返回数组的真实算子输出结构。
    """
    if isinstance(x, list) and len(x) == 1 and not isinstance(x[0], (dict, list, tuple)):
        return x[0]
    return x


def _id_for_output(raw_id: Any, path: str) -> Any:
    """
    回包里的 id 尽量保留“数字类型”：
    - 6 / "6" / 6.0 -> 6
    - 其它字符串保持字符串
    - 缺失则生成 auto_{path} 字符串
    """
    if raw_id is None:
        return f"auto_{path}"
    if isinstance(raw_id, bool):
        return raw_id
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, float):
        return int(raw_id) if raw_id.is_integer() else raw_id
    if isinstance(raw_id, str):
        s = raw_id.strip()
        if s == "":
            return f"auto_{path}"
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except ValueError:
                return s
        return s
    return raw_id


def build_result_tree(orig_node: Any, ctx: ExecutionContext, path: str = "0") -> Optional[Dict[str, Any]]:
    """
    生成用于回传的结果树片段。

    **是否带 ``result``**：请求里声明了 ``result``、或存在可下钻的子 ``steps``、或执行上下文已有该子节点
    的值时，继续构建（与 ``system_protocol`` 出站裁剪配合：最终是否保留 ``result`` 以请求是否带键为准）。

    **值的来源（不按 ``config`` 推断）**：先 ``_lookup_node_result(orig_node, ctx)``；若无且请求带
    ``result`` 且已有子片段，再用末子片段 ``result`` 等兜底；容器还可能已在 exec 阶段被写入与末子
    一致的上下文值。

    **顺序链**：子级构建后经 ``_collapse_sequential_step_results``；若请求在非末子 step 上也带
    ``result``，则不折叠为仅最后一步。

    输出字段：``id``、可选 ``node_id``、需要时的 ``result``、可选嵌套 ``steps``。
    """
    orig_node = normalize_tree_node(orig_node)
    if not isinstance(orig_node, dict):
        return None

    dr = _reasoning_display_lookup(ctx, orig_node)
    if dr is not None:
        raw_id = orig_node.get("node_id")
        if raw_id is None or str(raw_id).strip() == "":
            raw_id = orig_node.get("id")
        if raw_id is None or str(raw_id).strip() == "":
            raw_id = orig_node.get("indicatorId")
        node_id_out = _id_for_output(raw_id, path)
        frag_rd: Dict[str, Any] = {"id": node_id_out}
        if orig_node.get("node_id") is not None:
            frag_rd["node_id"] = orig_node["node_id"]
        if "result" in orig_node:
            frag_rd["result"] = _final_result_for_output(_jsonable(dr))
        if orig_node.get("type") is not None:
            frag_rd["type"] = orig_node["type"]
        return frag_rd

    child_frags: List[Dict[str, Any]] = []
    for idx, ch in enumerate(orig_node.get("steps") or []):
        if not isinstance(ch, dict):
            continue
        # 兼容“最底层算子节点未显式带 result:{}”的情况：
        # 即使输入没声明 result，只要该节点执行阶段写入了上下文（computed != None），也应纳入回包树。
        has_result_key = "result" in ch
        sub = ch.get("steps")
        has_child_steps = isinstance(sub, list) and len(sub) > 0
        has_computed = _lookup_node_result(ch, ctx) is not None
        if not has_result_key and not has_child_steps and not has_computed:
            continue
        cf = build_result_tree(ch, ctx, f"{path}.{idx}")
        if cf:
            child_frags.append(cf)

    child_frags = _collapse_sequential_step_results(child_frags, parent=orig_node)

    raw_id = orig_node.get("node_id")
    if raw_id is None or str(raw_id).strip() == "":
        raw_id = orig_node.get("id")
    if raw_id is None or str(raw_id).strip() == "":
        raw_id = orig_node.get("indicatorId")
    node_id_out = _id_for_output(raw_id, path)

    computed = _lookup_node_result(orig_node, ctx)
    wants_result = "result" in orig_node
    op_raw = orig_node.get("operator_key") or orig_node.get("operator")
    is_container = not str(op_raw or "").strip()
    # 容器常见写法：请求带 result，值可能已在 exec 阶段写入上下文（与末子一致）；若仍取不到则用子片段兜底。
    if computed is None and wants_result and child_frags:
        computed = child_frags[-1].get("result")
    # 容器节点的最终结果若是单元素标量数组，回包时解包为标量
    if is_container and wants_result:
        computed = _unwrap_single_scalar_list(computed)
    has_computed = computed is not None
    should_include_result = wants_result or has_computed

    if not child_frags and not should_include_result:
        return None

    frag: Dict[str, Any] = {"id": node_id_out}
    # 仅当输入声明了 result 或上下文能解析到值时输出 result，避免空壳节点带 null
    if should_include_result:
        frag["result"] = _final_result_for_output(_jsonable(computed))
    if orig_node.get("node_id") is not None:
        frag["node_id"] = orig_node["node_id"]
    if child_frags:
        # 若当前节点是“容器节点”（无 operator_key，仅包一层 steps），且顺序链已折叠为最后一步，
        # 则进一步“扁平化”：不再输出 steps，仅输出最终 result：
        # 例：{"id":"4","result":[15955.49],"steps":[{"id":"6","result":[15955.49]}]}
        #  => {"id":"4","result":15955.49}
        if is_container and len(child_frags) == 1 and not child_frags[0].get("steps") and "result" in child_frags[0]:
            if "result" in frag:
                frag["result"] = _unwrap_single_scalar_list(frag.get("result"))
            else:
                frag["result"] = _unwrap_single_scalar_list(child_frags[0].get("result"))
        else:
            frag["steps"] = child_frags
    return frag


def execute_tree_calculation(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行整棵推理树。成功时::

        {
          "success": true,
          "taskId", "runId", "systemId",
          "reasoningDataList": [
            { "reasoningId": ..., "steps": [ /* 各指标类型根节点片段 */ ] },
            ...
          ]
        }

    （与 HTTP 第一次回执 ``{ taskId, runId, systemId, status:\"success\" }`` 分离，后者见 ``refactored.service``。）
    """
    task_id, run_id, system_id = response_meta_triple(request_data)
    logger.info(
        "推理树计算开始 taskId=%s runId=%s systemId=%s",
        task_id,
        run_id,
        system_id,
    )
    ctx = ExecutionContext()
    ctx.set(
        "_tree_request_meta",
        {"taskId": task_id, "runId": run_id, "systemId": system_id},
    )
    try:
        from ..operators.connection.elasticsearch import warm_es_client_for_calculation

        warm_es_client_for_calculation(ctx, request_data)

        _t_wall = time.perf_counter()
        rdl = request_data.get("reasoningDataList") or []
        if not isinstance(rdl, list):
            return _tree_error(request_data, "reasoningDataList 必须为数组", ErrorCode.CONFIG_INVALID)

        for rd in rdl:
            if not isinstance(rd, dict):
                continue
            rid = rd.get("reasoningId") if rd.get("reasoningId") is not None else rd.get("reasoning_id")
            # 让同一 reasoningDataList 分支内的 es_extract 按当前 rid（ES 文档字段 sRID）过滤
            if rid is not None and rid != "":
                ctx.set("_active_reasoning_id", rid)
            try:
                for type_node in reasoning_entry_child_nodes(rd):
                    if isinstance(type_node, dict):
                        exec_metric_subtree(type_node, ctx)
            finally:
                ctx.remove("_active_reasoning_id")

        out_reasoning: List[Dict[str, Any]] = []
        for rd in rdl:
            if not isinstance(rd, dict):
                continue
            rid = rd.get("reasoningId") if rd.get("reasoningId") is not None else rd.get("reasoning_id")
            types_list = reasoning_entry_child_nodes(rd)
            if not types_list:
                continue
            step_nodes: List[Dict[str, Any]] = []
            for type_node in types_list:
                if not isinstance(type_node, dict):
                    continue
                frag = build_result_tree(type_node, ctx, "0")
                if frag:
                    step_nodes.append(frag)
            if step_nodes:
                out_reasoning.append({"reasoningId": rid, "steps": step_nodes})

        logger.info(
            "推理树计算完成 taskId=%s runId=%s 输出条目数=%s 总耗时_ms=%.2f",
            task_id,
            run_id,
            len(out_reasoning),
            (time.perf_counter() - _t_wall) * 1000.0,
        )
        return {
            "success": True,
            "taskId": task_id,
            "runId": run_id,
            "systemId": system_id,
            "reasoningDataList": out_reasoning,
        }
    except OperatorException as e:
        logger.warning("推理树执行失败: %s", e.message)
        return {
            "success": False,
            "taskId": task_id,
            "runId": run_id,
            "systemId": system_id,
            "error_code": int(e.code),
            "message": e.message,
            "reasoningDataList": [],
        }
    except KeyError as e:
        return _tree_error(request_data, f"未知算子或缺少配置键: {e}", ErrorCode.CONFIG_MISSING)
    except Exception as e:
        logger.exception("推理树未捕获异常")
        return _tree_error(request_data, str(e), ErrorCode.RUNTIME_ERROR)
    finally:
        release_pipeline_resources(ctx)
        ctx.dispose()


def _tree_error(req: Dict[str, Any], message: str, code: ErrorCode) -> Dict[str, Any]:
    tid, rid, sid = response_meta_triple(req)
    return {
        "success": False,
        "taskId": tid,
        "runId": rid,
        "systemId": sid,
        "error_code": int(code),
        "message": message,
        "reasoningDataList": [],
    }
