"""
Self-test for weighted operators:
- weighted_average
- weighted_std
- weighted_variance

Focus:
1) When weights length mismatches expanded observations length, MUST raise OperatorException(SCHEMA_MISMATCH=2003).
2) When lengths match, should compute without raising.

Run:
  python -m refactored.utils.weighted_ops_selftest
"""

from __future__ import annotations

from typing import Any, Dict

from ..core import ExecutionContext, OperatorException, OperatorRegistry, ErrorCode
from .. import operators  # noqa: F401  # ensure operator registration


def _run_case(op_name: str, data: Dict[str, Any], cfg: Dict[str, Any]) -> Any:
    op = OperatorRegistry.get(op_name)
    ctx = ExecutionContext()
    return op.run(data, cfg, ctx).to_python()


def _expect_schema_mismatch(op_name: str, data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    try:
        _run_case(op_name, data, cfg)
    except OperatorException as e:
        if int(e.code) != int(ErrorCode.SCHEMA_MISMATCH):
            raise AssertionError(f"{op_name} expected SCHEMA_MISMATCH(2003), got code={int(e.code)} err={e.message}")
        return
    raise AssertionError(f"{op_name} expected SCHEMA_MISMATCH(2003), but no exception raised")


def _expect_ok(op_name: str, data: Dict[str, Any], cfg: Dict[str, Any]) -> Any:
    try:
        return _run_case(op_name, data, cfg)
    except OperatorException as e:
        raise AssertionError(f"{op_name} expected OK, got OperatorException code={int(e.code)} err={e.message}") from e


def main() -> None:
    # Use field references (not numeric literals) to follow the operator's intended IO path.
    data = {"x1": 1.0, "x2": 2.0, "x3": 3.0, "x4": 4.0, "x5": 5.0, "x6": 6.0}

    # Case A: mismatch (observations len=6, weights len=3)
    mismatch_cfg = {
        "first_value": ["x1", "x2", "x3", "x4", "x5", "x6"],
        "second_value": [1.0, 1.0, 1.0],
    }

    for op_name in ("weighted_average", "weighted_std", "weighted_variance"):
        _expect_schema_mismatch(op_name, data, mismatch_cfg)

    # Case B: match (observations len=3, weights len=3)
    ok_cfg = {"first_value": ["x1", "x2", "x3"], "second_value": [1.0, 1.0, 1.0]}
    avg = _expect_ok("weighted_average", data, ok_cfg)
    var = _expect_ok("weighted_variance", data, ok_cfg)
    std = _expect_ok("weighted_std", data, ok_cfg)

    print("== weighted ops selftest ==")
    print("mismatch: PASS (raised SCHEMA_MISMATCH=2003)")
    print("ok: PASS")
    print("weighted_average:", avg)
    print("weighted_variance:", var)
    print("weighted_std:", std)
    print("== done ==")


if __name__ == "__main__":
    main()

