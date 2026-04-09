"""
Operator smoke test runner.

Goal:
- Enumerate all registered operators and run a minimal sequential-slot config for each.
- Report operators that crash with non-OperatorException (bug) or that appear to still require legacy keys.

Notes:
- This is not a full correctness test; it is a compatibility/robustness sweep.
- External-service operators (e.g. ES) are exercised only for "does not crash" behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .. import operators  # noqa: F401  # triggers registry import side effects
from ..core import ExecutionContext, OperatorException, OperatorRegistry, ErrorCode


@dataclass
class SmokeResult:
    operator: str
    status: str  # ok | operator_exception | crash
    error_code: Optional[int] = None
    error: Optional[str] = None


def _base_data() -> Dict[str, Any]:
    return {
        "a": 1,
        "b": 2,
        "n": 3,
        "s": "hello",
        "nums": [1, 2, 3],
        "rows": [{"k": 1, "v": 10}, {"k": 2, "v": 20}],
        "rows2": [{"k": 1, "x": 999}, {"k": 3, "x": 777}],
        "matrixA": [[1, 2], [3, 4]],
        "matrixB": [[5, 6], [7, 8]],
        "vec1": [1, 2, 3],
        "vec2": [4, 5, 6],
        "ts": "2025-01-01T00:00:00",
    }


def _default_config_for(op_name: str) -> Dict[str, Any]:
    """
    Return minimal sequential-slot config likely to reach execute() for most operators.
    """
    # Most arithmetic / compare / stats accept numeric literals directly.
    if op_name in {"add", "subtract", "multiply", "divide", "power"}:
        return {"first_value": 1, "second_value": 2}
    if op_name in {"absolute_value", "sin", "cos", "tan", "log", "sqrt"}:
        return {"first_value": 3}
    if op_name in {"max", "min", "sum"}:
        return {"first_value": 1, "second_value": 2, "third_value": 3}
    if op_name in {"precision_round"}:
        return {"first_value": 1.2345, "second_value": 2}
    if "mean" in op_name or op_name in {"median", "mode", "range", "stddev", "variance", "cv", "quartiles", "iqr"}:
        return {"first_value": [1, 2, 3]}
    if op_name in {"weighted_average", "weighted_variance", "weighted_std"}:
        return {"first_value": [1, 2, 3], "second_value": [1.0, 1.0, 1.0]}
    if op_name in {"count_items"}:
        return {"first_value": [1, 2, 3]}
    if op_name in {"extract_only"}:
        return {"first_value": "rows"}
    if op_name in {"json_extract"}:
        # json_extract input_spec is table; provide a table-like source to avoid TYPE_ERROR false positives.
        return {"first_value": "rows", "second_value": ["k"]}
    if op_name in {"select_fields"}:
        return {"first_value": "rows", "second_value": ["k"]}
    if op_name in {"rename_fields"}:
        return {"first_value": "rows", "second_value": {"v": "value"}}
    if op_name in {"merge_data"}:
        return {"first_value": ["rows", "rows2"], "second_value": "join", "third_value": "k"}
    if op_name in {"flatten_data"}:
        # Provide a dict-of-list so cleaning.flatten_data won't take the "dict scalar" path in DataValue
        return {"first_value": [[1, 2], [3]]}
    if op_name in {"remove_duplicates"}:
        return {"first_value": "rows", "second_value": ["k"]}
    if op_name in {"remove_nulls"}:
        # Use a stable, type-consistent input to avoid TableValue schema/type mismatches.
        return {"first_value": [{"a": 1}, {"a": 2}], "second_value": ["a"], "third_value": "remove"}
    if op_name in {"filter_by_condition"}:
        return {"first_value": "rows", "second_value": [{"field": "k", "operator": "ge", "value": 1}]}
    if op_name in {"sort_data"}:
        return {"first_value": "rows", "second_value": ["k"], "third_value": "asc"}
    if op_name in {"cosine_similarity"}:
        return {"first_value": "vec1", "second_value": "vec2"}
    if op_name in {"matrix_multiply"}:
        return {"first_value": "matrixA", "second_value": "matrixB"}
    if op_name in {"average_time"}:
        return {"first_value": ["2025-01-01T00:00:00", "2025-01-02T00:00:00"]}
    if op_name in {"extract_from_csv"}:
        return {"first_value": "E:/data_calculation/sample_data/sample.csv"}
    if op_name in {"extract_from_excel"}:
        return {"first_value": "E:/data_calculation/sample_data/sample.xlsx"}
    if op_name in {"time_add"}:
        return {"first_value": "ts", "second_value": 60}
    if op_name in {"time_subtract"}:
        return {"first_value": "ts", "second_value": "2024-01-01T00:00:00"}
    if op_name in {"as_number", "as_string", "to_bool"}:
        return {"first_value": 1}
    if op_name in {"split_string"}:
        return {"first_value": "a,b", "second_value": ","}
    if op_name in {"join_list"}:
        return {"first_value": ["a", "b"], "second_value": ","}
    if op_name in {"json_extract"}:
        # json_extract input_spec is table; provide a table-like source to avoid TYPE_ERROR false positives.
        return {"first_value": "rows", "second_value": ["k"]}
    if op_name in {"rows_to_columns"}:
        return {"first_value": "rows"}
    if op_name in {"columns_to_rows"}:
        return {"first_value": [{"k": [1, 2], "v": [10, 20]}]}
    if op_name in {"decline_rate", "growth_rate"}:
        return {"first_value": 10, "second_value": 12}
    if op_name in {"ratio", "proportion", "percentage"}:
        # numerator/part/ratio: base values
        return {"first_value": 1, "second_value": 2}
    if op_name in {"ratio_by_count", "proportion_by_count", "percentage_by_count"}:
        return {"first_value": 1, "second_value": 2}
    if op_name in {"compare_threshold"}:
        return {"first_value": 1, "second_value": 1, "third_value": "ge"}
    if op_name in {"vector_angle", "angle_between", "euclidean_distance", "covariance", "pearson_correlation", "spearman_correlation"}:
        return {"first_value": [1, 2, 3], "second_value": [4, 5, 6]}
    if op_name in {"weighted_average", "weighted_std", "weighted_variance"}:
        return {"first_value": [1, 2, 3], "second_value": [1.0, 1.0, 1.0]}

    # Fallback: provide at least first_value so schema-required ops can run/validate.
    return {"first_value": 1}


def run_smoke(*, exclude: Optional[set[str]] = None) -> Tuple[List[SmokeResult], Dict[str, int]]:
    results: List[SmokeResult] = []
    counts = {"ok": 0, "operator_exception": 0, "crash": 0}

    data = _base_data()
    ctx = ExecutionContext()
    exclude = exclude or set()

    for name in sorted(OperatorRegistry.all_names()):
        if name in exclude:
            continue
        op = OperatorRegistry.get(name)
        cfg = _default_config_for(name)
        try:
            op.run(data, cfg, ctx)
            results.append(SmokeResult(operator=name, status="ok"))
            counts["ok"] += 1
        except OperatorException as e:
            # Expected for some operators (e.g., external services, strict missing data)
            results.append(
                SmokeResult(
                    operator=name,
                    status="operator_exception",
                    error_code=int(e.code) if e.code is not None else None,
                    error=e.message,
                )
            )
            counts["operator_exception"] += 1
        except Exception as e:
            results.append(SmokeResult(operator=name, status="crash", error=str(e)))
            counts["crash"] += 1

    return results, counts


def run_each_operator_exception(*, exclude: Optional[set[str]] = None) -> List[SmokeResult]:
    """
    Run each operator once and only return those that raise OperatorException,
    excluding any operator names in `exclude`.
    """
    exclude = exclude or set()
    out: List[SmokeResult] = []
    data = _base_data()
    ctx = ExecutionContext()

    for name in sorted(OperatorRegistry.all_names()):
        if name in exclude:
            continue
        op = OperatorRegistry.get(name)
        cfg = _default_config_for(name)
        try:
            op.run(data, cfg, ctx)
        except OperatorException as e:
            out.append(
                SmokeResult(
                    operator=name,
                    status="operator_exception",
                    error_code=int(e.code) if e.code is not None else None,
                    error=e.message,
                )
            )
        except Exception as e:
            out.append(SmokeResult(operator=name, status="crash", error=str(e)))
    return out


def main() -> None:
    # Default: skip external-service operators in smoke runs.
    results, counts = run_smoke(exclude={"es_extract", "es_connect"})

    print("== Operator smoke test summary ==")
    print(counts)
    print()

    crashes = [r for r in results if r.status == "crash"]
    if crashes:
        print("== CRASHES (non-OperatorException) ==")
        for r in crashes:
            print(f"- {r.operator}: {r.error}")
        print()

    # Highlight likely "still not supported" ops: config missing/type errors can be normal,
    # but calc-logic errors often indicate code path assumptions.
    suspicious: List[SmokeResult] = []
    for r in results:
        if r.status != "operator_exception":
            continue
        if r.error_code in (int(ErrorCode.CALC_LOGIC_ERROR), int(ErrorCode.UNKNOWN), int(ErrorCode.RUNTIME_ERROR)):
            suspicious.append(r)

    if suspicious:
        print("== Suspicious OperatorExceptions (logic/runtime) ==")
        for r in suspicious:
            print(f"- {r.operator}: code={r.error_code} err={r.error}")
        print()

    print("== Done ==")


if __name__ == "__main__":
    main()

