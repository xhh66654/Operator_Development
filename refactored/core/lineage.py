"""数据血缘追踪"""
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class StepLineage:
    """单步血缘"""
    step_key: str
    operator: str
    input_sources: List[str] = field(default_factory=list)
    output_fields: List[str] = field(default_factory=list)
    raw_input_refs: Set[str] = field(default_factory=set)


@dataclass
class PipelineLineage:
    """整条流水线血缘"""

    steps: List[StepLineage] = field(default_factory=list)

    def add_step(
        self,
        step_key: str,
        operator: str,
        input_sources: List[str],
        output_fields: List[str],
    ) -> StepLineage:
        step = StepLineage(
            step_key=step_key,
            operator=operator,
            input_sources=input_sources,
            output_fields=output_fields,
        )
        self.steps.append(step)
        return step

    def resolve_final_dependencies(self) -> Set[str]:
        """解析最终结果依赖的原始 record（数据表字段快照）中的字段集合（含步骤间引用）"""
        # 先收集每步直接引用的键（非 ${...}）
        for step in self.steps:
            for ref in step.input_sources:
                if ref.startswith("record."):
                    step.raw_input_refs.add(ref.replace("record.", ""))
                elif ref.startswith("base_data."):
                    step.raw_input_refs.add(ref.replace("base_data.", ""))
                elif ref and not (ref.startswith("${") and ref.endswith("}")):
                    step.raw_input_refs.add(ref)
        # 再沿步骤顺序合并：被引用的 step 的 raw_input_refs 并入后续步骤
        resolved = set()
        for step in self.steps:
            resolved.update(step.raw_input_refs)
            step.raw_input_refs = set(resolved)
        return resolved
