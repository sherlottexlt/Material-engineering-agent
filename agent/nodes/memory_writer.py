"""
Memory Writer 节点：记忆持久化
对应 TDD 第 4.1 节

将完成的案例写入长期记忆（语义记忆 + 情景记忆），仅副作用，不修改状态。
"""
import json
from datetime import datetime

from loguru import logger

from agent.memory.memory_service import MemoryService
from models.entities import BatchParams, CaseRecord, ProcessType
from models.state import AgentState

# 全局 MemoryService 实例（延迟初始化）
_memory_service: MemoryService | None = None


def _get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


async def memory_writer(state: AgentState) -> dict:
    """记忆写入节点

    在流程结束后将案例归档至长期记忆，仅副作用，不修改 LangGraph 状态。

    Args:
        state: LangGraph 全局状态

    Returns:
        空字典（无状态变更）
    """
    trace_id = state.get("trace_id", "unknown")
    batch_id = state.get("batch_id", "unknown")
    logger.info(f"[MemoryWriter] 开始归档案例, trace_id={trace_id}")

    try:
        memory = _get_memory_service()

        # 从状态中提取归因信息
        decision_result = state.get("decision_result") or {}
        data_result = state.get("data_result") or {}
        batch_params = data_result.get("batch_params") or {}

        # 提取根因和方案
        proposals = decision_result.get("proposals", [])
        root_cause = proposals[0].get("root_cause", "未知") if proposals else "未知"
        solution = proposals[0].get("adjustments", "未知") if proposals else "未知"

        # 如果没有 proposals，从 final_answer 提取
        if not proposals:
            root_cause = state.get("final_answer", "未知")[:200]
            solution = "见最终回答"

        # 1. 写入短期记忆（情景记忆）
        memory.write_episodic(
            batch_id=batch_id,
            defect_type="hardness_low",
            root_cause=str(root_cause)[:500],
            solution=str(solution)[:500],
            quality_score=0.7,  # M0 默认评分，后续根据用户反馈更新
        )
        logger.info(f"[MemoryWriter] 短期记忆已写入, batch={batch_id}")

        # 2. 写入长期记忆（语义记忆）
        case = CaseRecord(
            case_id=f"case_{trace_id}",
            defect_type="hardness_low",
            batch_params=BatchParams(
                batch_id=batch_id,
                process_type=ProcessType.HEAT_TREATMENT,
                temperature=batch_params.get("temperature"),
                holding_time=batch_params.get("holding_time"),
                cooling_rate=batch_params.get("cooling_rate"),
                start_time=datetime.now(),
            ),
            root_cause=str(root_cause)[:500],
            solution=str(solution)[:500],
            confidence=0.7,
            created_at=datetime.now(),
            source="auto",
            tags=["M0", "auto_generated"],
        )

        success = memory.write_semantic(case)
        if success:
            logger.info(f"[MemoryWriter] 长期记忆已写入, case_id={case.case_id}")
        else:
            logger.warning("[MemoryWriter] 长期记忆写入跳过（Chroma 不可用）")

    except Exception as e:
        logger.error(f"[MemoryWriter] 记忆写入失败: {e}")

    return {}
