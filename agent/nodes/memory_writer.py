"""
Memory Writer 节点：记忆持久化
对应 TDD 第 4.1 节

将完成的案例写入长期记忆（语义记忆 + 情景记忆），仅副作用，不修改状态。
"""
import json
from datetime import datetime

from loguru import logger

from agent.memory.memory_service import MemoryService
from models.entities import BatchParams, CaseRecord, DefectType, ProcessType
from models.state import AgentState

# 全局 MemoryService 实例（延迟初始化）
_memory_service: MemoryService | None = None


def _get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


# 缺陷类型关键词映射（root_cause 文本 → DefectType 枚举）
_DEFECT_KEYWORD_MAP = [
    ("硬度偏低", DefectType.HARDNESS_LOW),
    ("硬度不足", DefectType.HARDNESS_LOW),
    ("硬度偏高", DefectType.HARDNESS_HIGH),
    ("硬度过高", DefectType.HARDNESS_HIGH),
    ("裂纹", DefectType.CRACK),
    ("开裂", DefectType.CRACK),
    ("变形", DefectType.DEFORMATION),
    ("翘曲", DefectType.DEFORMATION),
    ("组织异常", DefectType.MICROSTRUCTURE),
    ("金相", DefectType.MICROSTRUCTURE),
    ("相变", DefectType.MICROSTRUCTURE),
]


def _extract_defect_type(state: AgentState, root_cause: str) -> DefectType:
    """从状态多个信号中抽取缺陷类型

    优先级：
    1. state['defect_record'].defect_type（显式传入的缺陷记录）
    2. state['data_result']['defect_history']['records'][0]['defect_type']
    3. root_cause 文本关键词匹配
    4. 兜底：hardness_low（M1 单缺陷场景）

    Args:
        state: LangGraph 全局状态
        root_cause: 根因文本（用于关键词推断）

    Returns:
        DefectType 枚举值
    """
    # 1. 显式缺陷记录
    defect_record = state.get("defect_record")
    if defect_record is not None:
        dr_type = getattr(defect_record, "defect_type", None)
        if dr_type is None and isinstance(defect_record, dict):
            dr_type = defect_record.get("defect_type")
        if dr_type is not None:
            try:
                return DefectType(dr_type) if not isinstance(dr_type, DefectType) else dr_type
            except Exception:
                pass

    # 2. 历史缺陷记录中的类型
    data_result = state.get("data_result") or {}
    defect_history = data_result.get("defect_history") or {}
    records = defect_history.get("records") or []
    if records:
        rec_type = records[0].get("defect_type") if isinstance(records[0], dict) else None
        if rec_type:
            try:
                return DefectType(rec_type)
            except Exception:
                pass

    # 3. root_cause 关键词匹配
    if root_cause:
        text = str(root_cause)
        for keyword, dtype in _DEFECT_KEYWORD_MAP:
            if keyword in text:
                return dtype

    # 4. 兜底
    return DefectType.HARDNESS_LOW


def _format_solution(solution_raw) -> str:
    """把 adjustments（可能是 dict）格式化为可读字符串"""
    if solution_raw is None or solution_raw == "未知":
        return "未知"
    if isinstance(solution_raw, dict):
        if not solution_raw:
            return "无调整项"
        return "; ".join(f"{k} {v}" for k, v in solution_raw.items())
    return str(solution_raw)


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
        solution_raw = proposals[0].get("adjustments", "未知") if proposals else "未知"

        # 如果没有 proposals，从 final_answer 提取
        if not proposals:
            root_cause = state.get("final_answer", "未知")[:200]
            solution_raw = "见最终回答"

        solution = _format_solution(solution_raw)

        # 自动抽取缺陷类型（替代写死的 hardness_low）
        defect_type = _extract_defect_type(state, str(root_cause))
        logger.info(f"[MemoryWriter] 抽取缺陷类型: {defect_type.value}")

        # 1. 写入短期记忆（情景记忆）
        memory.write_episodic(
            batch_id=batch_id,
            defect_type=defect_type.value,
            root_cause=str(root_cause)[:500],
            solution=str(solution)[:500],
            quality_score=0.7,  # 默认评分，后续根据用户反馈更新
        )
        logger.info(f"[MemoryWriter] 短期记忆已写入, batch={batch_id}")

        # 2. 写入长期记忆（语义记忆）
        case = CaseRecord(
            case_id=f"case_{trace_id}",
            defect_type=defect_type,
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
            tags=["auto_generated"],
        )

        success = memory.write_semantic(case)
        if success:
            logger.info(f"[MemoryWriter] 长期记忆已写入, case_id={case.case_id}")
        else:
            logger.warning("[MemoryWriter] 长期记忆写入跳过（Chroma 不可用）")

    except Exception as e:
        logger.error(f"[MemoryWriter] 记忆写入失败: {e}")

    return {}
