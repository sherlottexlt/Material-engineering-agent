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

# M3-6: 根因关键词 → 标准化标签（中文根因 + 英文参数名）
# 支持 6 类根因（3 单根因 + 3 双根因组合），与 prompts.yaml 标准一致
_ROOT_CAUSE_TAG_MAP = [
    ("保温时间不足", ["保温时间不足", "holding_time"]),
    ("时间不够", ["保温时间不足", "holding_time"]),
    ("时间不足", ["保温时间不足", "holding_time"]),
    ("冷却速率过低", ["冷却速率过低", "cooling_rate"]),
    ("冷却不足", ["冷却速率过低", "cooling_rate"]),
    ("温度偏低", ["温度偏低", "temperature"]),
    ("温度不足", ["温度偏低", "temperature"]),
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


def _extract_root_cause_tags(root_cause: str) -> list[str]:
    """M3-6: 从根因文本抽取标准化标签

    支持多根因组合（"+" 分隔），与 prompts.yaml 标准根因命名一致。
    用于自动生成案例 tags，提升后续检索精度。

    Args:
        root_cause: 根因文本（如 "保温时间不足+冷却速率过低"）

    Returns:
        去重排序的标签列表（如 ["holding_time", "保温时间不足",
        "cooling_rate", "冷却速率过低"]）
    """
    if not root_cause:
        return []

    tags: set[str] = set()
    text = str(root_cause).replace("＋", "+")
    # 按 + 分割多根因组合（如 "保温时间不足+冷却速率过低"）
    parts = [p.strip() for p in text.split("+")]

    for part in parts:
        for pattern, labels in _ROOT_CAUSE_TAG_MAP:
            if pattern in part:
                tags.update(labels)
                break

    return sorted(tags)


def _assess_confidence(
    state: AgentState,
    root_cause: str,
    batch_params: dict,
    proposals: list,
) -> float:
    """M3-6: 基于多信号评估案例初始置信度

    评估维度（累计加分，上限 0.9）：
    1. 基础值 0.3
    2. 数据完整性：每个关键工艺参数 +0.1（最多 +0.3）
    3. 归因明确性：root_cause 含标准根因关键词 +0.2
    4. 方案存在性：有 proposals（非空）+0.1
    5. 缺陷记录：有 defect_record +0.1

    Args:
        state: LangGraph 全局状态
        root_cause: 根因文本
        batch_params: 批次工艺参数
        proposals: 调整建议列表

    Returns:
        置信度 0.3-0.9
    """
    confidence = 0.3  # 基础值

    # 1. 数据完整性（每个关键参数 +0.1，最多 +0.3）
    for key in ("temperature", "holding_time", "cooling_rate"):
        if batch_params.get(key) is not None:
            confidence += 0.1

    # 2. 归因明确性（包含标准根因关键词 +0.2）
    if root_cause:
        text = str(root_cause)
        standard_causes = ("保温时间不足", "冷却速率过低", "温度偏低")
        if any(cause in text for cause in standard_causes):
            confidence += 0.2

    # 3. 方案存在性（有 proposals +0.1）
    if proposals:
        confidence += 0.1

    # 4. 缺陷记录（有 defect_record +0.1）
    if state.get("defect_record") is not None:
        confidence += 0.1

    return min(confidence, 0.9)


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

        # M3-6: 自动结构化 — 根因标签 + 置信度评估
        root_cause_tags = _extract_root_cause_tags(str(root_cause))
        confidence = _assess_confidence(state, str(root_cause), batch_params, proposals)
        logger.info(
            f"[MemoryWriter] 结构化: tags={root_cause_tags}, confidence={confidence:.2f}"
        )

        # 1. 写入短期记忆（情景记忆）
        memory.write_episodic(
            batch_id=batch_id,
            defect_type=defect_type.value,
            root_cause=str(root_cause)[:500],
            solution=str(solution)[:500],
            quality_score=confidence,  # M3-6: 用评估值替代写死的 0.7
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
            confidence=confidence,  # M3-6: 用评估值替代写死的 0.7
            created_at=datetime.now(),
            source="auto",
            tags=["auto_generated"] + root_cause_tags,  # M3-6: 自动根因标签
        )

        success = memory.write_semantic(case)
        if success:
            logger.info(f"[MemoryWriter] 长期记忆已写入, case_id={case.case_id}")
        else:
            logger.warning("[MemoryWriter] 长期记忆写入跳过（Chroma 不可用）")

    except Exception as e:
        logger.error(f"[MemoryWriter] 记忆写入失败: {e}")

    return {}
