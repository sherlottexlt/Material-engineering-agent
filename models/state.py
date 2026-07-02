"""
MetaCraft Agent 状态定义
对应 TDD 第 3.2 节
LangGraph 全局状态，贯穿整个 Agent 执行流程
"""
from typing import Annotated, Optional, TypedDict

import operator

from models.entities import (
    AdjustmentProposal,
    DefectRecord,
)


class SubTask(TypedDict):
    """子任务定义"""
    step_id: int
    agent: str          # data / mechanism / knowledge / decision / end
    action: str         # 任务描述
    tool: Optional[str] # 调用的工具名
    status: str         # pending / running / done / failed


class Observation(TypedDict):
    """单步观察记录"""
    step_id: int
    agent: str
    tool: Optional[str]
    result: str
    timestamp: str


class AgentState(TypedDict):
    """LangGraph 全局状态

    所有节点共享此状态，通过 Annotated[list, operator.add] 实现累积式字段
    """
    # ===== 输入 =====
    user_query: str
    batch_id: Optional[str]
    defect_record: Optional[DefectRecord]
    line_id: str  # M4-9: 产线ID，默认 "heat_treatment"

    # ===== 任务规划 =====
    plan: list[SubTask]
    current_step: int

    # ===== 中间结果（累积式）=====
    observations: Annotated[list[Observation], operator.add]

    # ===== 各 Sub-Agent 输出 =====
    data_result: Optional[dict]
    mechanism_result: Optional[dict]
    knowledge_result: Optional[dict]
    arbitration_result: Optional[dict]  # M2: 冲突仲裁结果
    decision_result: Optional[dict]
    review_result: Optional[dict]

    # ===== 最终输出 =====
    proposal: Optional[AdjustmentProposal]
    final_answer: Optional[str]

    # ===== 控制流 =====
    retry_count: int
    needs_replan: bool
    max_replan: int

    # ===== 可观测 =====
    trace_id: str
    session_id: str
