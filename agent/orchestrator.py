"""
MetaCraft Agent 主编排器
对应 TDD 第 4.1 节

M1 阶段：线性流程 planner → data → mechanism → knowledge → decision → review → interaction
M2 阶段：并行协作 planner → [data || mechanism || knowledge] → arbitrate → decision → review → interaction

通过 build_orchestrator() 切换模式，默认 M2 并行协作。
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.nodes import (
    data_agent,
    decision_agent,
    interaction_agent,
    knowledge_agent,
    mechanism_agent,
    memory_writer,
    planner,
    review_agent,
)
from models.state import AgentState


# ===== M1 线性流程（保留作为回退）=====

def _route_after_review(state: AgentState) -> str:
    """Review 后路由：通过则交互，不通过则回退到 decision"""
    review = state.get("review_result") or {}
    retry = state.get("retry_count", 0)
    max_replan = state.get("max_replan", 3)

    if review.get("approved", False):
        return "interaction"

    if retry >= max_replan:
        from loguru import logger
        logger.warning(f"重试次数 {retry} 超过上限 {max_replan}，强制放行到 interaction")
        return "interaction"

    return "decision"


def build_linear_orchestrator():
    """构建 M1 线性编排图（顺序执行）

    planner → data → mechanism → knowledge → decision → review
    review → (interaction | decision 回退)
    interaction → memory_writer → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("data", data_agent)
    graph.add_node("mechanism", mechanism_agent)
    graph.add_node("knowledge", knowledge_agent)
    graph.add_node("decision", decision_agent)
    graph.add_node("review", review_agent)
    graph.add_node("interaction", interaction_agent)
    graph.add_node("memory_writer", memory_writer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "data")
    graph.add_edge("data", "mechanism")
    graph.add_edge("mechanism", "knowledge")
    graph.add_edge("knowledge", "decision")
    graph.add_edge("decision", "review")

    graph.add_conditional_edges(
        "review",
        _route_after_review,
        {"interaction": "interaction", "decision": "decision"},
    )

    graph.add_edge("interaction", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=MemorySaver())


# ===== M2 并行协作（默认）=====

def build_orchestrator(flow_name: str = None):
    """构建编排器（默认 M2 并行协作模式）

    Args:
        flow_name: 流程名（M2-11）
                   - None: 使用 config/flows.yaml 中的 default_flow（默认 parallel）
                   - "parallel": M2 并行协作
                   - "sequential": M1 线性兼容
                   - "data_first": 数据优先
                   - "quick": 快速模式（跳过 mechanism/knowledge）
                   - "knowledge_heavy": 知识密集

    M2 流程（parallel）：
        planner → fan-out → [data || mechanism || knowledge]
                             ↓ 汇聚
                          arbitrate → decision → review
                                                       ↓
                                        ┌── interaction (通过/超限)
                                        └── decision (回退)
                          interaction → memory_writer → END

    如需 M1 原始线性模式，调用 build_linear_orchestrator()。
    """
    from agent.nodes.coordinator import build_collaboration_graph
    return build_collaboration_graph(flow_name)


# 全局单例（延迟初始化）
_orchestrator = None


def get_orchestrator():
    """获取编排器单例"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator
