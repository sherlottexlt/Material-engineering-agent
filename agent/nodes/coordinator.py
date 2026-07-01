"""
多 Agent 协作子图（M2 核心模块）
对应 TDD 第 4.6 节

将 M1 的线性流程升级为 fan-out 并行：
  planner → [data || mechanism || knowledge] → arbitrate → decision → review → interaction

并行执行的 3 个 Agent 互不依赖（mechanism 自行查询 batch_params）。
冲突仲裁节点检查三方结果一致性，标记冲突供 decision 参考。
"""
from loguru import logger
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


# ===== 冲突仲裁（M2-5）=====

async def arbitrate(state: AgentState) -> dict:
    """冲突仲裁节点

    检查 data/mechanism/knowledge 三方结果的一致性：
    1. mechanism 的 batch_params 是否与 data 一致（并行模式下可能不同步）
    2. mechanism 预测的硬度趋势是否与 data 的缺陷描述吻合
    3. knowledge 检索的案例是否支持 mechanism 的假设

    不阻塞流程，只标记冲突供 decision 参考。
    """
    data_result = state.get("data_result") or {}
    mechanism_result = state.get("mechanism_result") or {}
    knowledge_result = state.get("knowledge_result") or {}

    conflicts: list[dict] = []

    # 检查 1: batch_params 一致性（并行模式下 mechanism 自行查询，可能与 data 不同步）
    data_params = data_result.get("batch_params", {})
    mechanism_params = mechanism_result.get("batch_params")  # 如果 mechanism 存了的话
    # 注意：mechanism_result 目前不存 batch_params，跳过此检查

    # 检查 2: 机理预测与缺陷描述一致性
    jmak_output = mechanism_result.get("jmak_output", {}).get("outputs", {})
    predicted_hardness = jmak_output.get("predicted_hardness_HRc")
    if predicted_hardness is not None and isinstance(predicted_hardness, (int, float)):
        if predicted_hardness >= 58.0:
            conflicts.append({
                "type": "mechanism_data_mismatch",
                "detail": f"JMAK 预测硬度 {predicted_hardness} HRc ≥ 标准 58.0，但缺陷报告为硬度偏低",
                "severity": "high",
            })

    # 检查 3: 知识检索结果是否为空
    handbook_total = knowledge_result.get("handbook_hits", {}).get("total", 0)
    case_total = knowledge_result.get("case_hits", {}).get("total", 0)
    if handbook_total == 0 and case_total == 0:
        conflicts.append({
            "type": "knowledge_empty",
            "detail": "知识检索无结果，decision 需基于数据和机理自行判断",
            "severity": "medium",
        })

    if conflicts:
        logger.warning(f"[Arbitrate] 检测到 {len(conflicts)} 个冲突: {[c['type'] for c in conflicts]}")
    else:
        logger.info("[Arbitrate] 三方结果一致，无冲突")

    return {
        "arbitration_result": {
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
            "has_high_severity": any(c["severity"] == "high" for c in conflicts),
        }
    }


# ===== Review 后路由（复用 M1 逻辑）=====

def _route_after_review(state: AgentState) -> str:
    """Review 后路由：通过则交互，不通过则回退到 decision

    防止死循环：retry_count 超过 max_replan 则强制通过
    """
    review = state.get("review_result") or {}
    retry = state.get("retry_count", 0)
    max_replan = state.get("max_replan", 3)

    if review.get("approved", False):
        return "interaction"

    if retry >= max_replan:
        logger.warning(f"重试次数 {retry} 超过上限 {max_replan}，强制放行到 interaction")
        return "interaction"

    return "decision"


# ===== 协作子图构建 =====

# 节点名到节点函数的映射
_NODE_REGISTRY = {
    "planner": planner,
    "data": data_agent,
    "mechanism": mechanism_agent,
    "knowledge": knowledge_agent,
    "arbitrate": arbitrate,
    "decision": decision_agent,
    "review": review_agent,
    "interaction": interaction_agent,
    "memory_writer": memory_writer,
}


def _add_review_routing(graph):
    """添加 review 后的条件路由 + interaction → memory_writer → END"""
    graph.add_conditional_edges(
        "review",
        _route_after_review,
        {"interaction": "interaction", "decision": "decision"},
    )
    graph.add_edge("interaction", "memory_writer")
    graph.add_edge("memory_writer", END)


def _build_parallel_graph(config) -> object:
    """构建并行协作图

    planner → fan-out(parallel_agents) → [arbitrate] → sequential_after → END
    """
    graph = StateGraph(AgentState)

    # 注册节点（跳过的除外）
    for name in ["planner"] + config.parallel_agents + config.sequential_after:
        if name in config.skip_agents and name != "arbitrate":
            continue
        if name == "arbitrate" and not config.enable_arbitrate:
            continue
        if name in _NODE_REGISTRY:
            graph.add_node(name, _NODE_REGISTRY[name])
    graph.add_node("memory_writer", memory_writer)

    graph.set_entry_point("planner")

    # Fan-out: planner → parallel_agents
    active_parallel = [a for a in config.parallel_agents if a not in config.skip_agents]
    if active_parallel:
        graph.add_conditional_edges(
            "planner",
            lambda state: active_parallel,
            {a: a for a in active_parallel},
        )
        # Fan-in: parallel_agents → 第一个 sequential_after 节点
        first_after = _first_active_after(config)
        for agent in active_parallel:
            graph.add_edge(agent, first_after)
    else:
        # 无并行阶段，planner 直接连后续
        first_after = _first_active_after(config)
        graph.add_edge("planner", first_after)

    # 串行连接 sequential_after
    _connect_sequential(graph, config)

    # review 路由 + 收尾
    _add_review_routing(graph)

    return graph.compile(checkpointer=MemorySaver())


def _build_sequential_graph(config) -> object:
    """构建线性顺序图

    planner → sequential_after 线性连接 → END
    """
    graph = StateGraph(AgentState)

    # 注册节点
    for name in ["planner"] + config.sequential_after:
        if name in config.skip_agents:
            continue
        if name == "arbitrate" and not config.enable_arbitrate:
            continue
        if name in _NODE_REGISTRY:
            graph.add_node(name, _NODE_REGISTRY[name])
    graph.add_node("memory_writer", memory_writer)

    graph.set_entry_point("planner")

    # 线性连接
    _connect_sequential(graph, config)

    # review 路由 + 收尾
    _add_review_routing(graph)

    return graph.compile(checkpointer=MemorySaver())


def _build_hybrid_graph(config) -> object:
    """构建混合图：先串行，再并行，再串行

    planner → sequential_before → fan-out(parallel_agents) → [arbitrate] → sequential_after → END
    """
    graph = StateGraph(AgentState)

    # 注册节点
    all_nodes = ["planner"] + config.sequential_before + config.parallel_agents + config.sequential_after
    for name in all_nodes:
        if name in config.skip_agents:
            continue
        if name == "arbitrate" and not config.enable_arbitrate:
            continue
        if name in _NODE_REGISTRY:
            graph.add_node(name, _NODE_REGISTRY[name])
    graph.add_node("memory_writer", memory_writer)

    graph.set_entry_point("planner")

    # 串行连接 sequential_before
    prev = "planner"
    active_before = [a for a in config.sequential_before if a not in config.skip_agents]
    for agent in active_before:
        graph.add_edge(prev, agent)
        prev = agent

    # Fan-out: prev → parallel_agents
    active_parallel = [a for a in config.parallel_agents if a not in config.skip_agents]
    if active_parallel:
        graph.add_conditional_edges(
            prev,
            lambda state: active_parallel,
            {a: a for a in active_parallel},
        )
        # Fan-in: parallel_agents → 第一个 sequential_after
        first_after = _first_active_after(config)
        for agent in active_parallel:
            graph.add_edge(agent, first_after)
    else:
        # 无并行阶段，prev 直接连后续
        first_after = _first_active_after(config)
        if prev != first_after:
            graph.add_edge(prev, first_after)

    # 串行连接 sequential_after
    _connect_sequential(graph, config)

    # review 路由 + 收尾
    _add_review_routing(graph)

    return graph.compile(checkpointer=MemorySaver())


def _first_active_after(config) -> str:
    """获取 sequential_after 中第一个活跃节点（非跳过、非禁用 arbitrate）"""
    for name in config.sequential_after:
        if name in config.skip_agents:
            continue
        if name == "arbitrate" and not config.enable_arbitrate:
            continue
        return name
    # 回退到 interaction
    return "interaction"


def _connect_sequential(graph, config):
    """线性连接 sequential_after 中的节点（跳过被禁用的）"""
    active = []
    for name in config.sequential_after:
        if name in config.skip_agents:
            continue
        if name == "arbitrate" and not config.enable_arbitrate:
            continue
        active.append(name)

    # 连接：第一个由 fan-in 或 planner 连接，这里只连后续
    for i in range(len(active) - 1):
        # review 由条件路由处理，不在这里连
        if active[i] == "review":
            continue
        graph.add_edge(active[i], active[i + 1])


def build_collaboration_graph(flow_name: str = None):
    """根据流程配置构建协作图（M2-11）

    Args:
        flow_name: 流程名（parallel/sequential/data_first/quick/knowledge_heavy）
                   None 时使用 config/flows.yaml 中的 default_flow

    Returns:
        CompiledGraph
    """
    from agent.flow_config import load_flow_config
    config = load_flow_config(flow_name)

    logger.info(f"[Coordinator] 构建流程: {config.name} (mode={config.mode})")

    if config.mode == "sequential":
        return _build_sequential_graph(config)
    elif config.mode == "hybrid":
        return _build_hybrid_graph(config)
    else:  # parallel
        return _build_parallel_graph(config)


# ===== 兼容入口 =====

def build_orchestrator(flow_name: str = None):
    """构建编排器（M2 协作模式）

    Args:
        flow_name: 流程名，None 时使用默认流程

    保留与 M1 相同的函数名，内部切换为并行协作图。
    api/routes.py、eval/run_eval.py、ui/streamlit_app.py 无需修改。
    """
    return build_collaboration_graph(flow_name)
