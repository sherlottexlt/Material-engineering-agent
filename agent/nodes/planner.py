"""
Planner 节点：任务拆解
对应 TDD 第 4.2 节

将用户问题拆解为 3-5 个有序子任务，作为后续 Sub-Agent 执行的蓝图。
LLM 不可用时降级为默认 plan，确保流程不中断。
"""
import json

from loguru import logger

from agent.utils import extract_json, get_llm, get_prompt
from models.state import AgentState


def _default_plan(batch_id: str) -> list:
    """降级方案：LLM 不可用时用默认 plan"""
    return [
        {"step_id": 1, "agent": "data", "action": f"查询批次 {batch_id} 工艺参数", "tool": "query_batch_params", "status": "pending"},
        {"step_id": 2, "agent": "mechanism", "action": "调用 JMAK 模型验证硬度假设", "tool": "run_metallurgy_model", "status": "pending"},
        {"step_id": 3, "agent": "knowledge", "action": "检索工艺手册和历史案例", "tool": "search_handbook", "status": "pending"},
        {"step_id": 4, "agent": "decision", "action": "综合生成参数调整方案", "tool": None, "status": "pending"},
    ]


async def planner(state: AgentState) -> dict:
    """任务规划节点

    通过 LLM 将用户原始查询拆解为可独立执行、可验证的子任务列表。
    LLM 不可用时降级为默认 plan。

    Args:
        state: LangGraph 全局状态

    Returns:
        包含 plan / current_step / retry_count 的部分状态更新
    """
    query = state.get("user_query", "")
    batch_id = state.get("batch_id") or "未知批次"

    logger.info(f"[Planner] 开始规划任务, query={query}, batch_id={batch_id}")

    plan: list = []
    try:
        llm = get_llm("planner")
        prompt_template = get_prompt("planner")
        prompt = prompt_template.format(query=query, batch_id=batch_id)

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        parsed = extract_json(content)
        if parsed and "plan" in parsed:
            plan = parsed["plan"]
            logger.info(f"[Planner] 规划完成, 共 {len(plan)} 个子任务")
        else:
            logger.warning("[Planner] LLM 输出 JSON 解析失败，降级为默认 plan")
            plan = _default_plan(batch_id)
    except Exception as e:
        logger.warning(f"[Planner] LLM 规划失败，降级为默认 plan: {e}")
        plan = _default_plan(batch_id)

    return {
        "plan": plan,
        "current_step": 0,
        "retry_count": state.get("retry_count", 0),
    }
