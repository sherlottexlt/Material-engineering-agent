"""
Executor 节点：子任务执行
对应 TDD 第 4.1 节路由

按 plan 顺序读取当前子任务，调用对应 MCP 工具（若指定）并记录观察。
"""
from datetime import datetime

from loguru import logger

from agent.utils import get_llm, get_prompt
from models.state import AgentState


async def executor(state: AgentState) -> dict:
    """子任务执行节点

    读取 plan[current_step]，若指定 tool 则调用 MCP 工具，记录观察并推进步数。

    Args:
        state: LangGraph 全局状态

    Returns:
        推进 current_step 并追加 observations
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        logger.warning(
            f"[Executor] current_step={current_step} 超出 plan 长度 {len(plan)}, 跳过"
        )
        return {"current_step": current_step, "observations": []}

    task = plan[current_step]
    agent = task.get("agent", "unknown")
    tool = task.get("tool")
    action = task.get("action", "")

    logger.info(
        f"[Executor] 执行 step={current_step}, agent={agent}, tool={tool}, action={action}"
    )

    result = action
    if tool:
        # TODO: 实现实际 MCP 工具调用（根据 tool 名分发到对应 MCP server）
        logger.debug(f"[Executor] 预留 MCP 调用: tool={tool}")
        result = f"[TODO] 调用 {tool} 的结果占位"

    observation = {
        "step_id": task.get("step_id", current_step),
        "agent": agent,
        "tool": tool,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "current_step": current_step + 1,
        "observations": [observation],
    }
