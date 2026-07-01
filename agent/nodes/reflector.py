"""
Reflector 节点：自我反思
对应 TDD 第 4.1 节

评估当前执行进度，判断是否需要重新规划。
"""
import json

from loguru import logger

from agent.utils import get_llm, get_prompt
from models.state import AgentState


async def reflector(state: AgentState) -> dict:
    """反思节点

    评估是否需要重新规划，必要时递增 retry_count。

    Args:
        state: LangGraph 全局状态

    Returns:
        needs_replan 与可能的 retry_count 更新
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    observations = state.get("observations", [])

    logger.info(
        f"[Reflector] 评估进度 step={current_step}, observations={len(observations)}"
    )

    needs_replan = False
    try:
        llm = get_llm("planner")
        prompt_template = get_prompt("reflector")
        prompt = prompt_template.format(
            plan=plan,
            current_step=current_step,
            observations=observations,
        )

        # TODO: 切换为 llm.ainvoke() 异步调用
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        parsed = json.loads(content)
        needs_replan = bool(parsed.get("needs_replan", False))
        logger.info(
            f"[Reflector] needs_replan={needs_replan}, reason={parsed.get('reason')}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"[Reflector] LLM 输出 JSON 解析失败: {e}")
    except Exception as e:
        logger.error(f"[Reflector] 反思失败: {e}")

    update: dict = {"needs_replan": needs_replan}
    if needs_replan:
        update["retry_count"] = state.get("retry_count", 0) + 1
    return update
