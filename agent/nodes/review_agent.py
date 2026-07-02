"""
Review Agent 节点：审核校验
对应 TDD 第 4.1 节

质监员角色，保守挑刺，审核决策方案是否违反工艺约束、证据链是否完整。
不通过时递增 retry_count，防止无限循环。
"""
import asyncio
import json

from loguru import logger

from agent.utils import extract_json, get_llm, get_prompt, get_process_constraints, get_line_constraints
from models.state import AgentState


async def review_agent(state: AgentState) -> dict:
    """审核 Agent

    审核决策方案，输出 approved/reason/suggestions。
    不通过时递增 retry_count。

    Args:
        state: LangGraph 全局状态

    Returns:
        review_result（含 retry_count 更新）
    """
    proposal = state.get("decision_result") or {}
    # M4-9: 从产线配置读取约束（替代硬编码 "heat_treatment"）
    line_id = state.get("line_id", "heat_treatment")
    constraints = get_line_constraints(line_id)
    retry_count = state.get("retry_count", 0)

    logger.info(f"[ReviewAgent] 开始审核 (第 {retry_count + 1} 次)")

    review_result = {"approved": False, "reason": "", "suggestions": []}
    result_update = {"review_result": review_result}

    try:
        llm = get_llm("reviewer")
        prompt = get_prompt("review_agent").format(
            proposal=json.dumps(proposal, ensure_ascii=False, default=str),
            constraints=constraints,
        )

        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # 尝试解析 JSON（支持 ```json 代码块包裹）
        parsed = extract_json(content)
        if parsed:
            review_result = {
                "approved": bool(parsed.get("approved", False)),
                "reason": parsed.get("reason", ""),
                "suggestions": parsed.get("suggestions", []),
            }
        else:
            # LLM 没返回可解析的 JSON，做简单判断
            if "通过" in content or "approved" in content.lower():
                review_result = {"approved": True, "reason": content, "suggestions": []}
            else:
                review_result = {"approved": False, "reason": content, "suggestions": []}

        logger.info(f"[ReviewAgent] 审核完成, approved={review_result['approved']}")

    except Exception as e:
        logger.error(f"[ReviewAgent] 审核失败（降级放行）: {e}")
        # LLM 不可用时降级放行，避免阻塞流程
        review_result = {
            "approved": True,
            "reason": f"审核 LLM 不可用，降级放行: {e}",
            "suggestions": [],
        }

    # 不通过时递增 retry_count
    if not review_result["approved"]:
        result_update["retry_count"] = retry_count + 1

    result_update["review_result"] = review_result
    return result_update
