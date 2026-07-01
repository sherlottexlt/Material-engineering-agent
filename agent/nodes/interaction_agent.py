"""
交互 Agent 节点
对应 TDD 第 4.5 节

职责：把技术结论翻译成操作员易懂的语言
- 先结论，后展开证据（渐进披露）
- 标注置信度（高/中/低）
- 提供确认/拒绝选项
- LLM 不可用时降级为格式化输出
"""
import asyncio
import json
from datetime import datetime

from loguru import logger

from agent.utils import get_llm, get_prompt
from models.state import AgentState


def _format_proposals_fallback(decision_result: dict, review_result: dict) -> str:
    """降级方案：LLM 不可用时格式化输出 proposals"""
    proposals = decision_result.get("proposals", [])
    review_approved = review_result.get("approved", True)
    review_reason = review_result.get("reason", "")

    if not proposals:
        return "未生成可用的参数调整建议，建议人工复核。"

    lines = ["## 归因结果与参数调整建议\n"]
    if not review_approved:
        lines.append(f"> ⚠️ 审核提示：{review_reason}\n")

    for i, p in enumerate(proposals, 1):
        confidence = p.get("confidence", 0.5)
        level = "高" if confidence >= 0.7 else "中" if confidence >= 0.5 else "低"

        lines.append(f"### 方案 {i}：{p.get('root_cause', '未知原因')}")
        lines.append(f"**置信度**：{level} ({confidence:.0%})\n")

        adjustments = p.get("adjustments", {})
        if adjustments:
            lines.append("**参数调整**：")
            for k, v in adjustments.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("**参数调整**：无（建议人工检查）")

        lines.append(f"\n**预期效果**：{p.get('expected_effect', '未知')}")

        risks = p.get("risks", [])
        if risks:
            lines.append("**风险提示**：")
            for r in risks:
                lines.append(f"  - {r}")

        evidence = p.get("evidence", [])
        if evidence:
            lines.append("**证据链**：")
            for e in evidence:
                lines.append(f"  - {e}")

        lines.append("")

    lines.append("---")
    lines.append("请选择：✅ 采纳 / ❌ 拒绝 / ⚠️ 部分采纳")

    return "\n".join(lines)


async def interaction_agent(state: AgentState) -> dict:
    """交互 Agent：生成面向操作员的最终回答

    Args:
        state: 全局状态，含 decision_result 和 review_result

    Returns:
        更新 final_answer 字段
    """
    logger.info("交互 Agent 启动：生成面向用户的回答")

    proposal = state.get("decision_result") or {}
    review = state.get("review_result") or {}

    if not proposal or not proposal.get("proposals"):
        logger.warning("无 proposals 可用于生成交互回答")
        return {"final_answer": "暂无可用的参数调整建议，建议人工复核。"}

    try:
        llm = get_llm("interaction")
        prompt_template = get_prompt("interaction_agent")
        prompt = prompt_template.format(
            proposal=json.dumps(proposal, ensure_ascii=False, default=str)
        )

        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=120)
        final_answer = response.content if hasattr(response, "content") else str(response)

        logger.info("交互 Agent 完成：LLM 生成最终回答")

        observation = {
            "step_id": -1,
            "agent": "interaction",
            "tool": None,
            "result": final_answer[:200] + "...",
            "timestamp": datetime.now().isoformat(),
        }

        return {
            "final_answer": final_answer,
            "observations": [observation],
        }
    except Exception as e:
        logger.warning(f"交互 Agent LLM 失败，降级为格式化输出: {e}")
        final_answer = _format_proposals_fallback(proposal, review)

        observation = {
            "step_id": -1,
            "agent": "interaction",
            "tool": None,
            "result": final_answer[:200] + "...",
            "timestamp": datetime.now().isoformat(),
        }

        return {
            "final_answer": final_answer,
            "observations": [observation],
        }
