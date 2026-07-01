"""
Knowledge Agent 节点：知识检索
对应 TDD 第 4.1 节

检索工艺手册与历史案例，引用来源，不臆造知识。
M0 阶段直接调用 agent.tools 中的检索函数。
"""
import json
from datetime import datetime

from loguru import logger

from agent.tools import search_cases, search_handbook
from models.state import AgentState


async def knowledge_agent(state: AgentState) -> dict:
    """知识检索 Agent

    调用工具检索手册与案例，输出带来源引用的知识片段。

    Args:
        state: LangGraph 全局状态

    Returns:
        knowledge_result 与 observations
    """
    query = state.get("user_query", "")
    # 从缺陷记录中提取关键词
    search_query = "硬度偏低 保温时间 冷却速率"

    logger.info(f"[KnowledgeAgent] 开始知识检索, query={search_query}")

    observations = []

    # 1. 检索工艺手册
    handbook_hits = search_handbook(search_query, top_k=3)
    observations.append({
        "step_id": 0,
        "agent": "knowledge",
        "tool": "search_handbook",
        "result": json.dumps(handbook_hits, ensure_ascii=False),
        "timestamp": datetime.now().isoformat(),
    })

    # 2. 检索历史案例
    case_hits = search_cases(search_query, top_k=3)
    observations.append({
        "step_id": 0,
        "agent": "knowledge",
        "tool": "search_cases",
        "result": json.dumps(case_hits, ensure_ascii=False),
        "timestamp": datetime.now().isoformat(),
    })

    # 3. 生成带引用的知识摘要
    handbook_summary = "\n".join(
        f"  - [{h['source']}]: {h['content'][:100]}..."
        for h in handbook_hits.get("results", [])
    )
    case_summary = "\n".join(
        f"  - 案例 {c.get('batch_id', '未知')}: {c.get('root_cause', '')} → {c.get('solution', '')}"
        for c in case_hits.get("results", [])
    )

    summary = (
        f"工艺手册检索结果 ({handbook_hits.get('total', 0)} 条):\n{handbook_summary}\n\n"
        f"历史案例检索结果 ({case_hits.get('total', 0)} 条):\n{case_summary}"
    )

    knowledge_result = {
        "handbook_hits": handbook_hits,
        "case_hits": case_hits,
        "summary": summary,
    }

    logger.info(
        f"[KnowledgeAgent] 检索完成, 手册 {handbook_hits.get('total', 0)} 条, "
        f"案例 {case_hits.get('total', 0)} 条"
    )

    return {"knowledge_result": knowledge_result, "observations": observations}
