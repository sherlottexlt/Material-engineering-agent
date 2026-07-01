"""
Decision Agent 节点：决策综合
对应 TDD 第 4.1 节

综合数据、机理、知识三方信息，输出至少 2 个候选调整方案，按可行性×置信度排序。
LLM 不可用时降级为规则方案，确保流程不中断。

M2-13 性能优化：上下文摘要减少 token 消耗。
"""
import asyncio
import json

from loguru import logger

from agent.utils import extract_json, get_llm, get_prompt, get_process_constraints
from models.state import AgentState


def _summarize_for_decision(data_result: dict, mechanism_result: dict, knowledge_result: dict) -> tuple[str, str, str]:
    """为 decision agent 生成精简上下文（M2-13 token 节省）

    只传递关键信息，避免完整 JSON 传入 LLM prompt。
    健壮处理 None / dict / list 等各种数据类型。

    Returns:
        (data_summary, mechanism_summary, knowledge_summary)
    """
    # 数据摘要：只保留关键工艺参数
    params = data_result.get("batch_params") or {}
    if not isinstance(params, dict):
        params = {}
    defects = data_result.get("defect_history") or []
    # defect_history 可能是 dict 而非 list，统一转为 list
    if isinstance(defects, dict):
        defects = list(defects.values())
    elif not isinstance(defects, list):
        defects = []
    data_summary = json.dumps({
        "batch_params": {
            "temperature": params.get("temperature"),
            "holding_time": params.get("holding_time"),
            "cooling_rate": params.get("cooling_rate"),
        },
        "defect_count": len(defects),
        "recent_defects": [
            {"type": d.get("defect_type") if isinstance(d, dict) else str(d),
             "batch_id": d.get("batch_id") if isinstance(d, dict) else ""}
            for d in defects[:3]  # 只取最近 3 条
        ],
    }, ensure_ascii=False, default=str)

    # 机理摘要：只保留模型预测输出
    jmak_raw = mechanism_result.get("jmak_output")
    jmak = jmak_raw.get("outputs", {}) if isinstance(jmak_raw, dict) else {}
    cooling_raw = mechanism_result.get("cooling_output")
    cooling = cooling_raw.get("outputs", {}) if isinstance(cooling_raw, dict) else {}
    mechanism_summary = json.dumps({
        "jmak_prediction": jmak,
        "cooling_analysis": cooling,
    }, ensure_ascii=False, default=str)

    # 知识摘要：只保留前 2 条检索结果的标题
    handbook = knowledge_result.get("handbook_hits") or {}
    if not isinstance(handbook, dict):
        handbook = {}
    cases = knowledge_result.get("case_hits") or {}
    if not isinstance(cases, dict):
        cases = {}
    # hits 可能是 None / dict / list，统一确保为 list
    handbook_hits_raw = handbook.get("hits")
    if isinstance(handbook_hits_raw, list):
        handbook_hits = handbook_hits_raw[:2]
    else:
        handbook_hits = []
    case_hits_raw = cases.get("hits")
    if isinstance(case_hits_raw, list):
        case_hits = case_hits_raw[:2]
    else:
        case_hits = []
    knowledge_summary = json.dumps({
        "handbook_total": handbook.get("total", 0),
        "handbook_top": [
            {"title": h.get("title", "") if isinstance(h, dict) else str(h),
             "snippet": str(h.get("content", ""))[:100] if isinstance(h, dict) else ""}
            for h in handbook_hits
        ],
        "case_total": cases.get("total", 0),
        "case_top": [
            {"id": c.get("id", "") if isinstance(c, dict) else str(c),
             "root_cause": c.get("root_cause", "") if isinstance(c, dict) else ""}
            for c in case_hits
        ],
    }, ensure_ascii=False, default=str)

    return data_summary, mechanism_summary, knowledge_summary


def _rule_based_proposals(data_result: dict, mechanism_result: dict) -> dict:
    """规则降级方案：LLM 不可用时基于规则生成

    根据工艺参数与机理模型输出，用简单规则生成候选方案
    """
    batch_params = data_result.get("batch_params") or {}
    jmak = mechanism_result.get("jmak_output", {}).get("outputs", {})
    cooling = mechanism_result.get("cooling_output", {}).get("outputs", {})

    proposals = []
    temperature = batch_params.get("temperature", 850)
    holding_time = batch_params.get("holding_time", 120)
    cooling_rate = batch_params.get("cooling_rate", 5.0)

    # 方案1：保温时间不足
    if holding_time < 120:
        proposals.append({
            "proposal_id": "P001",
            "root_cause": "保温时间不足",
            "adjustments": {"holding_time": f"+{120 - holding_time + 15} 分钟"},
            "expected_effect": f"保温时间从 {holding_time} 提升至 {holding_time + 120 - holding_time + 15} 分钟",
            "risks": ["可能增加能耗", "需确认设备产能"],
            "evidence": [f"当前保温时间 {holding_time} 分钟 < 标准 120 分钟"],
            "confidence": 0.75,
        })

    # 方案2：冷却速率过低
    if cooling_rate < 5.0:
        proposals.append({
            "proposal_id": "P002",
            "root_cause": "冷却速率过低",
            "adjustments": {"cooling_rate": f"提升至 {cooling_rate + 5.0} ℃/s"},
            "expected_effect": f"冷却速率从 {cooling_rate} 提升至 {cooling_rate + 5.0} ℃/s",
            "risks": ["可能导致变形风险增加"],
            "evidence": [f"当前冷却速率 {cooling_rate} ℃/s < 标准 5.0 ℃/s"],
            "confidence": 0.70,
        })

    # 方案3：温度偏低
    if temperature < 840:
        proposals.append({
            "proposal_id": "P003",
            "root_cause": "淬火温度偏低",
            "adjustments": {"temperature": f"+{840 - temperature + 10} ℃"},
            "expected_effect": f"温度从 {temperature} 提升至 {temperature + 840 - temperature + 10} ℃",
            "risks": ["需确认设备上限", "可能影响晶粒度"],
            "evidence": [f"当前温度 {temperature}℃ < 标准 840℃"],
            "confidence": 0.65,
        })

    # 如果没有生成方案，给一个默认方案
    if not proposals:
        proposals.append({
            "proposal_id": "P000",
            "root_cause": "参数边界情况，建议人工复核",
            "adjustments": {},
            "expected_effect": "人工检查全部工艺参数",
            "risks": ["无明显异常，可能为材料批次差异"],
            "evidence": ["所有参数均在标准范围内"],
            "confidence": 0.4,
        })

    return {"proposals": proposals, "source": "rule_based"}


async def decision_agent(state: AgentState) -> dict:
    """决策综合 Agent

    综合三方结果与工艺约束，生成排序的候选方案列表。
    LLM 不可用时降级为规则方案。

    Args:
        state: LangGraph 全局状态

    Returns:
        decision_result
    """
    data_result = state.get("data_result") or {}
    mechanism_result = state.get("mechanism_result") or {}
    knowledge_result = state.get("knowledge_result") or {}
    review_result = state.get("review_result") or {}
    arbitration_result = state.get("arbitration_result") or {}
    retry_count = state.get("retry_count", 0)
    constraints = get_process_constraints("heat_treatment")

    logger.info(f"[DecisionAgent] 开始决策综合 (第 {retry_count + 1} 次)")

    decision_result = {"proposals": []}
    try:
        llm = get_llm("planner")
        # M2-13: 使用摘要而非完整 JSON，减少 token 消耗
        data_summary, mechanism_summary, knowledge_summary = _summarize_for_decision(
            data_result, mechanism_result, knowledge_result
        )
        # M2 迭代优化：加入 arbitrate 冲突信息供 LLM 参考
        # M2 修复：同时传入 param_status，帮助 LLM 判断哪些参数真的偏离标准
        arbitration_summary = "无冲突"
        if isinstance(arbitration_result, dict):
            conflicts = arbitration_result.get("conflicts") or []
            param_status = arbitration_result.get("param_status") or {}
            parts = []
            if isinstance(conflicts, list) and conflicts:
                parts.append("冲突列表: " + json.dumps([
                    {"type": c.get("type", ""), "detail": c.get("detail", ""), "severity": c.get("severity", "")}
                    for c in conflicts if isinstance(c, dict)
                ], ensure_ascii=False))
            if isinstance(param_status, dict) and param_status:
                parts.append("参数偏离分析（权威，请严格参考）: " + json.dumps(param_status, ensure_ascii=False))
            if parts:
                arbitration_summary = "\n".join(parts)
        prompt = get_prompt("decision_agent").format(
            data_result=data_summary,
            mechanism_result=mechanism_summary,
            knowledge_result=knowledge_summary,
            constraints=constraints,
            arbitration_result=arbitration_summary,
        )

        # 重试时把上次审核反馈传给 LLM，让它针对性改进
        if retry_count > 0 and review_result:
            feedback = (
                f"\n\n【上次审核未通过，请改进】\n"
                f"不通过原因：{review_result.get('reason', '未知')}\n"
                f"改进建议：{json.dumps(review_result.get('suggestions', []), ensure_ascii=False)}"
            )
            prompt += feedback

        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=120)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            # 用 extract_json 支持 ```json 代码块包裹的输出
            parsed = extract_json(content)
            if parsed and "proposals" in parsed:
                decision_result = parsed
                decision_result["source"] = "llm"
            elif parsed:
                # JSON 解析成功但没有 proposals 字段，尝试从 raw 内容提取
                decision_result = {"proposals": [], "raw": content, "source": "llm"}
                logger.warning("[DecisionAgent] LLM 输出 JSON 但无 proposals 字段")
            else:
                # JSON 解析失败，降级为规则方案
                logger.warning(
                    "[DecisionAgent] LLM 输出非 JSON，降级为规则方案。原始输出前 300 字符: "
                    f"{content[:300]!r}"
                )
                decision_result = _rule_based_proposals(data_result, mechanism_result)
        except Exception as parse_err:
            logger.warning(f"[DecisionAgent] JSON 解析异常，降级为规则方案: {parse_err}")
            decision_result = _rule_based_proposals(data_result, mechanism_result)

        logger.info(
            f"[DecisionAgent] 决策完成, 方案数={len(decision_result.get('proposals', []))}, "
            f"来源={decision_result.get('source', 'unknown')}"
        )
    except Exception as e:
        logger.error(f"[DecisionAgent] LLM 决策失败，降级为规则方案: {e}")
        decision_result = _rule_based_proposals(data_result, mechanism_result)
        logger.info(
            f"[DecisionAgent] 规则方案生成, 方案数={len(decision_result.get('proposals', []))}"
        )

    return {"decision_result": decision_result}
