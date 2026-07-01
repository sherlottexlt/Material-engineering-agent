"""
Mechanism Agent 节点：冶金机理分析
对应 TDD 第 4.1 节

基于物理冶金原理，调用 JMAK 等机理模型验证假设，结论必须可证伪。
M2 阶段：支持并行执行，自行查询 batch_params（不依赖 data_result）。
"""
import json
from datetime import datetime

from loguru import logger

from agent.tools import query_batch_params, run_metallurgy_model
from agent.utils import get_llm, get_prompt
from models.state import AgentState


async def mechanism_agent(state: AgentState) -> dict:
    """机理分析 Agent

    调用 JMAK 模型和冷却速率模型，结合机理给出可证伪结论。
    M2: 优先从 data_result 获取 batch_params，若不存在则自行查询（支持并行）。

    Args:
        state: LangGraph 全局状态

    Returns:
        mechanism_result 与 observations
    """
    task = state.get("user_query", "")
    batch_id = state.get("batch_id") or ""
    data_result = state.get("data_result") or {}

    # M2: 优先复用 data_result，否则自行查询（并行模式下 data_result 为 None）
    batch_params = data_result.get("batch_params")
    if batch_params is None:
        batch_params = query_batch_params(batch_id) if batch_id else {}
        logger.info(f"[MechanismAgent] 并行模式：自行查询 batch_params, batch_id={batch_id}")

    temperature = batch_params.get("temperature")
    holding_time = batch_params.get("holding_time")
    cooling_rate = batch_params.get("cooling_rate")

    logger.info(
        f"[MechanismAgent] 开始机理分析, temp={temperature}, "
        f"holding={holding_time}, cooling={cooling_rate}"
    )

    observations = []

    # 1. 调用 JMAK 模型预测硬度
    jmak_output = run_metallurgy_model(
        model_type="jmak",
        temperature=temperature,
        holding_time=holding_time,
    )
    observations.append({
        "step_id": 0,
        "agent": "mechanism",
        "tool": "run_metallurgy_model(jmak)",
        "result": json.dumps(jmak_output, ensure_ascii=False),
        "timestamp": datetime.now().isoformat(),
    })

    # 2. 调用冷却速率模型
    cooling_output = run_metallurgy_model(
        model_type="cooling_rate",
        cooling_rate=cooling_rate,
    )
    observations.append({
        "step_id": 0,
        "agent": "mechanism",
        "tool": "run_metallurgy_model(cooling_rate)",
        "result": json.dumps(cooling_output, ensure_ascii=False),
        "timestamp": datetime.now().isoformat(),
    })

    # 3. LLM 综合机理分析
    hypothesis = "机理分析失败"
    try:
        llm = get_llm("executor")
        prompt = get_prompt("mechanism_agent").format(
            task=task,
            batch_params=json.dumps(batch_params, ensure_ascii=False),
        )
        # 把模型输出也加入 prompt
        prompt += f"\n\nJMAK 模型输出: {json.dumps(jmak_output, ensure_ascii=False)}"
        prompt += f"\n冷却速率模型输出: {json.dumps(cooling_output, ensure_ascii=False)}"
        prompt += "\n\n请基于以上模型输出，分析硬度偏低的机理原因，给出可证伪的假设。"

        response = llm.invoke(prompt)
        hypothesis = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"[MechanismAgent] LLM 分析失败: {e}")
        # 降级：基于模型输出做简单分析
        predicted = jmak_output.get("outputs", {}).get("predicted_hardness_HRc", "未知")
        cooling_est = cooling_output.get("outputs", {}).get("estimated_hardness_HRc", "未知")
        hypothesis = (
            f"JMAK 模型预测硬度: {predicted} HRc\n"
            f"冷却速率模型估计硬度: {cooling_est} HRc\n"
            f"（LLM 分析失败，已降级为纯模型输出）"
        )

    mechanism_result = {
        "hypothesis": hypothesis,
        "jmak_output": jmak_output,
        "cooling_output": cooling_output,
    }

    logger.info("[MechanismAgent] 机理分析完成")

    return {"mechanism_result": mechanism_result, "observations": observations}
