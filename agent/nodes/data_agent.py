"""
Data Agent 节点：数据查询
对应 TDD 第 4.1 节

严谨、只陈述事实的 Agent，查询批次工艺参数与历史缺陷，不做推断。
M0 阶段直接调用 agent.tools 中的函数（M1 切换为 MCP）。
"""
import json
from datetime import datetime

from loguru import logger

from agent.tools import query_batch_params, query_defect_history
from models.state import AgentState


async def data_agent(state: AgentState) -> dict:
    """数据查询 Agent

    调用工具查询批次参数与缺陷历史，输出纯事实。

    Args:
        state: LangGraph 全局状态

    Returns:
        data_result 与 observations
    """
    batch_id = state.get("batch_id") or "未知批次"
    logger.info(f"[DataAgent] 开始查询, batch_id={batch_id}")

    observations = []

    # 1. 查询批次工艺参数
    batch_params = query_batch_params(batch_id)
    observations.append({
        "step_id": 0,
        "agent": "data",
        "tool": "query_batch_params",
        "result": json.dumps(batch_params, ensure_ascii=False),
        "timestamp": datetime.now().isoformat(),
    })

    # 2. 查询历史缺陷
    defect_history = query_defect_history(defect_type="hardness_low", days_back=30)
    observations.append({
        "step_id": 0,
        "agent": "data",
        "tool": "query_defect_history",
        "result": json.dumps(defect_history, ensure_ascii=False),
        "timestamp": datetime.now().isoformat(),
    })

    # 3. 生成事实摘要（纯数据，不做推断）
    temp = batch_params.get("temperature", "未知")
    holding = batch_params.get("holding_time", "未知")
    cooling = batch_params.get("cooling_rate", "未知")

    summary = (
        f"批次 {batch_id} 工艺参数：\n"
        f"  - 温度: {temp}℃\n"
        f"  - 保温时间: {holding} 分钟\n"
        f"  - 冷却速率: {cooling}℃/s\n"
        f"历史缺陷记录: {defect_history['total']} 条硬度偏低案例\n"
        f"  - 案例1: {defect_history['records'][0]['root_cause'] if defect_history['records'] else '无'}\n"
        f"  - 案例2: {defect_history['records'][1]['root_cause'] if len(defect_history['records']) > 1 else '无'}"
    )

    data_result = {
        "batch_params": batch_params,
        "defect_history": defect_history,
        "summary": summary,
    }

    logger.info(f"[DataAgent] 查询完成, 历史缺陷 {defect_history['total']} 条")

    return {"data_result": data_result, "observations": observations}
