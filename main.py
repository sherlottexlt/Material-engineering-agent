"""
MetaCraft Agent - M0 Hello World 入口

这是 M0 阶段的主入口脚本，跑通 Agent 端到端流程：
planner → data → mechanism → knowledge → decision → review → interaction → memory_writer

运行方式：
    python main.py
    python main.py --batch-id B20260628-A
    python main.py --query "批次 B20260628-A 硬度偏低 5 HRc" --batch-id B20260628-A

无需 LLM API key 也能运行（自动降级为规则方案）。
配置 .env 中的 API key 后，Agent 会使用 LLM 增强。
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from agent.orchestrator import get_orchestrator
from agent.utils import setup_tracing
from models.state import AgentState


def setup_logging():
    """配置日志格式"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    logger.add(
        "data/metacraft.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
        level="DEBUG",
        rotation="10 MB",
    )


def create_initial_state(query: str, batch_id: str) -> AgentState:
    """构造 Agent 初始状态"""
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    return {
        "user_query": query,
        "batch_id": batch_id,
        "defect_record": None,
        "plan": [],
        "current_step": 0,
        "observations": [],
        "data_result": None,
        "mechanism_result": None,
        "knowledge_result": None,
        "decision_result": None,
        "review_result": None,
        "proposal": None,
        "final_answer": None,
        "retry_count": 0,
        "needs_replan": False,
        "max_replan": 3,
        "trace_id": trace_id,
        "session_id": trace_id,
    }


def print_separator(title: str):
    """打印分隔符"""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def print_result(final_state: dict):
    """打印最终结果"""
    print_separator("归因结果")

    # 最终回答
    answer = final_state.get("final_answer", "无结果")
    print(answer)

    # 执行摘要
    print_separator("执行摘要")
    trace_id = final_state.get("trace_id", "unknown")
    observations = final_state.get("observations", [])
    plan = final_state.get("plan", [])
    retry = final_state.get("retry_count", 0)
    decision = final_state.get("decision_result") or {}
    review = final_state.get("review_result") or {}

    print(f"Trace ID:    {trace_id}")
    print(f"规划步数:    {len(plan)}")
    print(f"观察记录:    {len(observations)} 条")
    print(f"重试次数:    {retry}")
    print(f"方案来源:    {decision.get('source', 'unknown')}")
    print(f"方案数量:    {len(decision.get('proposals', []))}")
    print(f"审核结果:    {'通过' if review.get('approved') else '未通过'}")
    print(f"  审核说明:  {review.get('reason', '无')}")

    # 各 Agent 关键输出
    print_separator("各 Agent 输出")

    data_result = final_state.get("data_result") or {}
    if data_result.get("summary"):
        print("【Data Agent 摘要】")
        print(data_result["summary"])
        print()

    mechanism_result = final_state.get("mechanism_result") or {}
    hypothesis = mechanism_result.get("hypothesis") or ""
    if hypothesis:
        print("【Mechanism Agent 假设】")
        print(hypothesis[:500] + ("..." if len(hypothesis) > 500 else ""))
        print()

    knowledge_result = final_state.get("knowledge_result") or {}
    if knowledge_result.get("summary"):
        print("【Knowledge Agent 摘要】")
        print(knowledge_result["summary"])
        print()

    # 观察记录
    print_separator("观察记录")
    for obs in observations:
        agent = str(obs.get("agent") or "?")
        tool = str(obs.get("tool") or "-")
        result = str(obs.get("result") or "")
        result_short = result[:120] + ("..." if len(result) > 120 else "")
        print(f"  [{agent:12s}] {tool:30s} → {result_short}")


async def run_agent(query: str, batch_id: str):
    """运行 Agent"""
    # 1. 初始化
    setup_logging()
    setup_tracing()

    print_separator("MetaCraft Agent 启动")
    print(f"  查询: {query}")
    print(f"  批次: {batch_id}")

    # 2. 构造初始状态
    initial_state = create_initial_state(query, batch_id)

    # 3. 执行 Agent
    print_separator("开始执行 Agent 流程")
    print("  planner → data → mechanism → knowledge → decision → review → interaction → memory_writer")
    print()

    orchestrator = get_orchestrator()
    config = {"configurable": {"thread_id": initial_state["trace_id"]}}

    try:
        final_state = await orchestrator.ainvoke(initial_state, config)
        print_result(final_state)
        return final_state
    except Exception as e:
        logger.error(f"Agent 执行失败: {e}")
        print(f"\n❌ Agent 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="MetaCraft Agent - M0 Hello World")
    parser.add_argument(
        "--query",
        type=str,
        default="批次 B20260628-A 硬度偏低 5 HRc，请分析原因并给出调整建议",
        help="归因查询",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default="B20260628-A",
        help="批次ID",
    )
    args = parser.parse_args()

    # 运行 Agent
    result = asyncio.run(run_agent(args.query, args.batch_id))

    if result:
        print_separator("执行完成")
        print("✅ Agent 端到端流程跑通！")
        print(f"\n日志文件: data/metacraft.log")
        print(f"数据库:   data/metacraft.db（短期记忆已写入）")
    else:
        print_separator("执行失败")
        print("❌ Agent 执行失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
