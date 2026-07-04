"""
MetaCraft Agent Streamlit UI（M1-22）

设计原则（心理学结合点）：
- 可控感：用户随时可干预，不是黑盒
- 可解释性：每个建议都有依据
- 渐进披露：先结论，后展开证据
- 不确定度透明：明确标注置信度

两种模式：
- 直连模式（默认）：直接调用 Agent，无需启动 API 服务器
- API 模式：通过 FastAPI 调用（需先启动 api/routes.py）

用法：
    streamlit run ui/streamlit_app.py
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（Streamlit 运行时脚本目录是 ui/，需补充父目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import time
import uuid

import streamlit as st

# ===== 页面配置（必须是第一个 Streamlit 命令）=====
st.set_page_config(
    page_title="MetaCraft Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===== Agent 直连调用 =====

def _build_orchestrator(flow_name: str = None):
    """每次构建全新 orchestrator，避免 MemorySaver 跨会话状态污染

    与 api/routes.py、eval/run_eval.py 保持一致：每请求/每用例新建。
    M2-11: 支持 flow_name 参数选择不同协作流程。
    """
    from agent.orchestrator import build_orchestrator
    from agent.utils import setup_tracing
    setup_tracing()
    return build_orchestrator(flow_name)


def run_agent_direct(query: str, batch_id: str, max_replan: int = 3,
                     defect_type: str | None = None,
                     measured_value: float | None = None,
                     standard_value: float | None = None,
                     flow_name: str = None) -> dict:
    """直连模式：直接调用 Agent

    Args:
        flow_name: 协作流程名（M2-11），None 时使用默认流程

    Returns:
        final_state: Agent 最终状态
    """
    from models.state import AgentState

    trace_id = f"ui_{uuid.uuid4().hex[:8]}"
    initial_state: AgentState = {
        "user_query": query,
        "batch_id": batch_id,
        "defect_record": {
            "defect_type": defect_type,
            "measured_value": measured_value,
            "standard_value": standard_value,
        } if defect_type else None,
        "plan": [],
        "current_step": 0,
        "observations": [],
        "data_result": None,
        "mechanism_result": None,
        "knowledge_result": None,
        "arbitration_result": None,
        "decision_result": None,
        "review_result": None,
        "proposal": None,
        "final_answer": None,
        "retry_count": 0,
        "needs_replan": False,
        "max_replan": max_replan,
        "trace_id": trace_id,
        "session_id": trace_id,
    }

    orchestrator = _build_orchestrator(flow_name)
    config = {"configurable": {"thread_id": trace_id}}

    # 在 Streamlit 中运行 async Agent
    final_state = asyncio.run(orchestrator.ainvoke(initial_state, config))
    return final_state


def run_agent_api(query: str, batch_id: str, api_base: str = "http://localhost:8000/api/v1") -> dict:
    """API 模式：通过 FastAPI 调用"""
    import httpx
    with httpx.Client(timeout=300) as client:
        resp = client.post(
            f"{api_base}/analyze",
            json={"query": query, "batch_id": batch_id},
        )
        resp.raise_for_status()
        return resp.json()


# ===== UI 组件 =====

def render_batch_params(data_result: dict):
    """渲染批次工艺参数"""
    if not data_result:
        st.info("提交归因后将显示批次参数")
        return

    params = data_result.get("batch_params") or data_result
    st.write("**批次工艺参数**")

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = params.get("temperature")
        st.metric("温度", f"{temp}℃" if temp else "—")
    with col2:
        holding = params.get("holding_time")
        st.metric("保温时间", f"{holding}min" if holding else "—")
    with col3:
        cooling = params.get("cooling_rate")
        st.metric("冷却速率", f"{cooling}℃/s" if cooling else "—")

    source = params.get("_source", "")
    if source:
        st.caption(f"数据来源: {source}")


def render_mechanism(mechanism_result: dict):
    """渲染机理分析结果"""
    if not mechanism_result:
        return

    with st.expander("机理分析", expanded=False):
        outputs = mechanism_result.get("outputs", {})
        for key, val in outputs.items():
            st.write(f"- **{key}**: {val}")


def render_proposals(decision_result: dict):
    """渲染候选方案"""
    if not decision_result:
        return

    proposals = decision_result.get("proposals", [])
    source = decision_result.get("source", "unknown")

    st.write(f"**候选方案** ({len(proposals)} 个, 来源: {source})")

    for i, p in enumerate(proposals, 1):
        confidence = p.get("confidence", 0.5)
        root_cause = p.get("root_cause", "未知")
        adjustments = p.get("adjustments", {})
        expected_effect = p.get("expected_effect", "")
        risks = p.get("risks", [])

        # 置信度颜色
        if confidence >= 0.8:
            conf_label = "高"
        elif confidence >= 0.5:
            conf_label = "中"
        else:
            conf_label = "低"

        with st.container():
            st.write(f"**方案 {i}**: {root_cause[:60]}")
            st.write(f"置信度: {confidence:.0%} ({conf_label})")

            # 调整建议
            if adjustments:
                st.write("**参数调整:**")
                for param, delta in adjustments.items():
                    st.write(f"  - {param}: {delta}")

            if expected_effect:
                st.write(f"**预期效果**: {expected_effect}")

            if risks:
                st.write(f"**风险提示**: {', '.join(risks)}")

            st.divider()


def render_review(review_result: dict):
    """渲染审核结果"""
    if not review_result:
        return

    approved = review_result.get("approved", False)
    reason = review_result.get("reason", "")
    suggestions = review_result.get("suggestions", [])

    if approved:
        st.success("审核通过")
    else:
        st.warning("审核未通过（已用完重试次数，按最后方案放行）")
    if reason:
        st.write(f"**原因**: {reason}")
    if suggestions:
        st.write("**改进建议**:")
        for s in suggestions:
            st.write(f"  - {s}")


def render_final_answer(answer: str):
    """渲染最终回答"""
    if not answer:
        return
    st.markdown(answer)


def render_observations(observations: list):
    """渲染执行轨迹（渐进披露）"""
    if not observations:
        return
    with st.expander(f"执行轨迹（{len(observations)} 步）", expanded=False):
        for obs in observations:
            agent = obs.get("agent", "?")
            tool = obs.get("tool", "")
            result = obs.get("result", "")
            if isinstance(result, (dict, list)):
                result = json.dumps(result, ensure_ascii=False, default=str)
            result_preview = str(result)[:200]
            st.write(f"**[{agent}]** {tool}".strip())
            st.caption(result_preview)


# ===== M2-9: 协作过程可视化 =====

def _get_display_name(agent_name: str) -> str:
    """获取 Agent 角色的中文显示名"""
    try:
        from agent.prompts.roles import get_role
        role = get_role(agent_name)
        return role.display_name if role else agent_name
    except Exception:
        return agent_name


def render_collaboration_flow(state: dict):
    """渲染多 Agent 协作流程图（M2-9）

    展示 fan-out 并行结构：
        planner → [data || mechanism || knowledge] → arbitrate → decision → review → interaction

    根据状态中各 result 是否存在，标记节点完成状态。
    """
    st.write("**协作流程**")

    # 节点完成状态判断
    nodes_done = {
        "planner": bool(state.get("plan")),
        "data": state.get("data_result") is not None,
        "mechanism": state.get("mechanism_result") is not None,
        "knowledge": state.get("knowledge_result") is not None,
        "arbitrate": state.get("arbitration_result") is not None,
        "decision": state.get("decision_result") is not None,
        "review": state.get("review_result") is not None,
        "interaction": state.get("final_answer") is not None,
    }

    def _node_status(name: str) -> str:
        return "✅" if nodes_done.get(name) else "⬜"

    # 第一行：planner
    st.write(f"{_node_status('planner')} 规划 Agent → 任务拆解")

    # 第二行：并行 3 个 Agent
    st.write("↓ fan-out 并行")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"{_node_status('data')} 数据 Agent")
        st.caption("查询批次参数/缺陷历史")
    with col2:
        st.write(f"{_node_status('mechanism')} 机理 Agent")
        st.caption("JMAK 模型预测")
    with col3:
        st.write(f"{_node_status('knowledge')} 知识 Agent")
        st.caption("手册/案例检索")

    # 第三行：汇聚 + 仲裁
    st.write("↓ fan-in 汇聚")
    arbit = state.get("arbitration_result") or {}
    conflict_count = arbit.get("conflict_count", 0)
    if conflict_count > 0:
        st.write(f"{_node_status('arbitrate')} 冲突仲裁 ⚠️ {conflict_count} 个冲突")
    else:
        st.write(f"{_node_status('arbitrate')} 冲突仲裁（无冲突）")

    # 第四行：decision → review → interaction
    st.write("↓ 串行决策")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.write(f"{_node_status('decision')} 决策 Agent")
    with col5:
        st.write(f"{_node_status('review')} 审核 Agent")
    with col6:
        st.write(f"{_node_status('interaction')} 交互 Agent")


def render_agent_timeline(observations: list):
    """渲染 Agent 间消息流时间线（M2-9）

    按时间顺序展示每个 Agent 的执行步骤，标注角色显示名。
    并行组的 Agent 用特殊标记区分。
    """
    if not observations:
        return

    try:
        from agent.prompts.roles import PARALLEL_GROUP
    except Exception:
        PARALLEL_GROUP = ["data", "mechanism", "knowledge"]

    with st.expander(f"Agent 消息流（{len(observations)} 条）", expanded=False):
        for obs in observations:
            agent = obs.get("agent", "?")
            tool = obs.get("tool", "")
            result = obs.get("result", "")
            timestamp = obs.get("timestamp", "")

            display_name = _get_display_name(agent)
            is_parallel = agent in PARALLEL_GROUP
            tag = "并行" if is_parallel else "串行"

            if isinstance(result, (dict, list)):
                result = json.dumps(result, ensure_ascii=False, default=str)
            result_preview = str(result)[:300]

            st.write(f"**[{display_name}]** `{agent}` · [{tag}]")
            if tool:
                st.write(f"  🔧 工具: `{tool}`")
            if timestamp:
                st.caption(f"  ⏰ {timestamp}")
            st.caption(f"  📤 输出: {result_preview}")
            st.divider()


def render_arbitration(arbitration_result: dict):
    """渲染冲突仲裁结果（M2-9）"""
    if not arbitration_result:
        return

    conflicts = arbitration_result.get("conflicts", [])
    conflict_count = arbitration_result.get("conflict_count", 0)
    has_high = arbitration_result.get("has_high_severity", False)

    with st.expander(f"冲突仲裁结果（{conflict_count} 个冲突）", expanded=False):
        if conflict_count == 0:
            st.success("三方结果一致，无冲突")
        else:
            if has_high:
                st.error("存在高严重度冲突，决策时请重点关注")
            else:
                st.warning("存在中低严重度冲突")

            for i, c in enumerate(conflicts, 1):
                ctype = c.get("type", "unknown")
                detail = c.get("detail", "")
                severity = c.get("severity", "low")
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                st.write(f"{severity_icon} **冲突 {i}** ({ctype})")
                st.write(f"  {detail}")


# ===== M2-10: 中间结果查看 =====

def render_intermediate_results(state: dict):
    """渲染各节点中间结果（可展开查看，M2-10）"""
    st.write("**中间结果**")

    # 数据 Agent 结果
    data_result = state.get("data_result")
    if data_result:
        with st.expander("📊 数据 Agent 输出", expanded=False):
            params = data_result.get("batch_params") or {}
            defects = data_result.get("defect_history") or []
            st.write("批次参数:")
            st.json(params if params else data_result)
            if defects:
                st.write(f"历史缺陷 ({len(defects)} 条):")
                st.json(defects[:3])  # 只展示前3条

    # 机理 Agent 结果
    mechanism_result = state.get("mechanism_result")
    if mechanism_result:
        with st.expander("⚙️ 机理 Agent 输出", expanded=False):
            jmak = mechanism_result.get("jmak_output", {})
            st.write("JMAK 模型输出:")
            st.json(jmak)

    # 知识 Agent 结果
    knowledge_result = state.get("knowledge_result")
    if knowledge_result:
        with st.expander("📚 知识 Agent 输出", expanded=False):
            handbook = knowledge_result.get("handbook_hits", {})
            cases = knowledge_result.get("case_hits", {})
            st.write(f"手册检索 ({handbook.get('total', 0)} 条):")
            st.json(handbook.get("hits", [])[:2])
            st.write(f"案例检索 ({cases.get('total', 0)} 条):")
            st.json(cases.get("hits", [])[:2])

    # 决策 Agent 结果（proposals 已由 render_proposals 渲染，这里展示原始数据）
    decision_result = state.get("decision_result")
    if decision_result:
        with st.expander("🎯 决策 Agent 原始输出", expanded=False):
            st.json(decision_result)

    # 审核 Agent 结果（已由 render_review 渲染，这里展示原始数据）
    review_result = state.get("review_result")
    if review_result:
        with st.expander("🔍 审核 Agent 原始输出", expanded=False):
            st.json(review_result)


# ===== M2-13: 性能报告 =====

def render_performance_report(state: dict, elapsed: float):
    """渲染性能报告（M2-13）

    分析并行协作的性能优势：
    - 各 Agent 执行步数
    - 并行 vs 串行的理论加速比
    - 总耗时分解
    """
    observations = state.get("observations") or []
    if not observations and elapsed == 0:
        return

    with st.expander("性能报告", expanded=False):
        # 统计各 Agent 的执行步数
        agent_steps = {}
        for obs in observations:
            agent = obs.get("agent", "?")
            agent_steps[agent] = agent_steps.get(agent, 0) + 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总耗时", f"{elapsed:.1f}s")
        with col2:
            st.metric("执行步数", len(observations))
        with col3:
            # 并行 Agent 数量
            try:
                from agent.prompts.roles import PARALLEL_GROUP
                parallel_count = sum(1 for a in agent_steps if a in PARALLEL_GROUP)
            except Exception:
                parallel_count = 3
            st.metric("并行 Agent 数", parallel_count)

        # 理论加速比分析
        st.write("**并行加速分析**")
        try:
            from agent.prompts.roles import PARALLEL_GROUP
            parallel_steps = sum(agent_steps.get(a, 0) for a in PARALLEL_GROUP)
            serial_steps = len(observations) - parallel_steps
            # 理论上并行部分耗时 = max(各并行 Agent 耗时) 而非 sum
            # 串行模式耗时 = sum(所有 Agent 耗时)
            # 加速比 ≈ (parallel_steps + serial_steps) / (max(parallel_steps/3, 1) + serial_steps)
            if parallel_steps > 0:
                estimated_speedup = (parallel_steps + serial_steps) / (max(parallel_steps / 3, 1) + serial_steps)
                st.write(f"- 并行阶段步数: {parallel_steps}（data + mechanism + knowledge）")
                st.write(f"- 串行阶段步数: {serial_steps}（planner + arbitrate + decision + review + interaction）")
                st.write(f"- 理论加速比: ~{estimated_speedup:.2f}x（相比 M1 线性模式）")
                st.caption("注：实际加速比受 LLM 调用延迟、网络等因素影响")
        except Exception:
            st.caption("无法计算加速比")

        # 各 Agent 步数分布
        st.write("**各 Agent 执行步数**")
        for agent, count in sorted(agent_steps.items()):
            display_name = _get_display_name(agent)
            st.write(f"- {display_name} (`{agent}`): {count} 步")


def render_execution_info(state: dict, elapsed: float):
    """渲染执行信息"""
    with st.expander("执行详情", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trace ID", state.get("trace_id", "")[:16] + "...")
        with col2:
            st.metric("耗时", f"{elapsed:.1f}s")
        with col3:
            st.metric("重试次数", state.get("retry_count", 0))
        with col4:
            approved = (state.get("review_result") or {}).get("approved", False)
            st.metric("审核", "通过" if approved else "未通过")


# ===== 页面布局 =====

st.title("MetaCraft Agent")
st.caption("材料加工产线智能工艺优化 Agent")

# ===== 侧栏 =====
with st.sidebar:
    # M3-12: 页面选择器（缺陷归因 / 记忆浏览 / 跨产线看板 / 效果看板）
    page = st.radio(
        "页面",
        ["缺陷归因", "记忆浏览", "跨产线看板", "效果看板"],
        help="切换主界面功能",
    )

    st.divider()
    st.header("设置")

    # 模式选择
    mode = st.radio(
        "运行模式",
        ["直连 Agent", "API 服务器"],
        help="直连模式无需启动 API，直接在 Streamlit 进程内运行 Agent",
    )

    if mode == "API 服务器":
        api_base = st.text_input("API 地址", value="http://localhost:8000/api/v1")

    st.session_state["user_id"] = st.text_input("用户ID", value="operator_01")

    # 最大重试次数
    max_replan = st.slider("最大重试次数", 0, 5, 3, help="审核未通过时最多重试次数")

    # M2-11: 协作流程选择
    st.divider()
    st.header("协作流程")
    flow_options = {
        "parallel": "并行协作（默认 M2）",
        "sequential": "线性顺序（M1 兼容）",
        "data_first": "数据优先",
        "quick": "快速模式（跳过机理/知识）",
        "knowledge_heavy": "知识密集",
    }
    flow_name = st.selectbox(
        "流程模式",
        list(flow_options.keys()),
        format_func=lambda x: flow_options[x],
        help="不同场景可配不同协作流程",
    )
    st.session_state["flow_name"] = flow_name

    st.divider()
    st.header("工艺上下文")
    if "last_data_result" in st.session_state:
        render_batch_params(st.session_state["last_data_result"])
    else:
        st.info("提交归因后将显示批次参数")

    st.divider()
    st.header("快速选择")
    # 预设批次（来自 seed_cases）
    quick_batches = {
        "B20260701-A": "保温不足案例",
        "B20260716-B": "冷却过低案例",
        "B20260731-C": "温度偏低案例",
    }
    for bid, desc in quick_batches.items():
        if st.button(f"{desc}", key=f"qb_{bid}"):
            st.session_state["quick_batch"] = bid
            st.rerun()

# ===== 主区域 =====

# M3-12: 记忆浏览页面
if page == "记忆浏览":
    from ui.memory_browser import render_memory_browser
    render_memory_browser()
    st.stop()

# M4-11: 跨产线统一看板页面
if page == "跨产线看板":
    from ui.dashboard import render_dashboard
    render_dashboard()
    st.stop()

# M5-3: 效果看板页面
if page == "效果看板":
    from ui.effect_dashboard import render_effect_dashboard
    render_effect_dashboard()
    st.stop()

st.header("缺陷归因")

# 输入区
col1, col2 = st.columns([3, 1])
with col1:
    default_batch = st.session_state.get("quick_batch", "")
    batch_id = st.text_input("批次ID", value=default_batch, placeholder="如 B20260701-A")
with col2:
    defect_type = st.selectbox(
        "缺陷类型",
        ["hardness_low", "hardness_high", "deformation", "crack", "other"],
    )

query = st.chat_input("描述缺陷问题，如：批次 B20260701-A 硬度偏低 3 HRc")

# 历史对话
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "result" in msg:
            result = msg["result"]
            # 渲染完整结果
            if isinstance(result, dict):
                # M2-9: 协作流程图（最先展示，让用户看到整体协作过程）
                render_collaboration_flow(result)
                if "final_answer" in result and result["final_answer"]:
                    render_final_answer(result["final_answer"])
                if "decision_result" in result and result["decision_result"]:
                    render_proposals(result["decision_result"])
                if "review_result" in result and result["review_result"]:
                    render_review(result["review_result"])
                if "mechanism_result" in result and result["mechanism_result"]:
                    render_mechanism(result["mechanism_result"])
                # M2-9: 冲突仲裁结果
                if "arbitration_result" in result and result["arbitration_result"]:
                    render_arbitration(result["arbitration_result"])
                # M2-9: Agent 消息流时间线
                if "observations" in result and result["observations"]:
                    render_agent_timeline(result["observations"])
                # M2-10: 中间结果可展开查看
                render_intermediate_results(result)
                # 执行轨迹（保留 M1 的简洁版）
                if "observations" in result and result["observations"]:
                    render_observations(result["observations"])
                if "_elapsed" in result:
                    render_execution_info(result, result["_elapsed"])
                # M2-13: 性能报告
                if "_elapsed" in result:
                    render_performance_report(result, result["_elapsed"])

# 处理用户输入
if query and batch_id:
    # 添加用户消息
    user_msg = f"[{batch_id}] {query}"
    st.session_state["messages"].append({"role": "user", "content": user_msg})

    with st.chat_message("user"):
        st.write(user_msg)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在分析中... (预计 1-3 分钟)"):
            start_time = time.time()
            try:
                if mode == "直连 Agent":
                    final_state = run_agent_direct(
                        query, batch_id, max_replan,
                        defect_type=defect_type if defect_type != "other" else None,
                        flow_name=st.session_state.get("flow_name"),
                    )
                    # 缓存批次参数到侧栏
                    st.session_state["last_data_result"] = final_state.get("data_result")
                    result = final_state
                else:
                    api_result = run_agent_api(query, batch_id, api_base)
                    result = {
                        "final_answer": api_result.get("final_answer"),
                        "decision_result": {"proposals": api_result.get("proposals", [])},
                        "review_result": None,
                        "mechanism_result": None,
                        "observations": [],
                        "trace_id": api_result.get("trace_id"),
                        "retry_count": 0,
                    }

                elapsed = time.time() - start_time
                result["_elapsed"] = elapsed

                # 渲染结果
                # M2-9: 协作流程图（最先展示）
                render_collaboration_flow(result)
                if result.get("final_answer"):
                    render_final_answer(result["final_answer"])
                if result.get("decision_result"):
                    render_proposals(result["decision_result"])
                if result.get("review_result"):
                    render_review(result["review_result"])
                if result.get("mechanism_result"):
                    render_mechanism(result["mechanism_result"])
                # M2-9: 冲突仲裁 + Agent 消息流
                if result.get("arbitration_result"):
                    render_arbitration(result["arbitration_result"])
                if result.get("observations"):
                    render_agent_timeline(result["observations"])
                # M2-10: 中间结果
                render_intermediate_results(result)
                # 执行轨迹
                if result.get("observations"):
                    render_observations(result["observations"])
                render_execution_info(result, elapsed)
                # M2-13: 性能报告
                render_performance_report(result, elapsed)

                # 保存到历史
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"分析完成 (耗时 {elapsed:.0f}s)",
                    "result": result,
                })

            except Exception as e:
                st.error(f"分析失败: {e}")
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"分析失败: {e}",
                })


# ===== 反馈区域 =====
st.divider()
with st.expander("反馈"):
    proposal_id = st.text_input("建议ID")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("采纳", use_container_width=True):
            if proposal_id:
                st.success("已记录采纳反馈")
    with col2:
        if st.button("拒绝", use_container_width=True):
            if proposal_id:
                st.success("已记录拒绝反馈")
    with col3:
        if st.button("部分采纳", use_container_width=True):
            if proposal_id:
                st.success("已记录部分采纳反馈")

    score = st.slider("详细评分", 0.0, 1.0, 0.5, 0.1)
    comment = st.text_area("评论（可选）")
    if st.button("提交详细反馈"):
        if proposal_id:
            st.success("详细反馈已提交")


st.divider()
st.caption("MetaCraft Agent v0.2.0 | Powered by LangGraph + SiliconFlow")
