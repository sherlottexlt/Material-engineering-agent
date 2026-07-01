"""
MetaCraft Agent 记忆浏览界面（M3-12）

展示 Agent 当前"记得"什么，回答记忆系统是否在正确积累/检索的问题。
直连 MemoryService，无需启动 API 服务器。

用法（被 streamlit_app.py 引入）：
    from ui.memory_browser import render_memory_browser
    render_memory_browser()
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from agent.memory.memory_service import MemoryService


@st.cache_resource(show_spinner=False)
def _get_memory_service() -> MemoryService:
    """获取 MemoryService 单例（Streamlit 缓存，避免每次交互重建）"""
    return MemoryService()


def _fmt_datetime(value) -> str:
    """格式化时间戳为可读字符串"""
    if not value:
        return "—"
    try:
        s = str(value)
        # 截取到分钟（YYYY-MM-DD HH:MM:SS → YYYY-MM-DD HH:MM）
        return s[:16] if len(s) >= 16 else s
    except Exception:
        return str(value)


def _fmt_confidence(value) -> str:
    """格式化置信度为百分比"""
    if value is None:
        return "—"
    try:
        return f"{float(value):.0%}"
    except Exception:
        return str(value)


def _render_overview(stats: dict):
    """渲染概览页：metric 卡片 + 分布图"""
    episodic_count = stats.get("episodic_count", 0)
    feedback_count = stats.get("feedback_count", 0)
    semantic_count = stats.get("semantic_count", 0)
    avg_confidence = stats.get("avg_confidence", 0.0)

    # 三层记忆总数
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("短期记忆（Episodic）", f"{episodic_count} 条")
    with col2:
        st.metric("长期记忆（Semantic）", f"{semantic_count} 条")
    with col3:
        st.metric("用户反馈", f"{feedback_count} 条")
    with col4:
        st.metric("平均置信度", f"{avg_confidence:.0%}")

    st.divider()

    # 缺陷类型分布 + 反馈动作分布
    left, right = st.columns(2)

    with left:
        st.write("**短期记忆缺陷类型分布**")
        defect_dist = stats.get("defect_type_distribution", {})
        if defect_dist:
            # 用 st.bar_chart 简单可视化
            st.bar_chart(defect_dist)
            # 同时用表格展示精确数值
            st.dataframe(
                [{"缺陷类型": k, "数量": v} for k, v in defect_dist.items()],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("暂无短期记忆数据")

    with right:
        st.write("**用户反馈动作分布**")
        action_dist = stats.get("action_distribution", {})
        if action_dist:
            st.bar_chart(action_dist)
            st.dataframe(
                [{"动作": k, "数量": v} for k, v in action_dist.items()],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("暂无用户反馈数据")

    st.divider()

    # 最近 5 条短期记忆
    st.write("**最近 5 条短期记忆**")
    recent_episodic = stats.get("recent_episodic", [])
    if recent_episodic:
        st.dataframe(
            [
                {
                    "批次": r.get("batch_id", ""),
                    "缺陷类型": r.get("defect_type", ""),
                    "根因": (r.get("root_cause", "") or "")[:50],
                    "质量评分": _fmt_confidence(r.get("quality_score")),
                    "创建时间": _fmt_datetime(r.get("created_at")),
                }
                for r in recent_episodic
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("暂无短期记忆数据")


def _render_episodic_table(records: list[dict]):
    """渲染短期记忆表格"""
    if not records:
        st.info("暂无短期记忆数据。完成一次归因后会自动写入。")
        return

    # 过滤器
    col1, col2 = st.columns(2)
    with col1:
        defect_types = sorted({r.get("defect_type", "") for r in records if r.get("defect_type")})
        selected_type = st.selectbox(
            "按缺陷类型过滤",
            options=["全部"] + defect_types,
            key="episodic_filter_type",
        )
    with col2:
        batch_search = st.text_input(
            "按批次 ID 搜索",
            value="",
            key="episodic_search_batch",
        )

    filtered = records
    if selected_type != "全部":
        filtered = [r for r in filtered if r.get("defect_type") == selected_type]
    if batch_search:
        batch_lower = batch_search.lower()
        filtered = [r for r in filtered if batch_lower in (r.get("batch_id", "") or "").lower()]

    st.write(f"**短期记忆**（{len(filtered)}/{len(records)} 条）")

    # 表格展示
    st.dataframe(
        [
            {
                "批次": r.get("batch_id", ""),
                "缺陷类型": r.get("defect_type", ""),
                "根因": r.get("root_cause", ""),
                "解决方案": (r.get("solution", "") or "")[:80],
                "质量评分": _fmt_confidence(r.get("quality_score")),
                "创建时间": _fmt_datetime(r.get("created_at")),
                "记录ID": r.get("record_id", ""),
            }
            for r in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_semantic_table(records: list[dict]):
    """渲染长期记忆表格（Chroma 案例库）"""
    if not records:
        st.info("暂无长期记忆数据。Chroma 不可用或案例库为空。")
        return

    # 搜索框
    search = st.text_input(
        "按文档内容搜索",
        value="",
        key="semantic_search",
        help="在案例文档中搜索关键词（不区分大小写）",
    )

    filtered = records
    if search:
        search_lower = search.lower()
        filtered = [
            r for r in records
            if search_lower in (r.get("document", "") or "").lower()
            or search_lower in (r.get("id", "") or "").lower()
        ]

    st.write(f"**长期记忆**（{len(filtered)}/{len(records)} 条）")

    # 表格展示
    st.dataframe(
        [
            {
                "案例ID": r.get("id", ""),
                "缺陷类型": (r.get("metadata") or {}).get("defect_type", ""),
                "置信度": _fmt_confidence((r.get("metadata") or {}).get("confidence")),
                "来源": (r.get("metadata") or {}).get("source", ""),
                "创建时间": _fmt_datetime((r.get("metadata") or {}).get("created_at")),
                "文档预览": (r.get("document", "") or "")[:100],
            }
            for r in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )

    # 语义检索测试
    st.divider()
    st.write("**语义检索测试**（验证长期记忆的检索效果）")
    query = st.text_input(
        "输入查询文本，看 Agent 能检索到哪些相似案例",
        value="",
        key="semantic_query_test",
    )
    if query:
        memory = _get_memory_service()
        with st.spinner("检索中..."):
            results = memory.search_semantic(query, top_k=5)
        if results:
            st.success(f"检索到 {len(results)} 条相似案例")
            for i, r in enumerate(results, 1):
                with st.expander(f"#{i} 相似度: {1 - (r.get('distance') or 0):.2f} - {r.get('id', '')}"):
                    st.write(f"**文档:** {r.get('document', '')}")
                    st.write(f"**元数据:** {r.get('metadata', {})}")
        else:
            st.warning("未检索到任何案例（可能 Chroma 不可用或案例库为空）")


def _render_feedback_table(records: list[dict]):
    """渲染用户反馈表格"""
    if not records:
        st.info("暂无用户反馈数据。在归因结果页提交反馈后会自动写入。")
        return

    # 过滤器
    actions = sorted({r.get("action", "") for r in records if r.get("action")})
    selected_action = st.selectbox(
        "按动作过滤",
        options=["全部"] + actions,
        key="feedback_filter_action",
    )

    filtered = records
    if selected_action != "全部":
        filtered = [r for r in filtered if r.get("action") == selected_action]

    st.write(f"**用户反馈**（{len(filtered)}/{len(records)} 条）")

    st.dataframe(
        [
            {
                "反馈ID": r.get("feedback_id", ""),
                "建议ID": r.get("proposal_id", ""),
                "用户": r.get("user_id", ""),
                "动作": r.get("action", ""),
                "评分": _fmt_confidence(r.get("score")),
                "评论": (r.get("comment", "") or "")[:80],
                "创建时间": _fmt_datetime(r.get("created_at")),
            }
            for r in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_memory_browser():
    """渲染记忆浏览界面（M3-12 主入口）

    四个 Tab：概览 / 短期记忆 / 长期记忆 / 用户反馈
    直连 MemoryService，无需启动 API 服务器。
    """
    st.subheader("🧠 记忆浏览")
    st.caption("展示 Agent 当前\"记得\"什么（短期记忆 / 长期记忆 / 用户反馈）")

    try:
        memory = _get_memory_service()
    except Exception as e:
        st.error(f"初始化记忆服务失败: {e}")
        return

    # 四个 Tab
    tab1, tab2, tab3, tab4 = st.tabs(["📊 概览", "📝 短期记忆", "💾 长期记忆", "💬 用户反馈"])

    with tab1:
        try:
            stats = memory.get_memory_stats()
            _render_overview(stats)
        except Exception as e:
            st.error(f"加载统计概览失败: {e}")

    with tab2:
        try:
            records = memory.list_all_episodic(limit=200)
            _render_episodic_table(records)
        except Exception as e:
            st.error(f"加载短期记忆失败: {e}")

    with tab3:
        try:
            records = memory.list_all_semantic(limit=200)
            _render_semantic_table(records)
        except Exception as e:
            st.error(f"加载长期记忆失败: {e}")

    with tab4:
        try:
            records = memory.list_all_feedback(limit=200)
            _render_feedback_table(records)
        except Exception as e:
            st.error(f"加载用户反馈失败: {e}")
