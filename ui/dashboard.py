"""
MetaCraft Agent 跨产线统一看板（M4-11）

3 产线数据汇总看板，展示各产线 KPI 对比、缺陷分布、反馈采纳率等。
直连 MemoryService + agent.utils，无需启动 API 服务器。

用法（被 streamlit_app.py 引入）：
    from ui.dashboard import render_dashboard
    render_dashboard()
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from agent.memory.memory_service import MemoryService
from agent.utils import (
    get_user_lines,
    get_user_permissions,
    list_available_lines,
    load_line_config,
)


@st.cache_resource(show_spinner=False)
def _get_memory_service() -> MemoryService:
    """获取 MemoryService 单例"""
    return MemoryService()


def _safe_pct(value: float) -> str:
    """格式化百分数"""
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "—"


def _render_global_kpi(totals: dict, line_count: int):
    """渲染全局 KPI 卡片区"""
    st.subheader("全局概览")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("可见产线数", line_count)
    col2.metric("总案例数（近N天）", totals.get("total_episodic", 0))
    col3.metric("总反馈数", totals.get("total_feedback", 0))
    col4.metric("总冲突数", totals.get("total_conflicts", 0))
    col5.metric("总采纳率", _safe_pct(totals.get("overall_adoption_rate", 0)))


def _render_line_comparison(lines: list[dict]):
    """渲染各产线对比表格"""
    st.subheader("产线对比")
    if not lines:
        st.info("暂无产线数据")
        return

    rows = []
    for line in lines:
        rows.append({
            "产线": line.get("name", line.get("line_id", "")),
            "line_id": line.get("line_id", ""),
            "案例数": line.get("episodic_count", 0),
            "长期案例": line.get("semantic_count", 0),
            "反馈数": line.get("feedback_count", 0),
            "冲突数": line.get("conflict_count", 0),
            "采纳率": _safe_pct(line.get("adoption_rate", 0)),
            "平均置信度": f"{line.get('avg_confidence', 0):.3f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_defect_distribution(lines: list[dict]):
    """渲染各产线缺陷类型分布对比图"""
    st.subheader("缺陷类型分布对比")
    if not lines:
        st.info("暂无缺陷数据")
        return

    # 收集所有缺陷类型
    all_defects: set[str] = set()
    for line in lines:
        all_defects.update((line.get("defect_distribution") or {}).keys())
    if not all_defects:
        st.info("暂无缺陷分布数据")
        return

    # 构建对比 DataFrame（行=产线，列=缺陷类型）
    rows = {}
    for line in lines:
        line_name = line.get("name", line.get("line_id", ""))
        dist = line.get("defect_distribution") or {}
        rows[line_name] = {d: dist.get(d, 0) for d in all_defects}
    df = pd.DataFrame(rows).T.fillna(0)
    st.bar_chart(df, use_container_width=True)
    with st.expander("查看明细数据"):
        st.dataframe(df, use_container_width=True)


def _render_action_distribution(lines: list[dict]):
    """渲染各产线反馈动作分布"""
    st.subheader("反馈动作分布")
    if not lines:
        return

    all_actions: set[str] = set()
    for line in lines:
        all_actions.update((line.get("action_distribution") or {}).keys())
    if not all_actions:
        st.info("暂无反馈数据")
        return

    rows = {}
    for line in lines:
        line_name = line.get("name", line.get("line_id", ""))
        dist = line.get("action_distribution") or {}
        rows[line_name] = {a: dist.get(a, 0) for a in all_actions}
    df = pd.DataFrame(rows).T.fillna(0)
    st.bar_chart(df, use_container_width=True)


def _render_single_line_detail(line_stats: dict, line_cfg: dict):
    """渲染单产线详情面板"""
    line_id = line_stats.get("line_id", "")
    name = line_stats.get("name", line_id)
    st.divider()
    st.subheader(f"产线详情：{name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("短期案例数", line_stats.get("episodic_count", 0))
    col2.metric("长期案例数", line_stats.get("semantic_count", 0))
    col3.metric("反馈数", line_stats.get("feedback_count", 0))
    col4.metric("冲突数", line_stats.get("conflict_count", 0))

    # 产线配置信息
    with st.expander("产线配置"):
        cfg = line_cfg or {}
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**材料**：", cfg.get("material", "—"))
            st.write("**工艺类型**：", cfg.get("process_type", "—"))
            st.write("**工艺名称**：", cfg.get("process_name", "—"))
        with col_b:
            std_params = cfg.get("standard_params", {})
            if std_params:
                st.write("**标准参数**：")
                for k, v in std_params.items():
                    st.write(f"- {k}: {v}")
            else:
                st.write("**标准参数**：—")

    # 缺陷分布
    defect_dist = line_stats.get("defect_distribution") or {}
    if defect_dist:
        st.write("**缺陷类型分布**")
        defect_df = pd.DataFrame(
            [{"缺陷类型": k, "数量": v} for k, v in defect_dist.items()]
        )
        st.bar_chart(defect_df.set_index("缺陷类型"), use_container_width=True)

    # 反馈分布
    action_dist = line_stats.get("action_distribution") or {}
    if action_dist:
        st.write("**反馈动作分布**")
        action_df = pd.DataFrame(
            [{"动作": k, "数量": v} for k, v in action_dist.items()]
        )
        st.bar_chart(action_df.set_index("动作"), use_container_width=True)


def render_dashboard():
    """M4-11: 渲染跨产线统一看板"""
    st.header("跨产线统一看板")
    st.caption("M4-11: 多产线数据汇总与对比分析")

    memory = _get_memory_service()

    # ===== 顶部控制栏 =====
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_id = st.text_input("用户ID", value="admin", help="不同用户可见产线范围不同")
    with col2:
        days = st.slider("统计天数", 7, 365, 30, help="统计近 N 天数据")
    with col3:
        perms = get_user_permissions(user_id)
        st.write("**角色**")
        st.code(perms["role"])

    # ===== 权限过滤 =====
    user_lines = get_user_lines(user_id)
    all_lines = list_available_lines()
    if "*" in user_lines:
        visible_lines = all_lines
    else:
        visible_lines = [lid for lid in all_lines if lid in user_lines]

    if not visible_lines:
        st.warning(f"用户 {user_id} 无权访问任何产线")
        return

    st.info(f"可见产线：{', '.join(visible_lines)}")

    # ===== 聚合各产线统计 =====
    line_stats_list = []
    total_episodic = 0
    total_feedback = 0
    total_conflicts = 0
    total_semantic = 0
    total_adopted = 0
    confidence_sum = 0.0
    confidence_count = 0

    for lid in visible_lines:
        stats = memory.get_line_stats(lid, days=days)
        cfg = load_line_config(lid)
        stats["name"] = cfg.get("name", lid)
        stats["_cfg"] = cfg  # 附带配置供详情面板使用
        line_stats_list.append(stats)

        total_episodic += stats["episodic_count"]
        total_feedback += stats["feedback_count"]
        total_conflicts += stats["conflict_count"]
        total_semantic += stats["semantic_count"]
        total_adopted += stats["action_distribution"].get("adopted", 0)
        if stats["semantic_count"] > 0:
            confidence_sum += stats["avg_confidence"] * stats["semantic_count"]
            confidence_count += stats["semantic_count"]

    overall_adoption = (
        round(total_adopted / total_feedback, 3) if total_feedback > 0 else 0.0
    )
    overall_confidence = (
        round(confidence_sum / confidence_count, 3) if confidence_count > 0 else 0.0
    )

    totals = {
        "total_episodic": total_episodic,
        "total_feedback": total_feedback,
        "total_conflicts": total_conflicts,
        "total_semantic": total_semantic,
        "overall_adoption_rate": overall_adoption,
        "overall_avg_confidence": overall_confidence,
    }

    # ===== 渲染各区域 =====
    _render_global_kpi(totals, len(visible_lines))
    st.divider()
    _render_line_comparison(line_stats_list)
    st.divider()
    _render_defect_distribution(line_stats_list)
    st.divider()
    _render_action_distribution(line_stats_list)

    # ===== 单产线详情选择 =====
    st.divider()
    st.subheader("产线详情查看")
    line_options = {f"{s['name']} ({s['line_id']})": s for s in line_stats_list}
    selected_key = st.selectbox("选择产线", list(line_options.keys()))
    if selected_key:
        selected_stats = line_options[selected_key]
        _render_single_line_detail(selected_stats, selected_stats.pop("_cfg", {}))
