"""
MetaCraft Agent 效果看板（M5-3）

调参效果跟踪的可视化看板：
- KPI 概览（总跟踪/已跟踪/待跟踪/平均改善/归因数）
- 改善率分布图
- 调整前后缺陷率对比
- 跟踪记录表格
- 归因结果（confidence 变化）

直连 EffectTracker + MemoryService，无需启动 API 服务器。

用法（被 streamlit_app.py 引入）：
    from ui.effect_dashboard import render_effect_dashboard
    render_effect_dashboard()
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from agent.effect_tracker import EffectTracker
from agent.memory.memory_service import MemoryService
from agent.utils import get_user_lines, get_user_permissions, list_available_lines


@st.cache_resource(show_spinner=False)
def _get_memory_service() -> MemoryService:
    return MemoryService()


@st.cache_resource(show_spinner=False)
def _get_effect_tracker() -> EffectTracker:
    return EffectTracker(_get_memory_service())


def _fmt_pct(value, suffix="%") -> str:
    """格式化百分数"""
    try:
        return f"{float(value):.1f}{suffix}"
    except Exception:
        return "—"


def _render_kpi(stats: dict, attributed_count: int):
    """渲染 KPI 卡片区"""
    st.subheader("效果跟踪概览")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("总跟踪数", stats.get("total", 0))
    col2.metric("已跟踪", stats.get("tracked", 0))
    col3.metric("待跟踪", stats.get("pending", 0))
    col4.metric("平均改善", _fmt_pct(stats.get("avg_improvement", 0)))
    col5.metric("正向/负向", f"{stats.get('positive_count', 0)} / {stats.get('negative_count', 0)}")
    col6.metric("已归因", attributed_count)


def _render_improvement_distribution(records: list[dict]):
    """渲染改善率分布图"""
    st.subheader("改善率分布")
    tracked = [r for r in records if r.get("improvement_pct") is not None]
    if not tracked:
        st.info("暂无已跟踪记录")
        return

    # 分桶：<=-10, -10~0, 0~10, 10~20, 20~30, >=30
    bins = [-1000, -10, 0, 10, 20, 30, 10000]
    labels = ["≤-10%（反效果）", "-10~0%（无改善）", "0~10%（微改善）",
              "10~20%（有效）", "20~30%（显著）", "≥30%（卓越）"]
    improvements = [r["improvement_pct"] for r in tracked]
    df = pd.DataFrame({"improvement_pct": improvements})
    df["区间"] = pd.cut(df["improvement_pct"], bins=bins, labels=labels, right=False)
    counts = df["区间"].value_counts().reindex(labels, fill_value=0)

    chart_df = pd.DataFrame({"跟踪数": counts.values}, index=counts.index)
    st.bar_chart(chart_df, use_container_width=True)
    with st.expander("查看明细"):
        st.dataframe(chart_df, use_container_width=True)


def _render_before_after_comparison(records: list[dict]):
    """渲染调整前后缺陷率对比图"""
    st.subheader("调整前后缺陷率对比")
    tracked = [
        r for r in records
        if r.get("metric_before") is not None and r.get("metric_after") is not None
    ]
    if not tracked:
        st.info("暂无对比数据")
        return

    rows = []
    for r in tracked[:50]:  # 最多展示 50 条避免图表过密
        rows.append({
            "tracking_id": r["tracking_id"][:12] + "...",
            "调参前": r["metric_before"],
            "调参后": r["metric_after"],
            "改善%": r.get("improvement_pct", 0),
        })
    df = pd.DataFrame(rows).set_index("tracking_id")
    st.bar_chart(df[["调参前", "调参后"]], use_container_width=True)
    with st.expander("查看明细数据"):
        st.dataframe(df, use_container_width=True)


def _render_attribution_results(records: list[dict]):
    """渲染归因结果（confidence 变化）"""
    st.subheader("归因结果（confidence 变化）")
    attributed = []
    for r in records:
        if r.get("attribution_done") == 1 and r.get("attribution_result"):
            try:
                attr = json.loads(r["attribution_result"]) if isinstance(r["attribution_result"], str) else r["attribution_result"]
                attributed.append(attr)
            except (json.JSONDecodeError, TypeError):
                continue

    if not attributed:
        st.info("暂无归因记录（需先执行 attribute_effect）")
        return

    rows = []
    for attr in attributed:
        old_c = attr.get("old_confidence")
        new_c = attr.get("new_confidence")
        delta = (new_c - old_c) if (old_c is not None and new_c is not None) else None
        rows.append({
            "tracking_id": attr.get("tracking_id", "")[:12] + "...",
            "case_id": attr.get("case_id", "")[:12] + "...",
            "改善%": attr.get("improvement_pct", 0),
            "效果分": attr.get("effect_score"),
            "旧 confidence": old_c,
            "新 confidence": new_c,
            "变化": delta,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # confidence 变化柱状图
    if len(df) > 0:
        chart_df = df[["旧 confidence", "新 confidence"]].set_index(df["tracking_id"])
        st.bar_chart(chart_df, use_container_width=True)


def _render_records_table(records: list[dict]):
    """渲染跟踪记录表格"""
    st.subheader("跟踪记录明细")
    if not records:
        st.info("暂无跟踪记录")
        return

    rows = []
    for r in records:
        rows.append({
            "tracking_id": r["tracking_id"],
            "proposal_id": r.get("proposal_id", ""),
            "case_id": r.get("case_id", ""),
            "产线": r.get("line_id", ""),
            "调参前缺陷率": r.get("metric_before"),
            "调参后缺陷率": r.get("metric_after"),
            "改善%": r.get("improvement_pct"),
            "状态": r.get("status", ""),
            "已归因": "是" if r.get("attribution_done") == 1 else "否",
            "调度时间": str(r.get("scheduled_at", ""))[:19],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_effect_dashboard():
    """M5-3: 渲染效果看板页面"""
    st.header("效果看板")
    st.caption("M5-3: 调参效果跟踪与归因可视化")

    tracker = _get_effect_tracker()
    memory = _get_memory_service()

    # ===== 顶部控制栏 =====
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_id = st.text_input("用户ID", value="admin", key="effect_user_id",
                                help="不同用户可见产线范围不同")
    with col2:
        days = st.slider("统计天数", 7, 365, 30, key="effect_days",
                         help="统计近 N 天数据")
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

    # 产线选择（可选单产线或全部）
    line_choice = st.selectbox(
        "产线筛选",
        ["全部可见产线"] + visible_lines,
        key="effect_line_filter",
    )
    target_line = None if line_choice == "全部可见产线" else line_choice

    st.info(f"可见产线：{', '.join(visible_lines)}")

    # ===== 获取数据 =====
    try:
        # 统计（单产线或全部）
        if target_line:
            stats = tracker.get_effect_stats(line_id=target_line, days=days)
            records = tracker.list_trackings(line_id=target_line, days=days, limit=200)
        else:
            # 全部产线：合并统计
            stats = {"total": 0, "tracked": 0, "pending": 0, "skipped": 0,
                     "avg_improvement": 0.0, "positive_count": 0, "negative_count": 0}
            records = []
            improvements = []
            for lid in visible_lines:
                s = tracker.get_effect_stats(line_id=lid, days=days)
                stats["total"] += s["total"]
                stats["tracked"] += s["tracked"]
                stats["pending"] += s["pending"]
                stats["skipped"] += s["skipped"]
                stats["positive_count"] += s["positive_count"]
                stats["negative_count"] += s["negative_count"]
                improvements.extend(
                    r["improvement_pct"] for r in tracker.list_trackings(
                        line_id=lid, status="tracked", days=days, limit=500
                    ) if r.get("improvement_pct") is not None
                )
                records.extend(tracker.list_trackings(
                    line_id=lid, days=days, limit=200))
            if improvements:
                stats["avg_improvement"] = round(sum(improvements) / len(improvements), 2)

        attributed_count = sum(1 for r in records if r.get("attribution_done") == 1)

        # ===== 渲染各区域 =====
        _render_kpi(stats, attributed_count)
        st.divider()
        _render_improvement_distribution(records)
        st.divider()
        _render_before_after_comparison(records)
        st.divider()
        _render_attribution_results(records)
        st.divider()
        _render_records_table(records)

    except Exception as e:
        st.error(f"加载效果数据失败：{e}")
        st.exception(e)
