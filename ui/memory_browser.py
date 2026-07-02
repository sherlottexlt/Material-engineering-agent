"""
MetaCraft Agent 记忆浏览界面（M3-12 / M3-7）

展示 Agent 当前"记得"什么，回答记忆系统是否在正确积累/检索的问题。
直连 MemoryService，无需启动 API 服务器。

用法（被 streamlit_app.py 引入）：
    from ui.memory_browser import render_memory_browser
    render_memory_browser()
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import streamlit as st

from agent.memory.memory_service import MemoryService
from models.entities import (
    BatchParams,
    CaseRecord,
    DefectType,
    ProcessType,
)


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


def _parse_tags_input(tags_input: str) -> list[str]:
    """将逗号分隔的标签字符串解析为列表（去重 + 去空白）"""
    if not tags_input:
        return []
    parts = [t.strip() for t in tags_input.split(",") if t.strip()]
    # 去重保序
    seen = set()
    result = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _render_case_create_form(memory: MemoryService):
    """渲染新增案例表单（M3-7 Create）"""
    with st.expander("➕ 新增案例", expanded=False):
        with st.form("case_create_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                case_id = st.text_input(
                    "案例 ID",
                    value=f"case-{uuid.uuid4().hex[:8]}",
                    help="留空将自动生成；重复 ID 写入会失败",
                )
                defect_type = st.selectbox(
                    "缺陷类型",
                    options=[e.value for e in DefectType],
                    index=0,
                )
                root_cause = st.text_area("根因分析", height=80)
                confidence = st.slider("置信度", 0.0, 1.0, 0.5, 0.05)
            with c2:
                batch_id = st.text_input("批次编号", value="B-MANUAL")
                process_type = st.selectbox(
                    "工艺类型",
                    options=[e.value for e in ProcessType],
                    index=0,
                )
                temperature = st.number_input("温度 (℃)", value=850.0, step=1.0)
                holding_time = st.number_input("保温时间 (分钟)", value=120.0, step=5.0)
                cooling_rate = st.number_input("冷却速率 (℃/s)", value=1.5, step=0.1)

            solution = st.text_area("解决方案", height=80)
            tags_input = st.text_input(
                "标签（逗号分隔）",
                value="",
                placeholder="例如: 紧急, 参考案例, 实验证实",
            )

            submitted = st.form_submit_button("写入长期记忆", type="primary")
            if submitted:
                if not case_id.strip():
                    st.error("案例 ID 不能为空")
                    return
                if not root_cause.strip():
                    st.error("根因分析不能为空")
                    return
                try:
                    case = CaseRecord(
                        case_id=case_id.strip(),
                        defect_type=DefectType(defect_type),
                        batch_params=BatchParams(
                            batch_id=batch_id.strip() or "B-MANUAL",
                            process_type=ProcessType(process_type),
                            temperature=float(temperature) if temperature else None,
                            holding_time=float(holding_time) if holding_time else None,
                            cooling_rate=float(cooling_rate) if cooling_rate else None,
                            start_time=datetime.now(),
                        ),
                        root_cause=root_cause.strip(),
                        solution=solution.strip(),
                        confidence=float(confidence),
                        source="manual",
                        tags=_parse_tags_input(tags_input),
                    )
                    ok = memory.write_semantic(case)
                    if ok:
                        st.success(f"案例已写入长期记忆: {case.case_id}")
                        # 写入后若检测到冲突，提示用户
                        conflicts = memory.list_conflicts(limit=5)
                        if conflicts:
                            st.warning(
                                f"检测到 {len(conflicts)} 条近期知识冲突，"
                                f"请到「用户反馈」或 API `/memory/conflicts` 查看"
                            )
                        st.rerun()
                    else:
                        st.error("写入失败：Chroma 不可用或 ID 已存在")
                except Exception as e:
                    st.error(f"写入异常: {e}")


def _render_case_table_with_actions(memory: MemoryService, records: list[dict]):
    """渲染案例表格 + 编辑/删除操作（M3-7 Read/Update/Delete）"""
    if not records:
        st.info("暂无长期记忆案例。可在上方「新增案例」表单录入。")
        return

    # 过滤器
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        defect_types = sorted({
            (r.get("metadata") or {}).get("defect_type", "")
            for r in records
            if (r.get("metadata") or {}).get("defect_type")
        })
        selected = st.selectbox(
            "按缺陷类型过滤",
            options=["全部"] + defect_types,
            key="case_mgmt_filter_type",
        )
    with fcol2:
        kw = st.text_input("按 ID/根因/方案搜索", value="", key="case_mgmt_search")

    filtered = records
    if selected != "全部":
        filtered = [
            r for r in filtered
            if (r.get("metadata") or {}).get("defect_type") == selected
        ]
    if kw:
        kw_lower = kw.lower()
        filtered = [
            r for r in filtered
            if kw_lower in (r.get("id", "") or "").lower()
            or kw_lower in (r.get("document", "") or "").lower()
        ]

    st.write(f"**案例列表**（{len(filtered)}/{len(records)} 条）")
    st.dataframe(
        [
            {
                "案例ID": r.get("id", ""),
                "缺陷类型": (r.get("metadata") or {}).get("defect_type", ""),
                "置信度": _fmt_confidence((r.get("metadata") or {}).get("confidence")),
                "标签": (r.get("metadata") or {}).get("tags", ""),
                "根因": ((r.get("document", "") or "").split("\n", 2)[1:2] or [""])[0][:60],
                "创建时间": _fmt_datetime((r.get("metadata") or {}).get("created_at")),
            }
            for r in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.write("**编辑 / 删除案例**")
    case_ids = [r.get("id", "") for r in filtered if r.get("id")]
    if not case_ids:
        st.info("当前过滤条件下无可操作案例")
        return

    selected_id = st.selectbox("选择案例 ID", options=case_ids, key="case_mgmt_select")
    if not selected_id:
        return

    # 拉取最新数据（避免使用缓存）
    detail = memory.get_semantic_case(selected_id)
    if detail is None:
        st.warning(f"案例 {selected_id} 已不存在（可能已被删除）")
        return

    meta = detail.get("metadata") or {}
    doc = detail.get("document") or ""
    parts = doc.split("\n", 2)
    old_root = parts[1] if len(parts) > 1 else ""
    old_solution = parts[2] if len(parts) > 2 else ""
    old_tags_str = meta.get("tags", "")
    old_confidence = float(meta.get("confidence", 0.5))

    with st.form("case_edit_form"):
        new_root = st.text_area("根因分析", value=old_root, height=80, key="case_edit_root")
        new_solution = st.text_area(
            "解决方案", value=old_solution, height=80, key="case_edit_solution"
        )
        new_confidence = st.slider(
            "置信度", 0.0, 1.0, old_confidence, 0.05, key="case_edit_conf"
        )
        new_tags = st.text_input(
            "标签（逗号分隔；清空则删除全部标签）",
            value=old_tags_str,
            key="case_edit_tags",
        )
        ec1, ec2 = st.columns(2)
        with ec1:
            save_clicked = st.form_submit_button("💾 保存修改", type="primary")
        with ec2:
            delete_clicked = st.form_submit_button("🗑 删除案例")

    if save_clicked:
        tags_list = _parse_tags_input(new_tags)
        ok = memory.update_semantic(
            case_id=selected_id,
            root_cause=new_root.strip() if new_root.strip() != old_root else None,
            solution=new_solution.strip() if new_solution.strip() != old_solution else None,
            confidence=float(new_confidence) if abs(new_confidence - old_confidence) > 1e-6 else None,
            tags=tags_list if new_tags != old_tags_str else None,
        )
        if ok:
            st.success(f"案例 {selected_id} 已更新")
            st.rerun()
        else:
            st.error("更新失败（Chroma 不可用或案例不存在）")

    if delete_clicked:
        # 二次确认
        st.session_state["case_delete_confirm"] = selected_id
        st.warning(f"确认要删除案例 {selected_id} 吗？此操作不可撤销。")

    if st.session_state.get("case_delete_confirm") == selected_id:
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("✅ 确认删除", key="case_delete_yes"):
                ok = memory.delete_semantic(selected_id)
                if ok:
                    st.success(f"案例 {selected_id} 已删除")
                    st.session_state["case_delete_confirm"] = None
                    st.rerun()
                else:
                    st.error("删除失败（Chroma 不可用）")
        with cc2:
            if st.button("❌ 取消", key="case_delete_no"):
                st.session_state["case_delete_confirm"] = None
                st.rerun()


def _render_case_management(memory: MemoryService):
    """渲染案例库管理界面（M3-7 主入口：CRUD + 标签）"""
    st.write("**M3-7 案例库管理**：新增、编辑、删除长期记忆案例，支持标签管理。")

    # 检测 Chroma 是否可用
    if memory._collection is None:
        memory._ensure_chroma()
    if memory._collection is None:
        st.error(
            "Chroma 不可用，无法管理长期记忆。请检查 `data/chroma/` 目录或 Chroma 依赖。"
        )
        return

    # 新增案例表单
    _render_case_create_form(memory)

    st.divider()

    # 案例列表 + 编辑/删除
    try:
        records = memory.list_all_semantic(limit=500)
    except Exception as e:
        st.error(f"加载案例列表失败: {e}")
        return

    _render_case_table_with_actions(memory, records)


def render_memory_browser():
    """渲染记忆浏览界面（M3-12 / M3-7 主入口）

    五个 Tab：概览 / 短期记忆 / 长期记忆 / 用户反馈 / 案例管理
    直连 MemoryService，无需启动 API 服务器。
    """
    st.subheader("🧠 记忆浏览")
    st.caption("展示 Agent 当前\"记得\"什么（短期记忆 / 长期记忆 / 用户反馈 / 案例管理）")

    try:
        memory = _get_memory_service()
    except Exception as e:
        st.error(f"初始化记忆服务失败: {e}")
        return

    # 五个 Tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 概览", "📝 短期记忆", "💾 长期记忆", "💬 用户反馈", "🔧 案例管理"]
    )

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

    with tab5:
        try:
            _render_case_management(memory)
        except Exception as e:
            st.error(f"加载案例管理失败: {e}")
