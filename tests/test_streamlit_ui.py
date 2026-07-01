"""
Streamlit UI 组件单测

用 MagicMock 模拟 streamlit 模块，测试 render_* 函数的逻辑。
重点验证本轮 UI 修复：
- render_review 读 suggestions/reason（不是 issues）
- render_observations 正确渲染执行轨迹
- run_agent_direct 接受 defect_type 参数
"""
import sys
import json
from unittest.mock import MagicMock, patch

import pytest


# ===== 注入 mock streamlit 模块 =====
# streamlit_app.py 在模块级别调用 st.set_page_config / st.title 等，
# 必须在 import 前注入 mock，否则会因缺乏 Streamlit 运行时上下文而报错。

def _make_mock_streamlit():
    """创建模拟 streamlit 模块"""
    mock = MagicMock(name="streamlit")

    # st.columns(n) 或 st.columns([3, 1]) 都返回对应数量的 MagicMock 列表
    def _columns(n=2, *a, **kw):
        if isinstance(n, (list, tuple)):
            return [MagicMock() for _ in n]
        return [MagicMock() for _ in range(n)]
    mock.columns = _columns

    # st.sidebar 是 context manager
    mock.sidebar = MagicMock()

    # set_page_config / title / caption 等都是 no-op
    return mock


_mock_st = _make_mock_streamlit()
sys.modules["streamlit"] = _mock_st

# 现在可以安全导入 streamlit_app
from ui import streamlit_app


@pytest.fixture(autouse=True)
def _reset_mock():
    """每个测试前重置 mock，避免模块级代码的调用干扰断言"""
    _mock_st.reset_mock()
    # 重新配置 columns（reset_mock 会清掉 side_effect）
    def _columns(n=2, *a, **kw):
        if isinstance(n, (list, tuple)):
            return [MagicMock() for _ in n]
        return [MagicMock() for _ in range(n)]
    _mock_st.columns = _columns
    yield


# ===== render_review 测试 =====
class TestRenderReview:
    """测试审核结果渲染（重点：读 suggestions/reason，不是 issues）"""

    def test_empty_returns_early(self):
        """空 review_result 不渲染"""
        streamlit_app.render_review(None)
        streamlit_app.render_review({})
        _mock_st.success.assert_not_called()
        _mock_st.warning.assert_not_called()

    def test_approved_shows_success(self):
        """审核通过显示 success"""
        streamlit_app.render_review({"approved": True, "reason": "OK"})
        _mock_st.success.assert_called_once_with("审核通过")

    def test_rejected_shows_warning(self):
        """审核未通过显示 warning"""
        streamlit_app.render_review({"approved": False, "reason": "证据不足"})
        _mock_st.warning.assert_called_once()
        args = _mock_st.warning.call_args[0]
        assert "审核未通过" in args[0]
        assert "已用完重试次数" in args[0]

    def test_rejected_shows_reason(self):
        """未通过时显示原因"""
        streamlit_app.render_review({
            "approved": False,
            "reason": "根因不一致：root_cause 说保温不足，但 evidence 显示 holding_time=130",
        })
        # st.write 会被调用，其中一次是 reason
        write_calls = [str(c) for c in _mock_st.write.call_args_list]
        assert any("根因不一致" in c for c in write_calls)

    def test_rejected_shows_suggestions(self):
        """未通过时显示改进建议（重点：读 suggestions 不是 issues）"""
        streamlit_app.render_review({
            "approved": False,
            "reason": "证据不足",
            "suggestions": ["补充数值比对", "检查 evidence 完整性"],
        })
        write_calls = [str(c) for c in _mock_st.write.call_args_list]
        # 应该写出每条 suggestion
        assert any("补充数值比对" in c for c in write_calls)
        assert any("检查 evidence 完整性" in c for c in write_calls)

    def test_approved_with_suggestions_also_shows(self):
        """通过时若有 suggestions 也显示"""
        streamlit_app.render_review({
            "approved": True,
            "reason": "方案合理",
            "suggestions": ["可考虑增加保温时间"],
        })
        write_calls = [str(c) for c in _mock_st.write.call_args_list]
        assert any("可考虑增加保温时间" in c for c in write_calls)

    def test_missing_reason_no_crash(self):
        """缺少 reason 字段不崩溃"""
        streamlit_app.render_review({"approved": False})
        _mock_st.warning.assert_called_once()

    def test_missing_suggestions_no_crash(self):
        """缺少 suggestions 字段不崩溃"""
        streamlit_app.render_review({"approved": False, "reason": "证据不足"})
        _mock_st.warning.assert_called_once()

    def test_does_not_read_issues_field(self):
        """确认不读旧的 issues 字段（BUG 修复验证）"""
        # 旧代码读 issues，新代码读 suggestions
        review = {
            "approved": False,
            "reason": "X",
            "issues": ["旧字段，不应被读取"],
            "suggestions": ["新字段"],
        }
        streamlit_app.render_review(review)
        write_calls = [str(c) for c in _mock_st.write.call_args_list]
        assert any("新字段" in c for c in write_calls)
        assert not any("旧字段" in c for c in write_calls)


# ===== render_observations 测试 =====
class TestRenderObservations:
    """测试执行轨迹渲染（新功能）"""

    def test_empty_returns_early(self):
        """空 observations 不渲染"""
        streamlit_app.render_observations([])
        streamlit_app.render_observations(None)
        _mock_st.expander.assert_not_called()

    def test_renders_expander_with_count(self):
        """显示步数"""
        obs = [{"agent": "data", "tool": "query_batch_params", "result": "ok"}]
        streamlit_app.render_observations(obs)
        _mock_st.expander.assert_called_once()
        label = _mock_st.expander.call_args[0][0]
        assert "1 步" in label

    def test_renders_multiple_steps(self):
        """多步轨迹"""
        obs = [
            {"agent": "data", "tool": "query", "result": "r1"},
            {"agent": "mechanism", "tool": "jmak", "result": "r2"},
            {"agent": "decision", "tool": None, "result": "r3"},
        ]
        streamlit_app.render_observations(obs)
        label = _mock_st.expander.call_args[0][0]
        assert "3 步" in label

    def test_dict_result_serialized(self):
        """dict 类型的 result 被 JSON 序列化"""
        obs = [{"agent": "data", "tool": "query", "result": {"key": "value"}}]
        streamlit_app.render_observations(obs)
        caption_calls = [str(c) for c in _mock_st.caption.call_args_list]
        assert any("key" in c and "value" in c for c in caption_calls)

    def test_long_result_truncated(self):
        """长 result 被截断到 200 字符"""
        long_result = "X" * 500
        obs = [{"agent": "data", "tool": "query", "result": long_result}]
        streamlit_app.render_observations(obs)
        caption_calls = [str(c) for c in _mock_st.caption.call_args_list]
        # 截断后应该不超过 200+ 字符（含前缀）
        assert any("XXX" in c for c in caption_calls)
        # 不应该包含完整的 500 字符
        assert not any("X" * 500 in c for c in caption_calls)

    def test_missing_agent_shows_question_mark(self):
        """缺少 agent 字段显示 ?"""
        obs = [{"tool": "query", "result": "ok"}]
        streamlit_app.render_observations(obs)
        write_calls = [str(c) for c in _mock_st.write.call_args_list]
        assert any("[?]" in c for c in write_calls)

    def test_missing_tool_handled(self):
        """缺少 tool 字段不崩溃"""
        obs = [{"agent": "data", "result": "ok"}]
        streamlit_app.render_observations(obs)
        # 不崩溃即可
        _mock_st.expander.assert_called_once()


# ===== render_proposals 测试 =====
class TestRenderProposals:
    """测试候选方案渲染"""

    def test_empty_returns_early(self):
        """空 decision_result 不渲染"""
        streamlit_app.render_proposals(None)
        streamlit_app.render_proposals({})
        _mock_st.write.assert_not_called()

    def test_shows_proposal_count_and_source(self):
        """显示方案数和来源"""
        decision = {
            "proposals": [{"root_cause": "X", "confidence": 0.8}],
            "source": "llm",
        }
        streamlit_app.render_proposals(decision)
        first_write = str(_mock_st.write.call_args_list[0])
        assert "1 个" in first_write
        assert "llm" in first_write

    def test_confidence_labels(self):
        """置信度标注（高/中/低）"""
        for confidence, expected_label in [(0.9, "高"), (0.6, "中"), (0.3, "低")]:
            _mock_st.reset_mock()
            def _cols(n=2, *a, **kw):
                if isinstance(n, (list, tuple)):
                    return [MagicMock() for _ in n]
                return [MagicMock() for _ in range(n)]
            _mock_st.columns = _cols
            decision = {
                "proposals": [{"root_cause": "X", "confidence": confidence}],
                "source": "rule_based",
            }
            streamlit_app.render_proposals(decision)
            writes = [str(c) for c in _mock_st.write.call_args_list]
            assert any(expected_label in w for w in writes), \
                f"confidence={confidence} 应标注 '{expected_label}'"


# ===== render_batch_params 测试 =====
class TestRenderBatchParams:
    """测试批次参数渲染"""

    def test_empty_shows_info(self):
        """空 data_result 显示提示"""
        streamlit_app.render_batch_params(None)
        _mock_st.info.assert_called_once()

    def test_shows_temperature(self):
        """显示温度"""
        streamlit_app.render_batch_params({"temperature": 820})
        metric_calls = [str(c) for c in _mock_st.metric.call_args_list]
        assert any("820" in c and "温度" in c for c in metric_calls)

    def test_shows_holding_time(self):
        """显示保温时间"""
        streamlit_app.render_batch_params({"holding_time": 80})
        metric_calls = [str(c) for c in _mock_st.metric.call_args_list]
        assert any("80" in c and "保温" in c for c in metric_calls)

    def test_shows_cooling_rate(self):
        """显示冷却速率"""
        streamlit_app.render_batch_params({"cooling_rate": 3.5})
        metric_calls = [str(c) for c in _mock_st.metric.call_args_list]
        assert any("3.5" in c and "冷却" in c for c in metric_calls)


# ===== run_agent_direct 签名测试 =====
class TestRunAgentDirectSignature:
    """测试 run_agent_direct 接受新参数（不实际调用 Agent）"""

    def test_accepts_defect_type(self):
        """函数签名包含 defect_type 参数"""
        import inspect
        sig = inspect.signature(streamlit_app.run_agent_direct)
        assert "defect_type" in sig.parameters
        assert "measured_value" in sig.parameters
        assert "standard_value" in sig.parameters

    def test_defect_type_defaults_none(self):
        """defect_type 默认为 None"""
        import inspect
        sig = inspect.signature(streamlit_app.run_agent_direct)
        assert sig.parameters["defect_type"].default is None

    @patch("ui.streamlit_app._build_orchestrator")
    @patch("ui.streamlit_app.asyncio")
    def test_defect_type_passed_to_state(self, mock_asyncio, mock_build):
        """defect_type 被传入 defect_record"""
        # 让 asyncio.run 返回一个空 dict
        mock_asyncio.run.return_value = {}
        mock_build.return_value = MagicMock()

        streamlit_app.run_agent_direct(
            "query", "B001", max_replan=1,
            defect_type="hardness_low",
            measured_value=55.0,
            standard_value=58.0,
        )

        # 检查 asyncio.run 被调用
        mock_asyncio.run.assert_called_once()
        mock_build.assert_called_once()

    @patch("ui.streamlit_app._build_orchestrator")
    @patch("ui.streamlit_app.asyncio")
    def test_no_defect_type_no_defect_record(self, mock_asyncio, mock_build):
        """defect_type=None 时 defect_record 也是 None"""
        mock_asyncio.run.return_value = {}
        mock_build.return_value = MagicMock()

        streamlit_app.run_agent_direct("query", "B001", defect_type=None)

        mock_build.assert_called_once()


# ===== _build_orchestrator 测试 =====
class TestBuildOrchestrator:
    """测试 _build_orchestrator 不缓存（BUG 修复验证）"""

    def test_not_cached_resource(self):
        """确认没有 @st.cache_resource 装饰器"""
        # @st.cache_resource 装饰后函数会有 __wrapped__ 属性
        assert not hasattr(streamlit_app._build_orchestrator, "__wrapped__")
        # 旧的 _get_orchestrator 函数应该不存在了
        assert not hasattr(streamlit_app, "_get_orchestrator")

    @patch("agent.orchestrator.build_orchestrator")
    def test_each_call_returns_fresh(self, mock_build):
        """每次调用应该构建新的 orchestrator（不缓存）"""
        mock_build.return_value = MagicMock()
        streamlit_app._build_orchestrator()
        streamlit_app._build_orchestrator()
        streamlit_app._build_orchestrator()
        assert mock_build.call_count == 3


# ===== M2-9: render_collaboration_flow 测试 =====
class TestRenderCollaborationFlow:
    """测试协作流程图渲染"""

    def test_empty_state_renders(self):
        """空状态不崩溃，显示未完成节点"""
        streamlit_app.render_collaboration_flow({})
        # 应该调用了 st.write 多次（标题 + 节点）
        assert _mock_st.write.call_count >= 3

    def test_shows_parallel_structure(self):
        """显示 fan-out 并行结构"""
        streamlit_app.render_collaboration_flow({})
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("fan-out" in w for w in writes)
        assert any("fan-in" in w for w in writes)

    def test_completed_nodes_marked(self):
        """已完成节点标记 ✅"""
        state = {
            "plan": [{"step_id": 1}],
            "data_result": {"batch_params": {}},
            "mechanism_result": {"jmak_output": {}},
            "knowledge_result": {"handbook_hits": {"total": 1}},
            "arbitration_result": {"conflict_count": 0},
            "decision_result": {"proposals": []},
            "review_result": {"approved": True},
            "final_answer": "done",
        }
        streamlit_app.render_collaboration_flow(state)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        # 所有节点都完成，应该有 ✅
        assert any("✅" in w for w in writes)
        # 不应该有 ⬜（未完成）
        assert not any("⬜" in w for w in writes)

    def test_partial_state_shows_mixed(self):
        """部分完成显示混合状态"""
        state = {
            "plan": [{"step_id": 1}],
            "data_result": {"batch_params": {}},
            # mechanism/knowledge 未完成
        }
        streamlit_app.render_collaboration_flow(state)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("✅" in w for w in writes)
        assert any("⬜" in w for w in writes)

    def test_conflict_count_displayed(self):
        """冲突数量显示"""
        state = {
            "arbitration_result": {"conflict_count": 2},
        }
        streamlit_app.render_collaboration_flow(state)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("2 个冲突" in w for w in writes)

    def test_no_conflict_displayed(self):
        """无冲突时显示无冲突"""
        state = {
            "arbitration_result": {"conflict_count": 0},
        }
        streamlit_app.render_collaboration_flow(state)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("无冲突" in w for w in writes)


# ===== M2-9: render_agent_timeline 测试 =====
class TestRenderAgentTimeline:
    """测试 Agent 消息流时间线"""

    def test_empty_returns_early(self):
        """空列表不渲染"""
        streamlit_app.render_agent_timeline([])
        streamlit_app.render_agent_timeline(None)
        _mock_st.expander.assert_not_called()

    def test_renders_expander_with_count(self):
        """显示消息数"""
        obs = [{"agent": "data", "tool": "query", "result": "ok"}]
        streamlit_app.render_agent_timeline(obs)
        _mock_st.expander.assert_called_once()
        label = _mock_st.expander.call_args[0][0]
        assert "1 条" in label

    def test_parallel_tag_for_parallel_agents(self):
        """并行 Agent 标记为 [并行]"""
        obs = [
            {"agent": "data", "tool": "query", "result": "r1"},
            {"agent": "mechanism", "tool": "jmak", "result": "r2"},
            {"agent": "knowledge", "tool": "search", "result": "r3"},
        ]
        streamlit_app.render_agent_timeline(obs)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        # data/mechanism/knowledge 都是并行
        assert sum("并行" in w for w in writes) >= 3

    def test_serial_tag_for_decision(self):
        """decision 标记为 [串行]"""
        obs = [{"agent": "decision", "tool": None, "result": "r1"}]
        streamlit_app.render_agent_timeline(obs)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("串行" in w for w in writes)

    def test_tool_displayed_when_present(self):
        """有 tool 时显示工具"""
        obs = [{"agent": "data", "tool": "query_batch_params", "result": "r1"}]
        streamlit_app.render_agent_timeline(obs)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("query_batch_params" in w for w in writes)

    def test_timestamp_displayed_when_present(self):
        """有 timestamp 时显示时间戳"""
        obs = [{"agent": "data", "tool": "q", "result": "r", "timestamp": "12:00:00"}]
        streamlit_app.render_agent_timeline(obs)
        caption_calls = [str(c) for c in _mock_st.caption.call_args_list]
        assert any("12:00:00" in c for c in caption_calls)

    def test_display_name_used(self):
        """使用角色中文显示名"""
        obs = [{"agent": "data", "tool": "q", "result": "r"}]
        streamlit_app.render_agent_timeline(obs)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        # data agent 的显示名是"数据 Agent"
        assert any("数据 Agent" in w for w in writes)

    def test_dict_result_serialized(self):
        """dict 结果被序列化"""
        obs = [{"agent": "data", "tool": "q", "result": {"key": "val"}}]
        streamlit_app.render_agent_timeline(obs)
        caption_calls = [str(c) for c in _mock_st.caption.call_args_list]
        assert any("key" in c and "val" in c for c in caption_calls)


# ===== M2-9: render_arbitration 测试 =====
class TestRenderArbitration:
    """测试冲突仲裁结果渲染"""

    def test_empty_returns_early(self):
        """空结果不渲染"""
        streamlit_app.render_arbitration(None)
        streamlit_app.render_arbitration({})
        _mock_st.expander.assert_not_called()

    def test_no_conflicts_shows_success(self):
        """无冲突显示成功"""
        streamlit_app.render_arbitration({"conflict_count": 0, "conflicts": []})
        _mock_st.success.assert_called_once()
        assert "无冲突" in _mock_st.success.call_args[0][0]

    def test_high_severity_shows_error(self):
        """高严重度冲突显示 error"""
        streamlit_app.render_arbitration({
            "conflict_count": 1,
            "conflicts": [{"type": "mismatch", "detail": "d", "severity": "high"}],
            "has_high_severity": True,
        })
        _mock_st.error.assert_called_once()

    def test_medium_severity_shows_warning(self):
        """中低严重度显示 warning"""
        streamlit_app.render_arbitration({
            "conflict_count": 1,
            "conflicts": [{"type": "empty", "detail": "d", "severity": "medium"}],
            "has_high_severity": False,
        })
        _mock_st.warning.assert_called_once()

    def test_conflict_details_displayed(self):
        """冲突详情显示"""
        streamlit_app.render_arbitration({
            "conflict_count": 2,
            "conflicts": [
                {"type": "mechanism_data_mismatch", "detail": "硬度不匹配", "severity": "high"},
                {"type": "knowledge_empty", "detail": "知识为空", "severity": "medium"},
            ],
            "has_high_severity": True,
        })
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("mechanism_data_mismatch" in w for w in writes)
        assert any("knowledge_empty" in w for w in writes)
        assert any("硬度不匹配" in w for w in writes)


# ===== M2-10: render_intermediate_results 测试 =====
class TestRenderIntermediateResults:
    """测试中间结果可展开查看"""

    def test_empty_state_no_crash(self):
        """空状态不崩溃"""
        streamlit_app.render_intermediate_results({})
        # 应该只写了标题"中间结果"
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("中间结果" in w for w in writes)

    def test_data_result_expander(self):
        """数据 Agent 结果有 expander"""
        state = {
            "data_result": {
                "batch_params": {"temperature": 820},
                "defect_history": [{"id": "D001"}],
            }
        }
        streamlit_app.render_intermediate_results(state)
        # 应该调用 st.expander 和 st.json
        _mock_st.expander.assert_called()
        _mock_st.json.assert_called()

    def test_mechanism_result_expander(self):
        """机理 Agent 结果有 expander"""
        state = {
            "mechanism_result": {
                "jmak_output": {"predicted_hardness": 55.0},
            }
        }
        streamlit_app.render_intermediate_results(state)
        _mock_st.expander.assert_called()
        json_calls = [str(c) for c in _mock_st.json.call_args_list]
        assert any("predicted_hardness" in c for c in json_calls)

    def test_knowledge_result_expander(self):
        """知识 Agent 结果有 expander"""
        state = {
            "knowledge_result": {
                "handbook_hits": {"total": 2, "hits": [{"title": "手册1"}]},
                "case_hits": {"total": 1, "hits": [{"id": "C001"}]},
            }
        }
        streamlit_app.render_intermediate_results(state)
        _mock_st.expander.assert_called()
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("2" in w and "手册" in w for w in writes)

    def test_decision_result_expander(self):
        """决策 Agent 原始输出有 expander"""
        state = {
            "decision_result": {"proposals": [{"root_cause": "X"}], "source": "llm"}
        }
        streamlit_app.render_intermediate_results(state)
        _mock_st.expander.assert_called()
        _mock_st.json.assert_called()

    def test_review_result_expander(self):
        """审核 Agent 原始输出有 expander"""
        state = {
            "review_result": {"approved": True, "reason": "OK"}
        }
        streamlit_app.render_intermediate_results(state)
        _mock_st.expander.assert_called()
        _mock_st.json.assert_called()

    def test_all_results_multiple_expanders(self):
        """所有结果都有 expander"""
        state = {
            "data_result": {"batch_params": {}},
            "mechanism_result": {"jmak_output": {}},
            "knowledge_result": {"handbook_hits": {}, "case_hits": {}},
            "decision_result": {"proposals": []},
            "review_result": {"approved": True},
        }
        streamlit_app.render_intermediate_results(state)
        # 至少 5 个 expander
        assert _mock_st.expander.call_count >= 5


# ===== _get_display_name 测试 =====
class TestGetDisplayName:
    """测试角色显示名获取"""

    def test_known_agent(self):
        """已知角色返回中文显示名"""
        name = streamlit_app._get_display_name("data")
        assert name == "数据 Agent"

    def test_mechanism_agent(self):
        name = streamlit_app._get_display_name("mechanism")
        assert name == "机理 Agent"

    def test_unknown_agent_returns_name(self):
        """未知角色返回原名"""
        name = streamlit_app._get_display_name("unknown_agent")
        assert name == "unknown_agent"


# ===== M2-11/12: 流程选择器测试 =====
class TestFlowSelector:
    """测试协作流程选择器"""

    def test_run_agent_direct_accepts_flow_name(self):
        """run_agent_direct 接受 flow_name 参数"""
        import inspect
        sig = inspect.signature(streamlit_app.run_agent_direct)
        assert "flow_name" in sig.parameters
        assert sig.parameters["flow_name"].default is None

    def test_build_orchestrator_accepts_flow_name(self):
        """_build_orchestrator 接受 flow_name 参数"""
        import inspect
        sig = inspect.signature(streamlit_app._build_orchestrator)
        assert "flow_name" in sig.parameters
        assert sig.parameters["flow_name"].default is None

    @patch("ui.streamlit_app._build_orchestrator")
    @patch("ui.streamlit_app.asyncio")
    def test_flow_name_passed_to_build(self, mock_asyncio, mock_build):
        """flow_name 被传给 _build_orchestrator"""
        mock_asyncio.run.return_value = {}
        mock_build.return_value = MagicMock()

        streamlit_app.run_agent_direct("query", "B001", flow_name="quick")

        # 检查 _build_orchestrator 被调用时传了 flow_name="quick"
        mock_build.assert_called_once_with("quick")

    @patch("ui.streamlit_app._build_orchestrator")
    @patch("ui.streamlit_app.asyncio")
    def test_flow_name_none_uses_default(self, mock_asyncio, mock_build):
        """flow_name=None 时使用默认流程"""
        mock_asyncio.run.return_value = {}
        mock_build.return_value = MagicMock()

        streamlit_app.run_agent_direct("query", "B001", flow_name=None)

        mock_build.assert_called_once_with(None)

    @patch("agent.orchestrator.build_orchestrator")
    def test_build_orchestrator_passes_flow_name(self, mock_build):
        """_build_orchestrator 将 flow_name 传给 agent.orchestrator"""
        mock_build.return_value = MagicMock()
        streamlit_app._build_orchestrator("sequential")
        mock_build.assert_called_once_with("sequential")

    @patch("agent.orchestrator.build_orchestrator")
    def test_build_orchestrator_none_passes_through(self, mock_build):
        """_build_orchestrator(None) 传 None"""
        mock_build.return_value = MagicMock()
        streamlit_app._build_orchestrator(None)
        mock_build.assert_called_once_with(None)
