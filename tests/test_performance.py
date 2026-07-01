"""
M2-13 性能优化测试

测试：
1. decision_agent 的上下文摘要函数 _summarize_for_decision
2. Streamlit UI 的 render_performance_report
"""
import json
import sys
from unittest.mock import MagicMock

import pytest

# ===== 注入 mock streamlit（与 test_streamlit_ui.py 相同方式）=====
def _make_mock_streamlit():
    mock = MagicMock(name="streamlit")
    def _columns(n=2, *a, **kw):
        if isinstance(n, (list, tuple)):
            return [MagicMock() for _ in n]
        return [MagicMock() for _ in range(n)]
    mock.columns = _columns
    mock.sidebar = MagicMock()
    return mock

_mock_st = _make_mock_streamlit()
sys.modules["streamlit"] = _mock_st

from ui import streamlit_app
from agent.nodes.decision_agent import _summarize_for_decision


@pytest.fixture(autouse=True)
def _reset_mock():
    _mock_st.reset_mock()
    def _columns(n=2, *a, **kw):
        if isinstance(n, (list, tuple)):
            return [MagicMock() for _ in n]
        return [MagicMock() for _ in range(n)]
    _mock_st.columns = _columns
    yield


# ===== _summarize_for_decision 测试 =====

class TestSummarizeForDecision:
    """测试上下文摘要函数（M2-13 token 节省）"""

    def test_returns_three_strings(self):
        """返回三个字符串"""
        data, mechanism, knowledge = _summarize_for_decision({}, {}, {})
        assert isinstance(data, str)
        assert isinstance(mechanism, str)
        assert isinstance(knowledge, str)

    def test_empty_inputs_no_crash(self):
        """空输入不崩溃"""
        data, mechanism, knowledge = _summarize_for_decision({}, {}, {})
        assert "batch_params" in data
        assert "jmak_prediction" in mechanism
        assert "handbook_total" in knowledge

    def test_data_summary_only_key_params(self):
        """数据摘要只包含关键参数"""
        data_result = {
            "batch_params": {
                "temperature": 820,
                "holding_time": 80,
                "cooling_rate": 3.5,
                "extra_field": "should_not_appear",
            },
            "defect_history": [],
        }
        data, _, _ = _summarize_for_decision(data_result, {}, {})
        parsed = json.loads(data)
        assert parsed["batch_params"]["temperature"] == 820
        assert parsed["batch_params"]["holding_time"] == 80
        assert parsed["batch_params"]["cooling_rate"] == 3.5
        # extra_field 不应该在摘要中
        assert "extra_field" not in parsed["batch_params"]

    def test_data_summary_defect_count(self):
        """数据摘要包含缺陷数量"""
        data_result = {
            "batch_params": {},
            "defect_history": [
                {"defect_type": "hardness_low", "batch_id": "B001"},
                {"defect_type": "crack", "batch_id": "B002"},
                {"defect_type": "deformation", "batch_id": "B003"},
                {"defect_type": "hardness_high", "batch_id": "B004"},
            ],
        }
        data, _, _ = _summarize_for_decision(data_result, {}, {})
        parsed = json.loads(data)
        assert parsed["defect_count"] == 4
        # 只取最近 3 条
        assert len(parsed["recent_defects"]) == 3

    def test_mechanism_summary_only_outputs(self):
        """机理摘要只包含模型输出"""
        mechanism_result = {
            "jmak_output": {"outputs": {"predicted_hardness_HRc": 55.0}},
            "cooling_output": {"outputs": {"cooling_rate": 3.5}},
            "extra_data": "should_not_appear",
        }
        _, mechanism, _ = _summarize_for_decision({}, mechanism_result, {})
        parsed = json.loads(mechanism)
        assert parsed["jmak_prediction"]["predicted_hardness_HRc"] == 55.0
        assert parsed["cooling_analysis"]["cooling_rate"] == 3.5
        assert "extra_data" not in parsed

    def test_knowledge_summary_limits_hits(self):
        """知识摘要只保留前 2 条"""
        knowledge_result = {
            "handbook_hits": {
                "total": 5,
                "hits": [
                    {"title": f"手册{i}", "content": f"内容{i}" * 50}
                    for i in range(5)
                ],
            },
            "case_hits": {
                "total": 3,
                "hits": [
                    {"id": f"C00{i}", "root_cause": f"原因{i}"}
                    for i in range(3)
                ],
            },
        }
        _, _, knowledge = _summarize_for_decision({}, {}, knowledge_result)
        parsed = json.loads(knowledge)
        assert parsed["handbook_total"] == 5
        assert len(parsed["handbook_top"]) == 2  # 只保留前 2 条
        assert parsed["case_total"] == 3
        assert len(parsed["case_top"]) == 2

    def test_knowledge_summary_snippet_truncated(self):
        """知识摘要的 snippet 被截断到 100 字符"""
        long_content = "X" * 500
        knowledge_result = {
            "handbook_hits": {
                "total": 1,
                "hits": [{"title": "手册", "content": long_content}],
            },
            "case_hits": {"total": 0, "hits": []},
        }
        _, _, knowledge = _summarize_for_decision({}, {}, knowledge_result)
        parsed = json.loads(knowledge)
        snippet = parsed["handbook_top"][0]["snippet"]
        assert len(snippet) <= 100

    def test_summary_smaller_than_full_json(self):
        """摘要比完整 JSON 小（token 节省验证）"""
        data_result = {
            "batch_params": {"temperature": 820, "holding_time": 80, "cooling_rate": 3.5},
            "defect_history": [{"defect_type": "hardness_low", "batch_id": "B001"}] * 10,
        }
        full_json = json.dumps(data_result, ensure_ascii=False, default=str)
        data_summary, _, _ = _summarize_for_decision(data_result, {}, {})
        # 摘要应该比完整 JSON 小（因为只取 3 条缺陷）
        assert len(data_summary) < len(full_json)


# ===== render_performance_report 测试 =====

class TestRenderPerformanceReport:
    """测试性能报告渲染（M2-13）"""

    def test_empty_state_returns_early(self):
        """空状态和 0 耗时不渲染"""
        streamlit_app.render_performance_report({}, 0)
        _mock_st.expander.assert_not_called()

    def test_renders_with_observations(self):
        """有 observations 时渲染"""
        state = {
            "observations": [
                {"agent": "data", "tool": "query", "result": "r1"},
                {"agent": "mechanism", "tool": "jmak", "result": "r2"},
                {"agent": "decision", "tool": None, "result": "r3"},
            ]
        }
        streamlit_app.render_performance_report(state, 60.0)
        _mock_st.expander.assert_called_once()

    def test_shows_total_elapsed(self):
        """显示总耗时"""
        state = {"observations": [{"agent": "data", "tool": "q", "result": "r"}]}
        streamlit_app.render_performance_report(state, 120.5)
        metric_calls = [str(c) for c in _mock_st.metric.call_args_list]
        assert any("120" in c for c in metric_calls)

    def test_shows_step_count(self):
        """显示执行步数"""
        state = {
            "observations": [
                {"agent": "data", "tool": "q", "result": "r1"},
                {"agent": "mechanism", "tool": "j", "result": "r2"},
                {"agent": "knowledge", "tool": "s", "result": "r3"},
            ]
        }
        streamlit_app.render_performance_report(state, 60.0)
        metric_calls = [str(c) for c in _mock_st.metric.call_args_list]
        assert any("3" in c for c in metric_calls)

    def test_shows_parallel_agent_count(self):
        """显示并行 Agent 数量"""
        state = {
            "observations": [
                {"agent": "data", "tool": "q", "result": "r1"},
                {"agent": "mechanism", "tool": "j", "result": "r2"},
                {"agent": "knowledge", "tool": "s", "result": "r3"},
            ]
        }
        streamlit_app.render_performance_report(state, 60.0)
        metric_calls = [str(c) for c in _mock_st.metric.call_args_list]
        # 并行 Agent 数 = 3（data + mechanism + knowledge）
        assert any("3" in c for c in metric_calls)

    def test_shows_speedup_analysis(self):
        """显示加速比分析"""
        state = {
            "observations": [
                {"agent": "data", "tool": "q", "result": "r1"},
                {"agent": "mechanism", "tool": "j", "result": "r2"},
                {"agent": "knowledge", "tool": "s", "result": "r3"},
                {"agent": "decision", "tool": None, "result": "r4"},
            ]
        }
        streamlit_app.render_performance_report(state, 60.0)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        assert any("并行阶段" in w for w in writes)
        assert any("串行阶段" in w for w in writes)
        assert any("加速比" in w for w in writes)

    def test_shows_agent_step_distribution(self):
        """显示各 Agent 步数分布"""
        state = {
            "observations": [
                {"agent": "data", "tool": "q", "result": "r1"},
                {"agent": "data", "tool": "q2", "result": "r2"},
                {"agent": "mechanism", "tool": "j", "result": "r3"},
            ]
        }
        streamlit_app.render_performance_report(state, 60.0)
        writes = [str(c) for c in _mock_st.write.call_args_list]
        # data agent 有 2 步
        assert any("2 步" in w for w in writes)

    def test_no_observations_but_has_elapsed(self):
        """无 observations 但有耗时仍渲染"""
        streamlit_app.render_performance_report({}, 60.0)
        _mock_st.expander.assert_called_once()
