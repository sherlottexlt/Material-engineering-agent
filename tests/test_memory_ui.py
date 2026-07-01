"""
记忆浏览界面单测（M3-12）

用 MagicMock 模拟 streamlit 模块，测试 render_memory_browser 的逻辑。
重点验证：
- 空数据时不崩溃
- 有数据时调用正确的 st 方法（metric/dataframe/bar_chart）
- 辅助函数格式化正确
"""
import sys
from unittest.mock import MagicMock, patch

import pytest


# ===== 注入 mock streamlit 模块 =====
# memory_browser.py 在模块级别调用 @st.cache_resource，
# 必须在 import 前注入 mock。

def _make_mock_streamlit():
    """创建模拟 streamlit 模块"""
    mock = MagicMock(name="streamlit")

    # st.columns(n) 返回对应数量的 MagicMock 列表
    def _columns(n=2, *a, **kw):
        if isinstance(n, (list, tuple)):
            return [MagicMock() for _ in n]
        return [MagicMock() for _ in range(n)]
    mock.columns = _columns

    # st.tabs(names) 返回对应数量的 context manager
    def _tabs(names, *a, **kw):
        return [MagicMock() for _ in names]
    mock.tabs = _tabs

    # st.cache_resource 装饰器：返回原函数
    def _cache_resource(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    mock.cache_resource = _cache_resource

    return mock


_mock_st = _make_mock_streamlit()
sys.modules["streamlit"] = _mock_st

# 现在可以安全导入 memory_browser
from ui import memory_browser


# ===== 辅助函数测试 =====
class TestFmtDatetime:
    """测试 _fmt_datetime 时间格式化"""

    def test_none_returns_dash(self):
        assert memory_browser._fmt_datetime(None) == "—"

    def test_empty_string_returns_dash(self):
        assert memory_browser._fmt_datetime("") == "—"

    def test_datetime_truncated_to_minute(self):
        result = memory_browser._fmt_datetime("2026-07-02 15:30:45")
        assert result == "2026-07-02 15:30"

    def test_short_string_returned_as_is(self):
        result = memory_browser._fmt_datetime("2026-07-02")
        assert result == "2026-07-02"


class TestFmtConfidence:
    """测试 _fmt_confidence 置信度格式化"""

    def test_none_returns_dash(self):
        assert memory_browser._fmt_confidence(None) == "—"

    def test_float_to_percentage(self):
        assert memory_browser._fmt_confidence(0.85) == "85%"

    def test_zero(self):
        assert memory_browser._fmt_confidence(0.0) == "0%"

    def test_one(self):
        assert memory_browser._fmt_confidence(1.0) == "100%"

    def test_string_value(self):
        # 字符串无法转 float 时原样返回
        assert memory_browser._fmt_confidence("abc") == "abc"


# ===== 概览页测试 =====
class TestRenderOverview:
    """测试 _render_overview 概览页渲染"""

    def test_empty_stats_no_crash(self, reset_mock):
        """空统计不崩溃"""
        memory_browser._render_overview({})
        # 应该调用了 metric 4 次（三层记忆 + 平均置信度）
        assert _mock_st.metric.call_count == 4

    def test_renders_metrics_with_counts(self, reset_mock):
        """有数据时显示计数"""
        stats = {
            "episodic_count": 10,
            "feedback_count": 5,
            "semantic_count": 20,
            "avg_confidence": 0.75,
        }
        memory_browser._render_overview(stats)
        # 检查 metric 调用参数
        calls = [str(c) for c in _mock_st.metric.call_args_list]
        assert any("10" in c and "短期记忆" in c for c in calls)
        assert any("20" in c and "长期记忆" in c for c in calls)
        assert any("5" in c and "用户反馈" in c for c in calls)

    def test_renders_bar_chart_when_dist_exists(self, reset_mock):
        """有分布数据时渲染 bar_chart"""
        stats = {
            "episodic_count": 0,
            "feedback_count": 0,
            "semantic_count": 0,
            "avg_confidence": 0.0,
            "defect_type_distribution": {"hardness_low": 5, "crack": 3},
            "action_distribution": {"adopted": 4, "rejected": 1},
        }
        memory_browser._render_overview(stats)
        # 应该调用了 bar_chart 2 次（缺陷类型 + 动作分布）
        assert _mock_st.bar_chart.call_count == 2

    def test_no_bar_chart_when_dist_empty(self, reset_mock):
        """无分布数据时不渲染 bar_chart"""
        stats = {
            "episodic_count": 0,
            "feedback_count": 0,
            "semantic_count": 0,
            "avg_confidence": 0.0,
            "defect_type_distribution": {},
            "action_distribution": {},
        }
        memory_browser._render_overview(stats)
        _mock_st.bar_chart.assert_not_called()


# ===== 短期记忆表格测试 =====
class TestRenderEpisodicTable:
    """测试 _render_episodic_table 短期记忆表格"""

    def test_empty_records_shows_info(self, reset_mock):
        """空数据显示提示"""
        memory_browser._render_episodic_table([])
        _mock_st.info.assert_called_once()
        _mock_st.dataframe.assert_not_called()

    def test_renders_dataframe_with_data(self, reset_mock):
        """有数据时渲染表格"""
        records = [
            {
                "record_id": "ep_001",
                "batch_id": "B001",
                "defect_type": "hardness_low",
                "root_cause": "温度偏低",
                "solution": "提升温度",
                "quality_score": 0.8,
                "created_at": "2026-07-02 15:30:00",
            }
        ]
        memory_browser._render_episodic_table(records)
        _mock_st.dataframe.assert_called_once()


# ===== 长期记忆表格测试 =====
class TestRenderSemanticTable:
    """测试 _render_semantic_table 长期记忆表格"""

    def test_empty_records_shows_info(self, reset_mock):
        """空数据显示提示"""
        memory_browser._render_semantic_table([])
        _mock_st.info.assert_called_once()
        _mock_st.dataframe.assert_not_called()

    def test_renders_dataframe_with_data(self, reset_mock):
        """有数据时渲染表格"""
        # st.text_input 默认返回 MagicMock（truthy），会导致进入搜索分支
        # 设置为空字符串，跳过搜索过滤
        _mock_st.text_input.return_value = ""
        records = [
            {
                "id": "C001",
                "document": "硬度偏低\n温度偏低\n提升温度",
                "metadata": {
                    "defect_type": "hardness_low",
                    "confidence": 0.9,
                    "source": "auto",
                    "created_at": "2026-07-02T15:30:00",
                },
            }
        ]
        memory_browser._render_semantic_table(records)
        _mock_st.dataframe.assert_called_once()


# ===== 反馈表格测试 =====
class TestRenderFeedbackTable:
    """测试 _render_feedback_table 反馈表格"""

    def test_empty_records_shows_info(self, reset_mock):
        """空数据显示提示"""
        memory_browser._render_feedback_table([])
        _mock_st.info.assert_called_once()
        _mock_st.dataframe.assert_not_called()

    def test_renders_dataframe_with_data(self, reset_mock):
        """有数据时渲染表格"""
        records = [
            {
                "feedback_id": "fb_001",
                "proposal_id": "P001",
                "user_id": "operator_01",
                "action": "adopted",
                "score": 0.9,
                "comment": "方案有效",
                "created_at": "2026-07-02 15:30:00",
            }
        ]
        memory_browser._render_feedback_table(records)
        _mock_st.dataframe.assert_called_once()


# ===== 主入口测试 =====
class TestRenderMemoryBrowser:
    """测试 render_memory_browser 主入口"""

    def test_init_failure_shows_error(self, reset_mock):
        """MemoryService 初始化失败时显示错误"""
        with patch("ui.memory_browser._get_memory_service", side_effect=Exception("DB locked")):
            memory_browser.render_memory_browser()
        _mock_st.error.assert_called_once()
        err_msg = str(_mock_st.error.call_args[0][0])
        assert "DB locked" in err_msg

    def test_renders_four_tabs(self, reset_mock):
        """正常时渲染 4 个 tab"""
        mock_memory = MagicMock()
        mock_memory.get_memory_stats.return_value = {}
        mock_memory.list_all_episodic.return_value = []
        mock_memory.list_all_semantic.return_value = []
        mock_memory.list_all_feedback.return_value = []

        with patch("ui.memory_browser._get_memory_service", return_value=mock_memory):
            memory_browser.render_memory_browser()

        # 应该调用了 tabs 一次，传入 4 个标签名
        # 注意：_mock_st.tabs 是带 side_effect 的 MagicMock（在 fixture 里设置）
        assert _mock_st.tabs.call_count == 1
        tab_names = _mock_st.tabs.call_args[0][0]
        assert len(tab_names) == 4
        # 标签名应包含"概览"/"短期记忆"/"长期记忆"/"用户反馈"
        tab_names_str = str(tab_names)
        assert "概览" in tab_names_str
        assert "短期记忆" in tab_names_str
        assert "长期记忆" in tab_names_str
        assert "用户反馈" in tab_names_str


# ===== fixture =====
@pytest.fixture(autouse=True)
def reset_mock():
    """每个测试前重置 mock，避免模块级代码的调用干扰断言"""
    _mock_st.reset_mock()
    # 重新配置 columns / tabs / cache_resource（reset_mock 会清掉 side_effect）
    def _columns(n=2, *a, **kw):
        if isinstance(n, (list, tuple)):
            return [MagicMock() for _ in n]
        return [MagicMock() for _ in range(n)]
    _mock_st.columns = _columns

    # tabs 用 MagicMock（带 side_effect），这样才有 call_count / call_args
    def _tabs_side_effect(names, *a, **kw):
        return [MagicMock() for _ in names]
    _mock_st.tabs = MagicMock(side_effect=_tabs_side_effect)

    # text_input 默认返回空字符串（避免 truthy MagicMock 进入搜索分支）
    _mock_st.text_input.return_value = ""

    def _cache_resource(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    _mock_st.cache_resource = _cache_resource

    yield
