"""
Orchestrator 单元测试（M0 线性流程）

测试内容：
- _route_after_review 路由逻辑（审核通过/不通过/重试超限）
- build_orchestrator 图构建
- get_orchestrator 单例行为
"""
import pytest

from agent.orchestrator import _route_after_review, build_orchestrator, get_orchestrator
from models.state import AgentState


def _make_state(**overrides) -> AgentState:
    """构造测试用 AgentState"""
    base = {
        "user_query": "test",
        "batch_id": "B001",
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
        "trace_id": "test",
        "session_id": "test",
    }
    base.update(overrides)
    return base


class TestRouteAfterReview:
    """测试 Review 后路由逻辑"""

    def test_approved_routes_to_interaction(self):
        """审核通过 → interaction"""
        state = _make_state(review_result={"approved": True})
        assert _route_after_review(state) == "interaction"

    def test_rejected_routes_to_decision(self):
        """审核不通过且未超限 → decision（重试）"""
        state = _make_state(
            review_result={"approved": False},
            retry_count=1,
            max_replan=3,
        )
        assert _route_after_review(state) == "decision"

    def test_rejected_retry_zero_routes_to_decision(self):
        """审核不通过，retry=0 → decision"""
        state = _make_state(
            review_result={"approved": False},
            retry_count=0,
            max_replan=3,
        )
        assert _route_after_review(state) == "decision"

    def test_retry_exceeded_forces_interaction(self):
        """重试超限 → 强制放行到 interaction"""
        state = _make_state(
            review_result={"approved": False},
            retry_count=3,
            max_replan=3,
        )
        assert _route_after_review(state) == "interaction"

    def test_retry_exceeds_max_forces_interaction(self):
        """重试次数超过上限 → 强制放行"""
        state = _make_state(
            review_result={"approved": False},
            retry_count=5,
            max_replan=3,
        )
        assert _route_after_review(state) == "interaction"

    def test_no_review_result_routes_to_decision(self):
        """review_result 为 None → 视为不通过，回退 decision"""
        state = _make_state(review_result=None, retry_count=0, max_replan=3)
        assert _route_after_review(state) == "decision"

    def test_max_replan_one_force_pass_on_second_retry(self):
        """max_replan=1 时，第 2 次重试应强制放行"""
        state = _make_state(
            review_result={"approved": False},
            retry_count=1,
            max_replan=1,
        )
        assert _route_after_review(state) == "interaction"


class TestBuildOrchestrator:
    """测试编排器构建"""

    def test_build_returns_compiled_graph(self):
        """build_orchestrator 应返回可执行图"""
        graph = build_orchestrator()
        assert graph is not None
        # 编译后的图应有 ainvoke 方法
        assert hasattr(graph, "ainvoke")

    def test_graph_has_all_nodes(self):
        """图应包含 M0 流程的所有节点"""
        graph = build_orchestrator()
        # LangGraph 编译后可以通过 nodes 属性查看节点
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "planner",
            "data",
            "mechanism",
            "knowledge",
            "decision",
            "review",
            "interaction",
            "memory_writer",
        }
        # 图会有额外的内置节点（如 __start__, __end__），只验证业务节点存在
        assert expected_nodes.issubset(node_names), f"缺失节点: {expected_nodes - node_names}"


class TestGetOrchestrator:
    """测试单例行为"""

    def test_returns_same_instance(self):
        """多次调用应返回同一实例"""
        orch1 = get_orchestrator()
        orch2 = get_orchestrator()
        assert orch1 is orch2

    def test_returns_callable(self):
        """返回的实例应可调用"""
        orch = get_orchestrator()
        assert hasattr(orch, "ainvoke")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
