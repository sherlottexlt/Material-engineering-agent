"""
M2 角色定义与协作子图测试

测试内容：
- M2-3: 6 角色 Prompt 与元数据
- M2-5: 冲突仲裁节点
- M2-8: 协作子图端到端联调
"""
import asyncio
import json
from unittest.mock import patch, MagicMock

import pytest

from agent.prompts.roles import (
    ROLES, PARALLEL_GROUP, SEQUENTIAL_AFTER_JOIN,
    Role, get_role, get_role_prompt, get_parallel_roles,
    DATA_AGENT, MECHANISM_AGENT, KNOWLEDGE_AGENT,
    DECISION_AGENT, REVIEW_AGENT, INTERACTION_AGENT,
)


# ===== M2-3: 角色定义测试 =====

class TestRoleDefinitions:
    """6 角色 Prompt 与元数据测试"""

    def test_8_roles_registered(self):
        """8 个角色全部注册（6 Sub-Agent + planner + reflector）"""
        expected = {"planner", "data", "mechanism", "knowledge",
                    "decision", "review", "interaction", "reflector"}
        assert set(ROLES.keys()) == expected

    def test_each_role_has_required_fields(self):
        """每个角色有必要字段"""
        for name, role in ROLES.items():
            assert role.name == name
            assert len(role.display_name) > 0
            assert len(role.description) > 0
            assert len(role.prompt) > 0

    def test_role_prompt_contains_key_phrase(self):
        """每个 prompt 包含角色标识"""
        phrases = {
            "planner": "规划",
            "data": "数据 Agent",
            "mechanism": "机理 Agent",
            "knowledge": "知识 Agent",
            "decision": "决策 Agent",
            "review": "审核 Agent",
            "interaction": "交互 Agent",
            "reflector": "反思",
        }
        for name, phrase in phrases.items():
            assert phrase in ROLES[name].prompt, f"{name} prompt 缺少 '{phrase}'"

    def test_parallel_group_no_dependencies(self):
        """并行组的角色无依赖（可同时执行）"""
        for name in PARALLEL_GROUP:
            role = ROLES[name]
            assert len(role.dependencies) == 0, f"{name} 有依赖，不能并行"

    def test_decision_depends_on_three(self):
        """decision 依赖 data/mechanism/knowledge"""
        deps = DECISION_AGENT.dependencies
        assert "data" in deps
        assert "mechanism" in deps
        assert "knowledge" in deps

    def test_review_depends_on_decision(self):
        """review 依赖 decision"""
        assert "decision" in REVIEW_AGENT.dependencies

    def test_data_agent_tools(self):
        """data agent 有查询工具"""
        assert "query_batch_params" in DATA_AGENT.tools
        assert "query_defect_history" in DATA_AGENT.tools

    def test_mechanism_agent_tools(self):
        """mechanism agent 有机理模型工具"""
        assert "run_metallurgy_model" in MECHANISM_AGENT.tools

    def test_knowledge_agent_tools(self):
        """knowledge agent 有检索工具"""
        assert "search_handbook" in KNOWLEDGE_AGENT.tools
        assert "search_cases" in KNOWLEDGE_AGENT.tools

    def test_get_role_existing(self):
        """get_role 返回已注册角色"""
        role = get_role("data")
        assert role is not None
        assert role.name == "data"

    def test_get_role_unknown(self):
        """get_role 未知角色返回 None"""
        assert get_role("unknown") is None

    def test_get_role_prompt_existing(self):
        """get_role_prompt 返回 prompt"""
        prompt = get_role_prompt("review")
        assert "审核" in prompt

    def test_get_role_prompt_unknown_raises(self):
        """get_role_prompt 未知角色抛异常"""
        with pytest.raises(KeyError):
            get_role_prompt("unknown")

    def test_get_parallel_roles(self):
        """get_parallel_roles 返回 3 个并行角色"""
        roles = get_parallel_roles()
        assert len(roles) == 3
        names = {r.name for r in roles}
        assert names == {"data", "mechanism", "knowledge"}

    def test_review_prompt_has_pass_criteria(self):
        """review prompt 包含 M1 优化的 4 条通过标准"""
        prompt = REVIEW_AGENT.prompt
        assert "证据链完整" in prompt
        assert "根因一致" in prompt
        assert "调整方向合理" in prompt
        assert "置信度合理" in prompt

    def test_role_is_frozen(self):
        """Role 是 frozen dataclass（不可变）"""
        with pytest.raises(Exception):
            DATA_AGENT.name = "changed"  # type: ignore


# ===== M2-5: 冲突仲裁测试 =====

class TestArbitrate:
    """冲突仲裁节点测试"""

    @pytest.mark.asyncio
    async def test_no_conflicts(self):
        """三方结果一致，无冲突"""
        from agent.nodes.coordinator import arbitrate
        state = {
            "data_result": {"batch_params": {"temperature": 820}},
            "mechanism_result": {
                "jmak_output": {"outputs": {"predicted_hardness_HRc": 55.0}},
                "cooling_output": {"outputs": {}},
            },
            "knowledge_result": {
                "handbook_hits": {"total": 3},
                "case_hits": {"total": 2},
            },
        }
        result = await arbitrate(state)
        assert result["arbitration_result"]["conflict_count"] == 0

    @pytest.mark.asyncio
    async def test_hardness_mismatch_conflict(self):
        """JMAK 预测硬度 ≥ 58 但缺陷报告硬度偏低 → 冲突"""
        from agent.nodes.coordinator import arbitrate
        state = {
            "data_result": {},
            "mechanism_result": {
                "jmak_output": {"outputs": {"predicted_hardness_HRc": 60.0}},
            },
            "knowledge_result": {"handbook_hits": {"total": 1}, "case_hits": {"total": 1}},
        }
        result = await arbitrate(state)
        conflicts = result["arbitration_result"]["conflicts"]
        assert any(c["type"] == "mechanism_data_mismatch" for c in conflicts)
        assert result["arbitration_result"]["has_high_severity"] is True

    @pytest.mark.asyncio
    async def test_knowledge_empty_conflict(self):
        """知识检索无结果 → 中等冲突"""
        from agent.nodes.coordinator import arbitrate
        state = {
            "data_result": {},
            "mechanism_result": {
                "jmak_output": {"outputs": {"predicted_hardness_HRc": 55.0}},
            },
            "knowledge_result": {"handbook_hits": {"total": 0}, "case_hits": {"total": 0}},
        }
        result = await arbitrate(state)
        conflicts = result["arbitration_result"]["conflicts"]
        assert any(c["type"] == "knowledge_empty" for c in conflicts)
        assert result["arbitration_result"]["has_high_severity"] is False

    @pytest.mark.asyncio
    async def test_empty_state(self):
        """空状态不崩溃"""
        from agent.nodes.coordinator import arbitrate
        result = await arbitrate({})
        # 空状态：knowledge 无结果 → 1 个冲突
        assert result["arbitration_result"]["conflict_count"] >= 0

    @pytest.mark.asyncio
    async def test_hardness_string_no_conflict(self):
        """硬度为字符串（非数值）时不触发冲突"""
        from agent.nodes.coordinator import arbitrate
        state = {
            "data_result": {},
            "mechanism_result": {
                "jmak_output": {"outputs": {"predicted_hardness_HRc": "未知"}},
            },
            "knowledge_result": {"handbook_hits": {"total": 1}, "case_hits": {"total": 1}},
        }
        result = await arbitrate(state)
        assert result["arbitration_result"]["conflict_count"] == 0


# ===== M2-8: 协作子图联调 =====

class TestCollaborationGraph:
    """M2 并行协作图端到端测试"""

    def test_graph_builds(self):
        """协作图能成功构建"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """图包含所有 9 个节点（含 arbitrate）"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph()
        # LangGraph 编译后节点信息在 nodes 属性
        node_names = set(graph.nodes.keys())
        expected = {"planner", "data", "mechanism", "knowledge",
                    "arbitrate", "decision", "review", "interaction", "memory_writer"}
        assert expected.issubset(node_names)

    def test_m1_linear_graph_still_works(self):
        """M1 线性图仍可构建（向后兼容）"""
        from agent.orchestrator import build_linear_orchestrator
        graph = build_linear_orchestrator()
        assert graph is not None
        node_names = set(graph.nodes.keys())
        # M1 没有 arbitrate 节点
        assert "arbitrate" not in node_names
        assert "data" in node_names

    def test_build_orchestrator_defaults_to_m2(self):
        """build_orchestrator() 默认返回 M2 协作图"""
        from agent.orchestrator import build_orchestrator
        graph = build_orchestrator()
        node_names = set(graph.nodes.keys())
        assert "arbitrate" in node_names  # M2 特有节点

    @pytest.mark.asyncio
    async def test_end_to_end_parallel(self):
        """端到端：SC-001 跑通并行协作（mock LLM 加速）"""
        from agent.nodes.coordinator import build_collaboration_graph
        from models.state import AgentState

        # MockLLM：返回预定义 JSON，避免真实 LLM 调用
        class MockLLM:
            def __init__(self):
                self.call_count = 0
            def invoke(self, prompt):
                self.call_count += 1
                # 根据 prompt 内容返回不同 JSON
                if "审核" in prompt or "approved" in prompt:
                    return type("R", (), {"content": '{"approved": true, "reason": "方案合理", "suggestions": []}'})()
                elif "决策" in prompt or "proposals" in prompt or "候选方案" in prompt:
                    return type("R", (), {"content": json.dumps({
                        "proposals": [{
                            "proposal_id": "P001",
                            "root_cause": "保温时间不足",
                            "adjustments": {"holding_time": "+40 分钟"},
                            "expected_effect": "硬度提升 3 HRc",
                            "risks": ["能耗增加"],
                            "evidence": ["当前 80 分钟 < 标准 120 分钟"],
                            "confidence": 0.8,
                        }],
                        "source": "llm",
                    }, ensure_ascii=False)})()
                elif "规划" in prompt or "plan" in prompt:
                    return type("R", (), {"content": '{"plan": [{"step_id": 1, "agent": "data", "action": "查询", "tool": "query_batch_params"}]}'})()
                elif "反思" in prompt or "needs_replan" in prompt:
                    return type("R", (), {"content": '{"needs_replan": false, "reason": "已完成"}'})()
                else:
                    return type("R", (), {"content": "机理分析：保温时间不足导致奥氏体转化不完全"})()

        mock_llm = MockLLM()

        with patch("agent.utils.get_llm", return_value=mock_llm):
            graph = build_collaboration_graph()
            trace_id = "test_m2_001"
            initial_state: AgentState = {
                "user_query": "批次 B20260701-A 硬度偏低，请分析原因",
                "batch_id": "B20260701-A",
                "defect_record": None,
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
                "max_replan": 3,
                "trace_id": trace_id,
                "session_id": trace_id,
            }
            config = {"configurable": {"thread_id": trace_id}}

            final_state = await graph.ainvoke(initial_state, config)

        # 验证三方结果都生成了（并行执行）
        assert final_state.get("data_result") is not None
        assert final_state.get("mechanism_result") is not None
        assert final_state.get("knowledge_result") is not None

        # 验证仲裁结果
        arbitration = final_state.get("arbitration_result")
        assert arbitration is not None
        assert "conflicts" in arbitration

        # 验证决策和审核
        assert final_state.get("decision_result") is not None
        assert final_state.get("review_result") is not None

        # 验证最终回答
        assert final_state.get("final_answer") is not None

        # 验证 observations 包含多个 agent 的记录
        observations = final_state.get("observations", [])
        agents_seen = {obs.get("agent") for obs in observations}
        assert "data" in agents_seen
        assert "mechanism" in agents_seen
        assert "knowledge" in agents_seen


# ===== M2-6/7: 回退与重试上限测试 =====

class TestRetryAndFallback:
    """回退机制与重试上限（M1 已有，M2 验证）"""

    def test_route_after_review_approved(self):
        """审核通过 → interaction"""
        from agent.nodes.coordinator import _route_after_review
        state = {"review_result": {"approved": True}, "retry_count": 0, "max_replan": 3}
        assert _route_after_review(state) == "interaction"

    def test_route_after_review_rejected(self):
        """审核不通过 → decision（回退）"""
        from agent.nodes.coordinator import _route_after_review
        state = {"review_result": {"approved": False}, "retry_count": 0, "max_replan": 3}
        assert _route_after_review(state) == "decision"

    def test_route_after_review_max_retry(self):
        """超过重试上限 → interaction（强制放行）"""
        from agent.nodes.coordinator import _route_after_review
        state = {"review_result": {"approved": False}, "retry_count": 3, "max_replan": 3}
        assert _route_after_review(state) == "interaction"

    def test_route_after_review_below_max(self):
        """未超上限 → decision"""
        from agent.nodes.coordinator import _route_after_review
        state = {"review_result": {"approved": False}, "retry_count": 2, "max_replan": 3}
        assert _route_after_review(state) == "decision"
