"""
节点降级逻辑单测（M1-12 补充）

测试各节点的降级/工具函数（不需要 LLM）：
- planner._default_plan
- decision_agent._rule_based_proposals
- interaction_agent._format_proposals_fallback
"""
import pytest

from agent.nodes.planner import _default_plan
from agent.nodes.decision_agent import _rule_based_proposals
from agent.nodes.interaction_agent import _format_proposals_fallback


# ===== Planner：_default_plan =====
class TestDefaultPlan:
    """测试 Planner 降级方案"""

    def test_returns_4_steps(self):
        plan = _default_plan("B001")
        assert len(plan) == 4

    def test_step_ids_sequential(self):
        plan = _default_plan("B001")
        ids = [s["step_id"] for s in plan]
        assert ids == [1, 2, 3, 4]

    def test_batch_id_in_action(self):
        plan = _default_plan("B20260701-A")
        assert "B20260701-A" in plan[0]["action"]

    def test_agents_correct(self):
        plan = _default_plan("B001")
        agents = [s["agent"] for s in plan]
        assert agents == ["data", "mechanism", "knowledge", "decision"]

    def test_all_steps_pending(self):
        plan = _default_plan("B001")
        assert all(s["status"] == "pending" for s in plan)

    def test_empty_batch_id(self):
        plan = _default_plan("")
        assert len(plan) == 4  # 仍然返回 4 步


# ===== Decision Agent：_rule_based_proposals =====
class TestRuleBasedProposals:
    """测试 Decision Agent 规则降级方案"""

    def _make_data(self, temperature=850, holding_time=120, cooling_rate=5.0):
        return {
            "batch_params": {
                "temperature": temperature,
                "holding_time": holding_time,
                "cooling_rate": cooling_rate,
            }
        }

    def _make_mechanism(self):
        return {
            "jmak_output": {"outputs": {"fraction_transformed": 0.85}},
            "cooling_output": {"outputs": {"hardness_drop": 3.0}},
        }

    def test_holding_time_low_generates_proposal(self):
        """保温时间 < 120 → 生成保温方案"""
        data = self._make_data(holding_time=80)
        result = _rule_based_proposals(data, self._make_mechanism())
        proposals = result["proposals"]
        assert len(proposals) >= 1
        assert any("保温" in p["root_cause"] for p in proposals)

    def test_cooling_rate_low_generates_proposal(self):
        """冷却速率 < 5.0 → 生成冷却方案"""
        data = self._make_data(cooling_rate=3.0)
        result = _rule_based_proposals(data, self._make_mechanism())
        proposals = result["proposals"]
        assert any("冷却" in p["root_cause"] for p in proposals)

    def test_temperature_low_generates_proposal(self):
        """温度 < 850 → 生成温度方案"""
        data = self._make_data(temperature=820)
        result = _rule_based_proposals(data, self._make_mechanism())
        proposals = result["proposals"]
        assert any("温度" in p["root_cause"] for p in proposals)

    def test_multiple_issues_multiple_proposals(self):
        """多个参数偏离 → 生成多个方案"""
        data = self._make_data(temperature=820, holding_time=80, cooling_rate=3.0)
        result = _rule_based_proposals(data, self._make_mechanism())
        assert len(result["proposals"]) == 3

    def test_all_normal_generates_default(self):
        """所有参数正常 → 生成默认方案"""
        data = self._make_data()  # 全部标准值
        result = _rule_based_proposals(data, self._make_mechanism())
        proposals = result["proposals"]
        assert len(proposals) == 1
        assert proposals[0]["proposal_id"] == "P000"
        assert result["source"] == "rule_based"

    def test_proposal_has_required_fields(self):
        """方案包含必要字段"""
        data = self._make_data(holding_time=80)
        result = _rule_based_proposals(data, self._make_mechanism())
        p = result["proposals"][0]
        for field in ["proposal_id", "root_cause", "adjustments", "expected_effect",
                       "risks", "evidence", "confidence"]:
            assert field in p

    def test_confidence_in_range(self):
        """置信度在 0-1 之间"""
        data = self._make_data(temperature=820, holding_time=80, cooling_rate=3.0)
        result = _rule_based_proposals(data, self._make_mechanism())
        for p in result["proposals"]:
            assert 0.0 <= p["confidence"] <= 1.0

    def test_evidence_contains_values(self):
        """evidence 包含具体数值"""
        data = self._make_data(holding_time=80)
        result = _rule_based_proposals(data, self._make_mechanism())
        p = result["proposals"][0]
        evidence_text = " ".join(p["evidence"])
        assert "80" in evidence_text  # 当前 holding_time
        assert "120" in evidence_text  # 标准

    def test_empty_data(self):
        """空 data_result → 默认方案"""
        result = _rule_based_proposals({}, self._make_mechanism())
        # batch_params 为空时，用默认值 850/120/5.0，全部正常 → P000
        proposals = result["proposals"]
        assert len(proposals) == 1
        assert proposals[0]["proposal_id"] == "P000"


# ===== Interaction Agent：_format_proposals_fallback =====
class TestFormatProposalsFallback:
    """测试 Interaction Agent 降级格式化"""

    def _make_proposal(self, confidence=0.85, root_cause="保温时间不足"):
        return {
            "proposals": [
                {
                    "proposal_id": "P001",
                    "root_cause": root_cause,
                    "adjustments": {"holding_time": "+30 分钟"},
                    "expected_effect": "硬度提升 3 HRc",
                    "risks": ["可能增加能耗"],
                    "evidence": ["当前 80 分钟 < 标准 120 分钟"],
                    "confidence": confidence,
                }
            ]
        }

    def test_empty_proposals(self):
        """无方案 → 提示人工复核"""
        result = _format_proposals_fallback({}, {"approved": True})
        assert "人工复核" in result

    def test_no_proposals_key(self):
        """无 proposals 字段 → 提示人工复核"""
        result = _format_proposals_fallback({}, {"approved": True})
        assert "人工复核" in result

    def test_contains_root_cause(self):
        """输出包含根因"""
        decision = self._make_proposal(root_cause="保温时间严重不足")
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "保温时间严重不足" in result

    def test_contains_adjustments(self):
        """输出包含参数调整"""
        decision = self._make_proposal()
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "holding_time" in result
        assert "+30 分钟" in result

    def test_confidence_high(self):
        """高置信度标注"""
        decision = self._make_proposal(confidence=0.9)
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "高" in result

    def test_confidence_medium(self):
        """中置信度标注"""
        decision = self._make_proposal(confidence=0.6)
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "中" in result

    def test_confidence_low(self):
        """低置信度标注"""
        decision = self._make_proposal(confidence=0.3)
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "低" in result

    def test_review_not_approved_shows_warning(self):
        """审核未通过 → 显示警告"""
        decision = self._make_proposal()
        result = _format_proposals_fallback(
            decision,
            {"approved": False, "reason": "证据链不完整"}
        )
        assert "审核提示" in result or "证据链不完整" in result

    def test_review_approved_no_warning(self):
        """审核通过 → 无警告"""
        decision = self._make_proposal()
        result = _format_proposals_fallback(
            decision,
            {"approved": True, "reason": ""}
        )
        assert "审核提示" not in result

    def test_contains_risks(self):
        """输出包含风险"""
        decision = self._make_proposal()
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "可能增加能耗" in result

    def test_contains_evidence(self):
        """输出包含证据链"""
        decision = self._make_proposal()
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "80 分钟" in result

    def test_contains_action_options(self):
        """输出包含操作选项"""
        decision = self._make_proposal()
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "采纳" in result or "拒绝" in result

    def test_multiple_proposals(self):
        """多方案输出"""
        decision = {
            "proposals": [
                self._make_proposal(confidence=0.9)["proposals"][0],
                self._make_proposal(confidence=0.6, root_cause="温度偏低")["proposals"][0],
            ]
        }
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "保温时间不足" in result
        assert "温度偏低" in result

    def test_proposal_no_adjustments(self):
        """方案无调整项 → 显示人工检查"""
        decision = {
            "proposals": [{
                "proposal_id": "P001",
                "root_cause": "未知",
                "adjustments": {},
                "expected_effect": "未知",
                "risks": [],
                "evidence": [],
                "confidence": 0.4,
            }]
        }
        result = _format_proposals_fallback(decision, {"approved": True})
        assert "人工检查" in result
