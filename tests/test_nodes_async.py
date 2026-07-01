"""
节点函数本身的单测（M1-12 补充）

测各 async 节点函数的降级行为、LLM 输出解析、retry/feedback 逻辑。
不需要真实 LLM 调用，用 mock 替换 get_llm / get_prompt / get_process_constraints。
"""
import asyncio
import json
from unittest.mock import patch

import pytest


# ===== Mock 工具 =====


class MockLLM:
    """假 LLM，invoke/ainvoke 返回固定内容或抛异常"""

    def __init__(self, content: str = "", raise_error: bool = False):
        self.content = content
        self.raise_error = raise_error
        self.invoked_prompts = []  # 记录被调用的 prompt，便于断言

    def invoke(self, prompt):
        if self.raise_error:
            raise Exception("mock LLM error")
        self.invoked_prompts.append(prompt)
        return type("Response", (), {"content": self.content})()

    async def ainvoke(self, prompt):
        """异步版本，与 invoke 行为一致"""
        return self.invoke(prompt)


def _mock_get_prompt(name: str) -> str:
    """假 prompt 模板（不含占位符，format 不会报错）"""
    return f"mock prompt for {name}"


def _mock_get_constraints(process_type: str = "heat_treatment") -> dict:
    return {"temperature_min": 800, "temperature_max": 1100}


# ===== Decision Agent 节点 =====


class TestDecisionAgentNode:
    """测试 decision_agent async 函数"""

    def _make_state(self, **overrides):
        base = {
            "data_result": {"batch_params": {"temperature": 830, "holding_time": 80, "cooling_rate": 3.0}},
            "mechanism_result": {},
            "knowledge_result": {},
            "review_result": {},
            "retry_count": 0,
        }
        base.update(overrides)
        return base

    @patch("agent.nodes.decision_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.decision_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.decision_agent.get_llm")
    def test_degrades_to_rule_based_on_llm_error(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 抛异常 → 降级到 _rule_based_proposals"""
        from agent.nodes.decision_agent import decision_agent
        mock_get_llm.return_value = MockLLM(raise_error=True)
        result = asyncio.run(decision_agent(self._make_state()))
        assert "decision_result" in result
        assert result["decision_result"]["source"] == "rule_based"
        assert len(result["decision_result"]["proposals"]) > 0

    @patch("agent.nodes.decision_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.decision_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.decision_agent.get_llm")
    def test_parses_llm_json_proposals(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回有效 JSON proposals → 解析成功"""
        from agent.nodes.decision_agent import decision_agent
        llm_content = json.dumps({
            "proposals": [
                {"proposal_id": "P001", "root_cause": "保温时间不足", "confidence": 0.85}
            ]
        }, ensure_ascii=False)
        mock_get_llm.return_value = MockLLM(content=llm_content)
        result = asyncio.run(decision_agent(self._make_state()))
        assert result["decision_result"]["source"] == "llm"
        assert len(result["decision_result"]["proposals"]) == 1
        assert result["decision_result"]["proposals"][0]["root_cause"] == "保温时间不足"

    @patch("agent.nodes.decision_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.decision_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.decision_agent.get_llm")
    def test_falls_back_on_non_json(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回纯文本 → 降级到规则方案"""
        from agent.nodes.decision_agent import decision_agent
        mock_get_llm.return_value = MockLLM(content="这不是 JSON 格式的回答")
        result = asyncio.run(decision_agent(self._make_state()))
        assert result["decision_result"]["source"] == "rule_based"

    @patch("agent.nodes.decision_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.decision_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.decision_agent.get_llm")
    def test_retry_count_adds_feedback_to_prompt(self, mock_get_llm, mock_prompt, mock_constraints):
        """retry_count > 0 + review_result → prompt 包含审核反馈"""
        from agent.nodes.decision_agent import decision_agent
        mock_llm = MockLLM(content='{"proposals": []}')
        mock_get_llm.return_value = mock_llm
        state = self._make_state(
            retry_count=1,
            review_result={"reason": "证据不足", "suggestions": ["补充数值比对"]},
        )
        asyncio.run(decision_agent(state))
        # 检查 LLM 收到的 prompt 包含审核反馈
        assert len(mock_llm.invoked_prompts) == 1
        assert "上次审核未通过" in mock_llm.invoked_prompts[0]
        assert "证据不足" in mock_llm.invoked_prompts[0]

    @patch("agent.nodes.decision_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.decision_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.decision_agent.get_llm")
    def test_no_feedback_on_first_try(self, mock_get_llm, mock_prompt, mock_constraints):
        """retry_count=0 → prompt 不包含审核反馈"""
        from agent.nodes.decision_agent import decision_agent
        mock_llm = MockLLM(content='{"proposals": []}')
        mock_get_llm.return_value = mock_llm
        asyncio.run(decision_agent(self._make_state()))
        assert "上次审核未通过" not in mock_llm.invoked_prompts[0]

    @patch("agent.nodes.decision_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.decision_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.decision_agent.get_llm")
    def test_empty_state_no_crash(self, mock_get_llm, mock_prompt, mock_constraints):
        """空 state → 不崩溃，降级到规则方案"""
        from agent.nodes.decision_agent import decision_agent
        mock_get_llm.return_value = MockLLM(raise_error=True)
        result = asyncio.run(decision_agent({}))
        assert "decision_result" in result
        assert result["decision_result"]["source"] == "rule_based"

    @patch("agent.nodes.decision_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.decision_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.decision_agent.get_llm")
    def test_llm_json_without_proposals_key(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回 JSON 但无 proposals 字段 → 返回空 proposals"""
        from agent.nodes.decision_agent import decision_agent
        mock_get_llm.return_value = MockLLM(content='{"other": "value"}')
        result = asyncio.run(decision_agent(self._make_state()))
        assert result["decision_result"]["source"] == "llm"
        assert result["decision_result"]["proposals"] == []


# ===== Interaction Agent 节点 =====


class TestInteractionAgentNode:
    """测试 interaction_agent async 函数"""

    @patch("agent.nodes.interaction_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.interaction_agent.get_llm")
    def test_degrades_to_fallback_on_llm_error(self, mock_get_llm, mock_prompt):
        """LLM 抛异常 → 降级到 _format_proposals_fallback"""
        from agent.nodes.interaction_agent import interaction_agent
        mock_get_llm.return_value = MockLLM(raise_error=True)
        state = {
            "decision_result": {"proposals": [
                {"proposal_id": "P001", "root_cause": "测试", "confidence": 0.8}
            ]},
            "review_result": {"approved": True},
        }
        result = asyncio.run(interaction_agent(state))
        assert "final_answer" in result
        assert "测试" in result["final_answer"]
        assert "observations" in result

    @patch("agent.nodes.interaction_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.interaction_agent.get_llm")
    def test_returns_final_answer_from_llm(self, mock_get_llm, mock_prompt):
        """LLM 返回文本 → final_answer = 文本"""
        from agent.nodes.interaction_agent import interaction_agent
        mock_get_llm.return_value = MockLLM(content="这是 LLM 生成的操作员友好回答")
        state = {
            "decision_result": {"proposals": [{"proposal_id": "P001", "root_cause": "x"}]},
            "review_result": {},
        }
        result = asyncio.run(interaction_agent(state))
        assert result["final_answer"] == "这是 LLM 生成的操作员友好回答"

    @patch("agent.nodes.interaction_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.interaction_agent.get_llm")
    def test_empty_proposals_returns_default(self, mock_get_llm, mock_prompt):
        """空 proposals → 返回默认提示，不调 LLM"""
        from agent.nodes.interaction_agent import interaction_agent
        mock_llm = MockLLM(content="不应被调用")
        mock_get_llm.return_value = mock_llm
        state = {"decision_result": {"proposals": []}, "review_result": {}}
        result = asyncio.run(interaction_agent(state))
        assert "人工复核" in result["final_answer"]
        assert len(mock_llm.invoked_prompts) == 0  # LLM 未被调用

    @patch("agent.nodes.interaction_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.interaction_agent.get_llm")
    def test_no_decision_result_returns_default(self, mock_get_llm, mock_prompt):
        """state 无 decision_result → 返回默认提示"""
        from agent.nodes.interaction_agent import interaction_agent
        mock_get_llm.return_value = MockLLM(content="x")
        result = asyncio.run(interaction_agent({}))
        assert "人工复核" in result["final_answer"]

    @patch("agent.nodes.interaction_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.interaction_agent.get_llm")
    def test_returns_observations(self, mock_get_llm, mock_prompt):
        """返回值包含 observations 列表"""
        from agent.nodes.interaction_agent import interaction_agent
        mock_get_llm.return_value = MockLLM(content="回答")
        state = {"decision_result": {"proposals": [{"proposal_id": "P001"}]}, "review_result": {}}
        result = asyncio.run(interaction_agent(state))
        assert "observations" in result
        assert isinstance(result["observations"], list)
        assert len(result["observations"]) == 1
        assert result["observations"][0]["agent"] == "interaction"


# ===== Reflector 节点 =====


class TestReflectorNode:
    """测试 reflector async 函数"""

    @patch("agent.nodes.reflector.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.reflector.get_llm")
    def test_returns_needs_replan_false_on_llm_error(self, mock_get_llm, mock_prompt):
        """LLM 抛异常 → needs_replan=False"""
        from agent.nodes.reflector import reflector
        mock_get_llm.return_value = MockLLM(raise_error=True)
        result = asyncio.run(reflector({"plan": [], "current_step": 0, "observations": []}))
        assert result["needs_replan"] is False

    @patch("agent.nodes.reflector.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.reflector.get_llm")
    def test_parses_llm_json_true(self, mock_get_llm, mock_prompt):
        """LLM 返回 {"needs_replan": true} → needs_replan=True"""
        from agent.nodes.reflector import reflector
        mock_get_llm.return_value = MockLLM(content='{"needs_replan": true, "reason": "信息缺失"}')
        result = asyncio.run(reflector({"plan": [], "current_step": 1, "observations": []}))
        assert result["needs_replan"] is True

    @patch("agent.nodes.reflector.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.reflector.get_llm")
    def test_parses_llm_json_false(self, mock_get_llm, mock_prompt):
        """LLM 返回 {"needs_replan": false} → needs_replan=False"""
        from agent.nodes.reflector import reflector
        mock_get_llm.return_value = MockLLM(content='{"needs_replan": false, "reason": "已完成"}')
        result = asyncio.run(reflector({"plan": [], "current_step": 4, "observations": []}))
        assert result["needs_replan"] is False
        assert "retry_count" not in result  # 不重规划时不更新 retry_count

    @patch("agent.nodes.reflector.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.reflector.get_llm")
    def test_needs_replan_increments_retry(self, mock_get_llm, mock_prompt):
        """needs_replan=True → retry_count+1"""
        from agent.nodes.reflector import reflector
        mock_get_llm.return_value = MockLLM(content='{"needs_replan": true}')
        result = asyncio.run(reflector({"plan": [], "current_step": 0, "observations": [], "retry_count": 2}))
        assert result["retry_count"] == 3

    @patch("agent.nodes.reflector.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.reflector.get_llm")
    def test_json_parse_error(self, mock_get_llm, mock_prompt):
        """LLM 返回非 JSON → needs_replan=False"""
        from agent.nodes.reflector import reflector
        mock_get_llm.return_value = MockLLM(content="这不是 JSON")
        result = asyncio.run(reflector({"plan": [], "current_step": 0, "observations": []}))
        assert result["needs_replan"] is False

    @patch("agent.nodes.reflector.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.reflector.get_llm")
    def test_empty_state_no_crash(self, mock_get_llm, mock_prompt):
        """空 state → 不崩溃"""
        from agent.nodes.reflector import reflector
        mock_get_llm.return_value = MockLLM(content='{"needs_replan": false}')
        result = asyncio.run(reflector({}))
        assert result["needs_replan"] is False


# ===== Review Agent 节点 =====


class TestReviewAgentNode:
    """测试 review_agent async 函数"""

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_degrades_to_approved_on_llm_error(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 抛异常 → 降级放行 approved=True"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(raise_error=True)
        result = asyncio.run(review_agent({"decision_result": {"proposals": []}, "retry_count": 0}))
        assert result["review_result"]["approved"] is True
        assert "降级放行" in result["review_result"]["reason"]

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_parses_llm_json_approved(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回 {"approved": true} → approved=True"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(content='{"approved": true, "reason": "证据充分", "suggestions": []}')
        result = asyncio.run(review_agent({"decision_result": {"proposals": []}, "retry_count": 0}))
        assert result["review_result"]["approved"] is True
        assert result["review_result"]["reason"] == "证据充分"

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_parses_llm_json_rejected(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回 {"approved": false} → approved=False + retry_count+1"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(content='{"approved": false, "reason": "根因不一致", "suggestions": ["修正根因"]}'
        )
        result = asyncio.run(review_agent({"decision_result": {"proposals": []}, "retry_count": 0}))
        assert result["review_result"]["approved"] is False
        assert result["review_result"]["reason"] == "根因不一致"
        assert result["retry_count"] == 1  # 拒绝时递增

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_approved_does_not_increment_retry(self, mock_get_llm, mock_prompt, mock_constraints):
        """approved=True → 不递增 retry_count"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(content='{"approved": true, "reason": "ok"}')
        result = asyncio.run(review_agent({"decision_result": {}, "retry_count": 0}))
        assert "retry_count" not in result  # 通过时不更新 retry_count

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_keyword_fallback_pass(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回非 JSON 但含"通过" → approved=True"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(content="方案审核通过，可以执行")
        result = asyncio.run(review_agent({"decision_result": {}, "retry_count": 0}))
        assert result["review_result"]["approved"] is True

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_keyword_fallback_fail(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回非 JSON 且不含"通过" → approved=False"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(content="方案有问题，需要修改")
        result = asyncio.run(review_agent({"decision_result": {}, "retry_count": 0}))
        assert result["review_result"]["approved"] is False
        assert result["retry_count"] == 1

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_empty_proposal_no_crash(self, mock_get_llm, mock_prompt, mock_constraints):
        """空 proposal → 不崩溃"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(content='{"approved": true, "reason": "ok"}')
        result = asyncio.run(review_agent({}))
        assert result["review_result"]["approved"] is True

    @patch("agent.nodes.review_agent.get_process_constraints", side_effect=_mock_get_constraints)
    @patch("agent.nodes.review_agent.get_prompt", side_effect=_mock_get_prompt)
    @patch("agent.nodes.review_agent.get_llm")
    def test_json_with_code_block_wrapper(self, mock_get_llm, mock_prompt, mock_constraints):
        """LLM 返回 ```json 代码块包裹的 JSON → 能解析"""
        from agent.nodes.review_agent import review_agent
        mock_get_llm.return_value = MockLLM(
            content='```json\n{"approved": true, "reason": "证据完整", "suggestions": []}\n```'
        )
        result = asyncio.run(review_agent({"decision_result": {}, "retry_count": 0}))
        assert result["review_result"]["approved"] is True
        assert result["review_result"]["reason"] == "证据完整"
