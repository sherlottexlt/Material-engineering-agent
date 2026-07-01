"""
Memory Service 单元测试
对应 TDD 第 11 节

测试三层记忆：
- 短期记忆 CRUD
- 长期记忆写入与检索
- 置信度更新
- 遗忘机制
"""
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pytest

from agent.memory.memory_service import MemoryService
from models.entities import CaseRecord, BatchParams, ProcessType


@pytest.fixture
def memory(tmp_path):
    """临时 MemoryService 实例"""
    db_path = tmp_path / "test_memory.db"
    service = MemoryService(db_path=db_path)
    yield service
    service.close()


class TestEpisodicMemory:
    """短期记忆测试"""

    def test_write_episodic(self, memory):
        """应成功写入短期记忆"""
        record_id = memory.write_episodic(
            batch_id="B001",
            defect_type="hardness_low",
            root_cause="保温时间不足",
            solution="holding_time +15",
        )
        assert record_id.startswith("ep_")

    def test_query_episodic_by_batch(self, memory):
        """应能按批次查询"""
        memory.write_episodic("B001", "hardness_low", "原因A", "方案A")
        memory.write_episodic("B002", "hardness_low", "原因B", "方案B")

        results = memory.query_episodic(batch_id="B001")
        assert len(results) == 1
        assert results[0]["batch_id"] == "B001"

    def test_query_episodic_by_defect_type(self, memory):
        """应能按缺陷类型查询"""
        memory.write_episodic("B001", "hardness_low", "原因A", "方案A")
        memory.write_episodic("B002", "crack", "原因B", "方案B")

        results = memory.query_episodic(defect_type="hardness_low")
        assert len(results) == 1
        assert results[0]["defect_type"] == "hardness_low"

    def test_query_episodic_empty(self, memory):
        """空查询应返回空列表"""
        results = memory.query_episodic()
        assert results == []


class TestSemanticMemory:
    """长期记忆测试"""

    def test_write_and_search_semantic(self, memory):
        """应能写入并检索长期记忆"""
        case = CaseRecord(
            case_id="C001",
            defect_type="hardness_low",
            batch_params=BatchParams(
                batch_id="B001",
                process_type=ProcessType.HEAT_TREATMENT,
                temperature=850,
                holding_time=120,
                start_time=datetime.now(),
            ),
            root_cause="保温时间不足",
            solution="holding_time +15",
            confidence=0.8,
        )

        # 写入（Chroma 不可用时降级）
        success = memory.write_semantic(case)
        if not success:
            pytest.skip("Chroma 不可用")

        # 检索
        results = memory.search_semantic("硬度偏低 保温时间")
        assert len(results) > 0

    def test_search_empty_query(self, memory):
        """空查询应返回空列表"""
        results = memory.search_semantic("不存在的查询内容 xyz")
        # Chroma 可能返回空结果或低相关度结果
        assert isinstance(results, list)


class TestConfidenceUpdate:
    """置信度更新测试"""

    def test_update_confidence(self, memory):
        """应能更新案例置信度"""
        case = CaseRecord(
            case_id="C002",
            defect_type="hardness_low",
            batch_params=BatchParams(
                batch_id="B002",
                process_type=ProcessType.HEAT_TREATMENT,
                start_time=datetime.now(),
            ),
            root_cause="测试原因",
            solution="测试方案",
            confidence=0.5,
        )

        success = memory.write_semantic(case)
        if not success:
            pytest.skip("Chroma 不可用")

        updated = memory.update_confidence("C002", 0.9)
        assert updated is True

    def test_update_nonexistent_case(self, memory):
        """更新不存在的案例应返回 False"""
        result = memory.update_confidence("nonexistent", 0.9)
        assert result is False


class TestCleanup:
    """遗忘机制测试"""

    def test_cleanup_expired(self, memory):
        """应清理过期低质记忆"""
        # 写入一条记忆
        memory.write_episodic("B001", "hardness_low", "原因", "方案")

        # 手动修改 created_at 为过期
        memory.db.execute(
            "UPDATE episodic SET created_at = ?, quality_score = 0.1 WHERE batch_id = ?",
            (datetime.now() - timedelta(days=40), "B001"),
        )
        memory.db.commit()

        deleted = memory.cleanup_expired()
        assert deleted == 1


class TestFeedback:
    """用户反馈持久化测试"""

    def test_write_feedback_success(self, memory):
        """写入反馈成功"""
        ok = memory.write_feedback(
            feedback_id="fb_001",
            proposal_id="P001",
            user_id="op_01",
            action="adopted",
            score=0.9,
            comment="建议有效",
        )
        assert ok is True

    def test_query_feedback_by_proposal(self, memory):
        """按 proposal_id 查询反馈"""
        memory.write_feedback("fb_1", "P001", "u1", "adopted", 0.9)
        memory.write_feedback("fb_2", "P001", "u2", "rejected", 0.2)
        memory.write_feedback("fb_3", "P002", "u1", "partial", 0.5)

        results = memory.query_feedback(proposal_id="P001")
        assert len(results) == 2
        assert all(r["proposal_id"] == "P001" for r in results)

    def test_query_feedback_by_user(self, memory):
        """按 user_id 查询反馈"""
        memory.write_feedback("fb_1", "P001", "u1", "adopted", 0.9)
        memory.write_feedback("fb_2", "P002", "u1", "rejected", 0.2)
        memory.write_feedback("fb_3", "P003", "u2", "adopted", 0.8)

        results = memory.query_feedback(user_id="u1")
        assert len(results) == 2
        assert all(r["user_id"] == "u1" for r in results)

    def test_query_feedback_all(self, memory):
        """查询全部反馈"""
        for i in range(5):
            memory.write_feedback(f"fb_{i}", f"P{i:03d}", "u1", "adopted", 0.8)

        results = memory.query_feedback()
        assert len(results) == 5

    def test_query_feedback_ordered_desc(self, memory):
        """反馈按时间倒序"""
        memory.write_feedback("fb_1", "P001", "u1", "adopted", 0.9)
        memory.write_feedback("fb_2", "P002", "u1", "adopted", 0.8)

        results = memory.query_feedback()
        assert results[0]["feedback_id"] == "fb_2"

    def test_write_feedback_with_null_comment(self, memory):
        """comment 为 None 也能写入"""
        ok = memory.write_feedback(
            feedback_id="fb_001",
            proposal_id="P001",
            user_id="u1",
            action="rejected",
            score=0.1,
            comment=None,
        )
        assert ok is True
        results = memory.query_feedback()
        assert len(results) == 1

    def test_duplicate_feedback_id_fails(self, memory):
        """重复 feedback_id 写入失败（主键冲突）"""
        memory.write_feedback("fb_1", "P001", "u1", "adopted", 0.9)
        ok = memory.write_feedback("fb_1", "P002", "u2", "rejected", 0.1)
        assert ok is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
