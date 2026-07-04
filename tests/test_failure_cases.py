"""
M5-4 失败案例归集测试

测试三层：
1. FailureCaseCollector 收集逻辑（低分案例 + 反效果跟踪 + 被拒绝反馈）
2. FailureCaseCollector 查询/统计/状态更新
3. API 端点（/api/v1/failures/* 共 5 个）
"""
import sys
import types
from datetime import datetime, timedelta

import pytest

# mock langchain_openai.ChatOpenAI（sandbox 中模块存在但缺 ChatOpenAI 属性）
if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = types.ModuleType("langchain_openai")
_mock_lc = sys.modules["langchain_openai"]
if not hasattr(_mock_lc, "ChatOpenAI"):
    _mock_lc.ChatOpenAI = type("ChatOpenAI", (), {"__init__": lambda self, **kw: None})


# ===== 轻量级 Chroma collection 模拟 =====


class _FakeChromaCollection:
    """模拟 Chroma collection 的内存实现，覆盖 get/add/update/delete 接口。"""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def add(self, ids, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._store[doc_id] = {
                "document": documents[i] if i < len(documents) else "",
                "metadata": dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
            }

    def get(self, ids=None, limit=None, include=None, where=None):
        if ids is not None:
            out_ids, out_docs, out_metas = [], [], []
            for doc_id in ids:
                if doc_id in self._store:
                    out_ids.append(doc_id)
                    out_docs.append(self._store[doc_id]["document"])
                    out_metas.append(dict(self._store[doc_id]["metadata"]))
            return {"ids": out_ids, "documents": out_docs, "metadatas": out_metas}

        items = list(self._store.items())
        # 服务端 where 过滤
        if where:
            key = next(iter(where.keys()))
            cond = where[key]
            if isinstance(cond, dict) and "$in" in cond:
                allowed = set(cond["$in"])
                items = [(i, v) for i, v in items if v["metadata"].get(key) in allowed]
            else:
                items = [(i, v) for i, v in items if v["metadata"].get(key) == cond]
        if limit is not None:
            items = items[:limit]
        return {
            "ids": [doc_id for doc_id, _ in items],
            "documents": [item["document"] for _, item in items],
            "metadatas": [dict(item["metadata"]) for _, item in items],
        }

    def update(self, ids, metadatas=None, documents=None):
        for i, doc_id in enumerate(ids):
            if doc_id not in self._store:
                continue
            if metadatas is not None and i < len(metadatas):
                self._store[doc_id]["metadata"] = dict(metadatas[i])
            if documents is not None and i < len(documents):
                self._store[doc_id]["document"] = documents[i]

    def delete(self, ids):
        for doc_id in ids:
            self._store.pop(doc_id, None)


# ===== Fixtures =====


@pytest.fixture
def memory(tmp_path):
    """临时 MemoryService（无 Chroma，降级模式）"""
    from agent.memory.memory_service import MemoryService
    db_path = tmp_path / "test_failure.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def memory_with_chroma(tmp_path):
    """带 FakeChromaCollection 的 MemoryService"""
    from agent.memory.memory_service import MemoryService
    db_path = tmp_path / "test_failure_chroma.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = _FakeChromaCollection()
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def collector(memory):
    """FailureCaseCollector（无 Chroma）"""
    from agent.failure_case_collector import FailureCaseCollector
    return FailureCaseCollector(memory)


@pytest.fixture
def collector_with_chroma(memory_with_chroma):
    """FailureCaseCollector（有 Chroma）"""
    from agent.failure_case_collector import FailureCaseCollector
    return FailureCaseCollector(memory_with_chroma)


@pytest.fixture
def client(memory_with_chroma):
    """FastAPI 测试客户端（注入临时 memory + collector + tracker）"""
    from fastapi.testclient import TestClient
    from api.routes import app, failure_collector, effect_tracker
    # 替换 collector 的 memory 和 db
    orig_collector_memory = failure_collector.memory
    orig_collector_db = failure_collector.db
    orig_tracker_memory = effect_tracker.memory
    orig_tracker_db = effect_tracker.db
    failure_collector.memory = memory_with_chroma
    failure_collector.db = memory_with_chroma.db
    effect_tracker.memory = memory_with_chroma
    effect_tracker.db = memory_with_chroma.db
    yield TestClient(app)
    failure_collector.memory = orig_collector_memory
    failure_collector.db = orig_collector_db
    effect_tracker.memory = orig_tracker_memory
    effect_tracker.db = orig_tracker_db


# ===== 辅助函数 =====


def _seed_low_confidence_case(memory, case_id, confidence, line_id="heat_treatment",
                              root_cause="温度偏低", solution="提高温度"):
    """注入低分案例到 Chroma"""
    memory._collection.add(
        ids=[case_id],
        documents=[f"案例 {case_id}"],
        metadatas=[{
            "confidence": confidence,
            "line_id": line_id,
            "root_cause": root_cause,
            "solution": solution,
            "defect_type": "crack",
        }],
    )


def _seed_negative_tracking(memory, tracking_id, case_id="case_neg",
                            improvement_pct=-15.0, line_id="heat_treatment"):
    """注入反效果跟踪记录到 effect_tracking 表"""
    memory.db.execute(
        """INSERT INTO effect_tracking
        (tracking_id, proposal_id, case_id, line_id, batch_id_before,
         batch_id_after, metric_before, metric_after, improvement_pct,
         status, days_offset, scheduled_at, tracked_at, note,
         attribution_done, attribution_result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'tracked', 7, ?, ?, ?, 0, NULL)""",
        (tracking_id, case_id, case_id, line_id, "b1", "b2",
         10.0, 10.0 * (1 + improvement_pct / 100), improvement_pct,
         datetime.now() - timedelta(days=2),
         datetime.now() - timedelta(days=1), "反效果测试", ),
    )
    memory.db.commit()


def _seed_rejected_feedback(memory, feedback_id="fb_rej_1",
                            proposal_id="prop_rej_1",
                            line_id="heat_treatment",
                            score=0.2):
    """注入被拒绝反馈"""
    memory.write_feedback(
        feedback_id=feedback_id,
        proposal_id=proposal_id,
        user_id="operator_01",
        action="rejected",
        score=score,
        comment="方案无效",
        line_id=line_id,
    )


# ===== 1. 收集逻辑测试 =====


class TestCollectLowConfidence:
    """collect_low_confidence_cases: 从 Chroma 收集低 confidence 案例"""

    def test_collects_cases_below_threshold(self, collector_with_chroma, memory_with_chroma):
        """confidence < min_confidence 的案例应被收集"""
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        _seed_low_confidence_case(memory_with_chroma, "case_low_2", confidence=0.25)
        _seed_low_confidence_case(memory_with_chroma, "case_high", confidence=0.8)

        result = collector_with_chroma.collect_low_confidence_cases(min_confidence=0.3)
        ids = [c["id"] for c in result]
        assert "case_low_1" in ids
        assert "case_low_2" in ids
        assert "case_high" not in ids

    def test_respects_limit(self, collector_with_chroma, memory_with_chroma):
        """limit 参数应限制返回数量"""
        for i in range(10):
            _seed_low_confidence_case(memory_with_chroma, f"case_low_{i}", confidence=0.1)
        result = collector_with_chroma.collect_low_confidence_cases(min_confidence=0.3, limit=3)
        assert len(result) <= 3

    def test_returns_empty_when_chroma_unavailable(self, collector):
        """Chroma 不可用时应返回空列表"""
        result = collector.collect_low_confidence_cases()
        assert result == []

    def test_line_id_filter(self, collector_with_chroma, memory_with_chroma):
        """line_id 过滤应只返回指定产线的案例"""
        _seed_low_confidence_case(memory_with_chroma, "case_ht", confidence=0.1,
                                  line_id="heat_treatment")
        _seed_low_confidence_case(memory_with_chroma, "case_weld", confidence=0.1,
                                  line_id="welding")
        result = collector_with_chroma.collect_low_confidence_cases(
            line_id="heat_treatment", min_confidence=0.3
        )
        ids = [c["id"] for c in result]
        assert "case_ht" in ids
        assert "case_weld" not in ids


class TestCollectNegativeEffect:
    """collect_negative_effect_trackings: 收集反效果跟踪"""

    def test_collects_negative_improvement(self, collector, memory):
        """improvement_pct < 0 的跟踪应被收集"""
        _seed_negative_tracking(memory, "trk_neg_1", improvement_pct=-15.0)
        _seed_negative_tracking(memory, "trk_pos_1", improvement_pct=20.0)
        result = collector.collect_negative_effect_trackings()
        ids = [r["tracking_id"] for r in result]
        assert "trk_neg_1" in ids
        assert "trk_pos_1" not in ids

    def test_line_id_filter(self, collector, memory):
        """line_id 过滤"""
        _seed_negative_tracking(memory, "trk_ht", improvement_pct=-10.0,
                                line_id="heat_treatment")
        _seed_negative_tracking(memory, "trk_weld", improvement_pct=-10.0,
                                line_id="welding")
        result = collector.collect_negative_effect_trackings(line_id="heat_treatment")
        ids = [r["tracking_id"] for r in result]
        assert "trk_ht" in ids
        assert "trk_weld" not in ids


class TestCollectRejectedFeedback:
    """collect_rejected_feedback: 收集被拒绝反馈"""

    def test_collects_rejected_only(self, collector, memory):
        """只收集 action=rejected 的反馈"""
        memory.write_feedback("fb_1", "p1", "u1", "rejected", 0.2, "无效", "heat_treatment")
        memory.write_feedback("fb_2", "p2", "u1", "adopted", 0.9, "有效", "heat_treatment")
        memory.write_feedback("fb_3", "p3", "u1", "partial", 0.5, "部分有效", "heat_treatment")
        result = collector.collect_rejected_feedback()
        ids = [r["feedback_id"] for r in result]
        assert "fb_1" in ids
        assert "fb_2" not in ids
        assert "fb_3" not in ids

    def test_line_id_filter(self, collector, memory):
        """line_id 过滤"""
        memory.write_feedback("fb_ht", "p1", "u1", "rejected", 0.2, "", "heat_treatment")
        memory.write_feedback("fb_weld", "p2", "u1", "rejected", 0.2, "", "welding")
        result = collector.collect_rejected_feedback(line_id="heat_treatment")
        ids = [r["feedback_id"] for r in result]
        assert "fb_ht" in ids
        assert "fb_weld" not in ids


class TestCollectAll:
    """collect_all: 综合收集三类失败案例 + 幂等性"""

    def test_collects_all_three_categories(self, collector_with_chroma, memory_with_chroma):
        """三类来源都应有收集"""
        # 低分案例
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        # 反效果跟踪
        _seed_negative_tracking(memory_with_chroma, "trk_neg_1", improvement_pct=-15.0)
        # 被拒绝反馈
        _seed_rejected_feedback(memory_with_chroma, "fb_rej_1", "prop_rej_1")

        result = collector_with_chroma.collect_all(min_confidence=0.3)
        assert result["low_confidence"] == 1
        assert result["negative_effect"] == 1
        assert result["rejected_feedback"] == 1
        assert result["total_collected"] == 3
        assert result["duplicates_skipped"] == 0

    def test_idempotent(self, collector_with_chroma, memory_with_chroma):
        """重复收集应跳过已存在的失败案例"""
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        # 第一次收集
        result1 = collector_with_chroma.collect_all(min_confidence=0.3)
        assert result1["total_collected"] == 1
        assert result1["duplicates_skipped"] == 0
        # 第二次收集（应跳过）
        result2 = collector_with_chroma.collect_all(min_confidence=0.3)
        assert result2["total_collected"] == 0
        assert result2["duplicates_skipped"] == 1

    def test_empty_when_no_failures(self, collector_with_chroma, memory_with_chroma):
        """无失败数据时应返回 0"""
        result = collector_with_chroma.collect_all()
        assert result["total_collected"] == 0
        assert result["low_confidence"] == 0
        assert result["negative_effect"] == 0
        assert result["rejected_feedback"] == 0

    def test_mixed_scenarios(self, collector_with_chroma, memory_with_chroma):
        """混合场景：多个低分 + 多个反效果 + 多个拒绝"""
        for i in range(3):
            _seed_low_confidence_case(memory_with_chroma, f"case_low_{i}", confidence=0.1)
        # 注意：_save_failure 幂等基于 case_id+category，需用不同 case_id
        _seed_negative_tracking(memory_with_chroma, "trk_neg_1", case_id="case_neg_1",
                                improvement_pct=-5.0)
        _seed_negative_tracking(memory_with_chroma, "trk_neg_2", case_id="case_neg_2",
                                improvement_pct=-20.0)
        memory_with_chroma.write_feedback("fb_r1", "p1", "u", "rejected", 0.1, "", "heat_treatment")
        memory_with_chroma.write_feedback("fb_r2", "p2", "u", "rejected", 0.2, "", "heat_treatment")

        result = collector_with_chroma.collect_all(min_confidence=0.3)
        assert result["low_confidence"] == 3
        assert result["negative_effect"] == 2
        assert result["rejected_feedback"] == 2
        assert result["total_collected"] == 7


# ===== 2. 查询/统计/状态更新测试 =====


class TestListFailures:
    """list_failures: 列表查询"""

    def test_list_all(self, collector, memory):
        """列出全部失败案例"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", confidence=0.1,
            failure_reason="test1",
        )
        collector._save_failure(
            case_id="c2", tracking_id=None, line_id="heat_treatment",
            category="negative_effect", improvement_pct=-10.0,
            failure_reason="test2",
        )
        result = collector.list_failures()
        assert len(result) == 2

    def test_filter_by_category(self, collector, memory):
        """按类别过滤"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        collector._save_failure(
            case_id="c2", tracking_id=None, line_id="heat_treatment",
            category="rejected_feedback", failure_reason="t2",
        )
        result = collector.list_failures(category="low_confidence")
        assert len(result) == 1
        assert result[0]["category"] == "low_confidence"

    def test_filter_by_line_id_list(self, collector, memory):
        """line_id 支持 list[str] 多产线 IN 查询"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        collector._save_failure(
            case_id="c2", tracking_id=None, line_id="welding",
            category="low_confidence", failure_reason="t2",
        )
        collector._save_failure(
            case_id="c3", tracking_id=None, line_id="casting",
            category="low_confidence", failure_reason="t3",
        )
        result = collector.list_failures(line_id=["heat_treatment", "welding"])
        assert len(result) == 2
        line_ids = {r["line_id"] for r in result}
        assert line_ids == {"heat_treatment", "welding"}

    def test_filter_by_status(self, collector, memory):
        """按状态过滤"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        # 更新一条状态
        failures = collector.list_failures()
        collector.update_failure_status(failures[0]["failure_id"], "resolved")

        result_open = collector.list_failures(status="open")
        result_resolved = collector.list_failures(status="resolved")
        assert len(result_open) == 0
        assert len(result_resolved) == 1


class TestGetFailure:
    """get_failure: 单条查询"""

    def test_get_existing(self, collector, memory):
        """查询存在的失败案例"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", confidence=0.1, failure_reason="test",
        )
        failures = collector.list_failures()
        failure_id = failures[0]["failure_id"]
        result = collector.get_failure(failure_id)
        assert result is not None
        assert result["case_id"] == "c1"
        assert result["category"] == "low_confidence"

    def test_get_nonexistent(self, collector, memory):
        """查询不存在的失败案例应返回 None"""
        result = collector.get_failure("fail_nonexistent")
        assert result is None


class TestGetFailureStats:
    """get_failure_stats: 统计"""

    def test_empty_stats(self, collector, memory):
        """无失败案例时统计为 0"""
        stats = collector.get_failure_stats()
        assert stats["total"] == 0
        assert stats["by_category"] == {}
        assert stats["by_status"] == {}

    def test_stats_by_category(self, collector, memory):
        """按类别统计"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        collector._save_failure(
            case_id="c2", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t2",
        )
        collector._save_failure(
            case_id="c3", tracking_id=None, line_id="heat_treatment",
            category="negative_effect", failure_reason="t3",
        )
        stats = collector.get_failure_stats()
        assert stats["total"] == 3
        assert stats["by_category"]["low_confidence"] == 2
        assert stats["by_category"]["negative_effect"] == 1

    def test_stats_by_status(self, collector, memory):
        """按状态统计（GROUP BY 只返回存在的状态，用 .get() 兜底）"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        failures = collector.list_failures()
        collector.update_failure_status(failures[0]["failure_id"], "analyzed")
        stats = collector.get_failure_stats()
        # 'open' 状态已无记录，GROUP BY 不会返回该键，用 .get() 兜底
        assert stats["by_status"].get("open", 0) == 0
        assert stats["by_status"]["analyzed"] == 1

    def test_stats_line_id_filter(self, collector, memory):
        """统计的 line_id 过滤"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        collector._save_failure(
            case_id="c2", tracking_id=None, line_id="welding",
            category="low_confidence", failure_reason="t2",
        )
        stats = collector.get_failure_stats(line_id="heat_treatment")
        assert stats["total"] == 1


class TestUpdateFailureStatus:
    """update_failure_status: 状态更新"""

    def test_update_to_analyzed(self, collector, memory):
        """更新状态为 analyzed"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="原始原因",
        )
        failures = collector.list_failures()
        success = collector.update_failure_status(
            failures[0]["failure_id"], "analyzed", note="已分析"
        )
        assert success is True
        updated = collector.get_failure(failures[0]["failure_id"])
        assert updated["status"] == "analyzed"
        assert "已分析" in updated["failure_reason"]

    def test_update_nonexistent_returns_false(self, collector, memory):
        """更新不存在的失败案例应返回 False"""
        success = collector.update_failure_status("fail_xxx", "resolved")
        assert success is False

    def test_note_appended_to_reason(self, collector, memory):
        """note 应追加到 failure_reason"""
        collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="原始",
        )
        failures = collector.list_failures()
        collector.update_failure_status(failures[0]["failure_id"], "resolved", note="已解决")
        updated = collector.get_failure(failures[0]["failure_id"])
        assert "原始" in updated["failure_reason"]
        assert "已解决" in updated["failure_reason"]
        assert "[resolved]" in updated["failure_reason"]


class TestSaveFailureIdempotent:
    """_save_failure 幂等性"""

    def test_same_case_id_category_skipped(self, collector, memory):
        """同 case_id + category 不重复保存"""
        r1 = collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        r2 = collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        assert r1 is True
        assert r2 is False
        # 数据库只有一条
        assert len(collector.list_failures()) == 1

    def test_same_case_id_different_category_saved(self, collector, memory):
        """同 case_id + 不同 category 都保存"""
        r1 = collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t1",
        )
        r2 = collector._save_failure(
            case_id="c1", tracking_id=None, line_id="heat_treatment",
            category="negative_effect", failure_reason="t2",
        )
        assert r1 is True
        assert r2 is True
        assert len(collector.list_failures()) == 2

    def test_empty_case_id_and_tracking_id_returns_false(self, collector, memory):
        """case_id 和 tracking_id 都为空时不保存"""
        r = collector._save_failure(
            case_id="", tracking_id=None, line_id="heat_treatment",
            category="low_confidence", failure_reason="t",
        )
        assert r is False


# ===== 3. API 端点测试 =====


class TestFailuresAPI:
    """/api/v1/failures/* 端点测试"""

    def test_collect_requires_admin(self, client):
        """非 admin 用户不能触发收集"""
        resp = client.post("/api/v1/failures/collect?user_id=operator_01")
        assert resp.status_code == 403

    def test_collect_success(self, client, memory_with_chroma):
        """admin 触发收集成功"""
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        resp = client.post(
            "/api/v1/failures/collect?user_id=admin&min_confidence=0.3"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["low_confidence"] == 1
        assert data["total_collected"] == 1

    def test_list_failures_admin(self, client, memory_with_chroma):
        """admin 列出全部失败案例"""
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        client.post("/api/v1/failures/collect?user_id=admin&min_confidence=0.3")
        resp = client.get("/api/v1/failures?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["records"]) == 1

    def test_list_failures_operator_filtered(self, client, memory_with_chroma):
        """operator 只能看到有权限的产线"""
        _seed_low_confidence_case(memory_with_chroma, "case_ht", confidence=0.1,
                                  line_id="heat_treatment")
        _seed_low_confidence_case(memory_with_chroma, "case_weld", confidence=0.1,
                                  line_id="welding")
        client.post("/api/v1/failures/collect?user_id=admin&min_confidence=0.3")
        # operator_01 只有 heat_treatment 权限（按 line_access.yaml 配置）
        resp = client.get("/api/v1/failures?user_id=operator_01")
        assert resp.status_code == 200
        data = resp.json()
        # 至少能看到 heat_treatment 的失败案例
        assert all(r["line_id"] == "heat_treatment" for r in data["records"])

    def test_failure_stats(self, client, memory_with_chroma):
        """统计端点"""
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        client.post("/api/v1/failures/collect?user_id=admin&min_confidence=0.3")
        resp = client.get("/api/v1/failures/stats?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert "by_category" in data
        assert "by_status" in data

    def test_get_failure_by_id(self, client, memory_with_chroma):
        """按 ID 查询单条失败案例"""
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        client.post("/api/v1/failures/collect?user_id=admin&min_confidence=0.3")
        # 先列出获取 failure_id
        list_resp = client.get("/api/v1/failures?user_id=admin")
        failure_id = list_resp.json()["records"][0]["failure_id"]
        # 查询
        resp = client.get(f"/api/v1/failures/{failure_id}")
        assert resp.status_code == 200
        assert resp.json()["failure_id"] == failure_id

    def test_get_failure_not_found(self, client):
        """查询不存在的 failure_id 应 404"""
        resp = client.get("/api/v1/failures/fail_nonexistent")
        assert resp.status_code == 404

    def test_update_failure_status(self, client, memory_with_chroma):
        """更新失败案例状态"""
        _seed_low_confidence_case(memory_with_chroma, "case_low_1", confidence=0.1)
        client.post("/api/v1/failures/collect?user_id=admin&min_confidence=0.3")
        list_resp = client.get("/api/v1/failures?user_id=admin")
        failure_id = list_resp.json()["records"][0]["failure_id"]
        # 更新（admin 有写权限）
        resp = client.patch(
            f"/api/v1/failures/{failure_id}?status=analyzed&note=已分析&user_id=admin"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "analyzed"

    def test_update_failure_not_found(self, client):
        """更新不存在的 failure_id 应 404"""
        resp = client.patch(
            "/api/v1/failures/fail_xxx?status=resolved&user_id=admin"
        )
        assert resp.status_code == 404
