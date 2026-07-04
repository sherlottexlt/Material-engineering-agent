"""
M5-1 调参效果跟踪测试

测试两层：
1. EffectTracker 单元测试（schedule/track/query/list/stats/run_due）
2. API 端点测试（/api/v1/effect/*）
"""
import pytest
from fastapi.testclient import TestClient

from agent.effect_tracker import EffectTracker, _default_quality_fetcher
from agent.memory.memory_service import MemoryService


# ===== Fixtures =====

@pytest.fixture
def memory(tmp_path):
    """临时 MemoryService（无 Chroma）"""
    db_path = tmp_path / "test_effect.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def tracker(memory):
    """EffectTracker 实例（用默认 quality_fetcher）"""
    return EffectTracker(memory)


@pytest.fixture
def client(memory):
    """FastAPI 测试客户端（注入临时 memory + tracker）"""
    from api.routes import app, effect_tracker
    # 替换全局 effect_tracker 的 db 连接为临时 memory 的
    original_db = effect_tracker.db
    effect_tracker.db = memory.db
    effect_tracker.memory = memory
    yield TestClient(app)
    effect_tracker.db = original_db
    effect_tracker.memory = memory


# ===== EffectTracker 单元测试 =====

class TestEffectTrackerSchedule:
    """M5-1: schedule_tracking 调度跟踪"""

    def test_schedule_tracking_creates_pending_record(self, tracker):
        """调度跟踪应创建 pending 状态记录"""
        tracking_id = tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            line_id="heat_treatment",
            metric_before=0.15,
            days_offset=7,
        )
        assert tracking_id.startswith("trk_")

        rec = tracker.get_tracking(tracking_id)
        assert rec is not None
        assert rec["proposal_id"] == "P001"
        assert rec["case_id"] == "case_001"
        assert rec["batch_id_before"] == "B001"
        assert rec["line_id"] == "heat_treatment"
        assert rec["metric_before"] == 0.15
        assert rec["status"] == "pending"
        assert rec["metric_after"] is None
        assert rec["improvement_pct"] is None
        assert rec["tracked_at"] is None
        assert rec["days_offset"] == 7

    def test_schedule_tracking_default_line_id(self, tracker):
        """未指定 line_id 时默认 heat_treatment"""
        tracking_id = tracker.schedule_tracking(
            proposal_id="P002",
            case_id="case_002",
            batch_id_before="B002",
        )
        rec = tracker.get_tracking(tracking_id)
        assert rec["line_id"] == "heat_treatment"

    def test_schedule_tracking_returns_unique_id(self, tracker):
        """每次调度返回唯一 tracking_id"""
        id1 = tracker.schedule_tracking("P1", "C1", "B1")
        id2 = tracker.schedule_tracking("P2", "C2", "B2")
        assert id1 != id2


class TestEffectTrackerTrack:
    """M5-1: track_effect 执行跟踪"""

    def test_track_effect_computes_improvement(self, tracker):
        """执行跟踪应计算改善百分比"""
        tracking_id = tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            metric_before=0.20,
            days_offset=0,  # 立即可跟踪
        )
        result = tracker.track_effect(tracking_id, batch_id_after="B002")

        assert result is not None
        assert result["status"] == "tracked"
        assert result["batch_id_after"] == "B002"
        assert result["metric_after"] is not None
        # 改善百分比 = (0.20 - after) / 0.20 * 100
        expected = (0.20 - result["metric_after"]) / 0.20 * 100
        assert abs(result["improvement_pct"] - round(expected, 2)) < 0.01

    def test_track_effect_uses_default_batch_id_after(self, tracker):
        """未指定 batch_id_after 时用 batch_id_before + '_after' 衍生"""
        tracking_id = tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            metric_before=0.15,
            days_offset=0,
        )
        result = tracker.track_effect(tracking_id)
        assert result["batch_id_after"] == "B001_after"

    def test_track_effect_idempotent(self, tracker):
        """重复执行已跟踪的记录返回已有结果"""
        tracking_id = tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            metric_before=0.15,
            days_offset=0,
        )
        first = tracker.track_effect(tracking_id)
        second = tracker.track_effect(tracking_id)

        assert first is not None
        assert second is not None
        assert first["tracking_id"] == second["tracking_id"]
        assert first["metric_after"] == second["metric_after"]

    def test_track_effect_nonexistent_returns_none(self, tracker):
        """跟踪不存在的记录返回 None"""
        result = tracker.track_effect("trk_nonexistent")
        assert result is None

    def test_track_effect_with_custom_fetcher(self, memory):
        """自定义 quality_fetcher 应被调用"""
        calls = []

        def custom_fetcher(batch_id, line_id):
            calls.append((batch_id, line_id))
            return 0.05  # 固定低缺陷率

        tracker = EffectTracker(memory, quality_fetcher=custom_fetcher)
        tracking_id = tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            metric_before=0.20,
            days_offset=0,
        )
        result = tracker.track_effect(tracking_id, batch_id_after="B002")

        assert ("B002", "heat_treatment") in calls
        assert result["metric_after"] == 0.05
        # 改善 = (0.20 - 0.05) / 0.20 * 100 = 75.0
        assert result["improvement_pct"] == 75.0

    def test_track_effect_zero_metric_before(self, tracker):
        """metric_before=0 时 improvement_pct 为 None（避免除零）"""
        tracking_id = tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            metric_before=0.0,
            days_offset=0,
        )
        result = tracker.track_effect(tracking_id)
        assert result["improvement_pct"] is None

    def test_track_effect_none_metric_before(self, tracker):
        """metric_before=None 时 improvement_pct 为 None"""
        tracking_id = tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            metric_before=None,
            days_offset=0,
        )
        result = tracker.track_effect(tracking_id)
        assert result["improvement_pct"] is None
        assert result["metric_before"] is None


class TestEffectTrackerQuery:
    """M5-1: 查询方法"""

    def test_get_tracking_nonexistent_returns_none(self, tracker):
        assert tracker.get_tracking("trk_nope") is None

    def test_list_trackings_by_line(self, tracker):
        """按产线过滤"""
        tracker.schedule_tracking("P1", "C1", "B1", line_id="heat_treatment", days_offset=0)
        tracker.schedule_tracking("P2", "C2", "B2", line_id="welding", days_offset=0)

        ht = tracker.list_trackings(line_id="heat_treatment", days=365)
        wd = tracker.list_trackings(line_id="welding", days=365)

        assert len(ht) == 1
        assert ht[0]["line_id"] == "heat_treatment"
        assert len(wd) == 1
        assert wd[0]["line_id"] == "welding"

    def test_list_trackings_by_line_list(self, tracker):
        """多产线 IN 查询"""
        tracker.schedule_tracking("P1", "C1", "B1", line_id="heat_treatment", days_offset=0)
        tracker.schedule_tracking("P2", "C2", "B2", line_id="welding", days_offset=0)

        both = tracker.list_trackings(
            line_id=["heat_treatment", "welding"], days=365
        )
        assert len(both) == 2

    def test_list_trackings_by_status(self, tracker):
        """按状态过滤"""
        tid1 = tracker.schedule_tracking("P1", "C1", "B1", metric_before=0.1, days_offset=0)
        tid2 = tracker.schedule_tracking("P2", "C2", "B2", metric_before=0.1, days_offset=0)
        tracker.track_effect(tid1)  # tid1 → tracked

        pending = tracker.list_trackings(status="pending", days=365)
        tracked = tracker.list_trackings(status="tracked", days=365)

        assert len(pending) == 1
        assert pending[0]["tracking_id"] == tid2
        assert len(tracked) == 1
        assert tracked[0]["tracking_id"] == tid1

    def test_list_pending(self, tracker):
        """list_pending 返回已到期的 pending 记录"""
        tid1 = tracker.schedule_tracking("P1", "C1", "B1", days_offset=0)  # 立即到期
        tid2 = tracker.schedule_tracking("P2", "C2", "B2", days_offset=30)  # 30 天后

        pending = tracker.list_pending()
        assert len(pending) == 1
        assert pending[0]["tracking_id"] == tid1


class TestEffectTrackerStats:
    """M5-1: 统计方法"""

    def test_get_effect_stats_empty(self, tracker):
        """无记录时返回零值统计"""
        stats = tracker.get_effect_stats()
        assert stats["total"] == 0
        assert stats["tracked"] == 0
        assert stats["pending"] == 0
        assert stats["avg_improvement"] == 0.0

    def test_get_effect_stats_with_data(self, tracker):
        """有数据时正确统计"""
        # 3 条跟踪：2 条已跟踪（1 改善 1 恶化），1 条 pending
        tid1 = tracker.schedule_tracking("P1", "C1", "B1", metric_before=0.20, days_offset=0)
        tid2 = tracker.schedule_tracking("P2", "C2", "B2", metric_before=0.10, days_offset=0)
        tid3 = tracker.schedule_tracking("P3", "C3", "B3", metric_before=0.15, days_offset=0)

        # 用自定义 fetcher 控制结果
        tracker.quality_fetcher = lambda bid, lid: 0.05 if "B1" in bid else 0.20
        tracker.track_effect(tid1)  # 0.20→0.05 改善 75%
        tracker.track_effect(tid2)  # 0.10→0.20 恶化 -100%

        stats = tracker.get_effect_stats(days=365)
        assert stats["total"] == 3
        assert stats["tracked"] == 2
        assert stats["pending"] == 1
        assert stats["positive_count"] == 1
        assert stats["negative_count"] == 1
        # 平均改善 = (75 + (-100)) / 2 = -12.5
        assert stats["avg_improvement"] == -12.5

    def test_get_effect_stats_by_line(self, tracker):
        """按产线过滤统计"""
        tracker.schedule_tracking("P1", "C1", "B1", line_id="heat_treatment", days_offset=0)
        tracker.schedule_tracking("P2", "C2", "B2", line_id="welding", days_offset=0)

        ht_stats = tracker.get_effect_stats(line_id="heat_treatment", days=365)
        wd_stats = tracker.get_effect_stats(line_id="welding", days=365)

        assert ht_stats["total"] == 1
        assert wd_stats["total"] == 1


class TestEffectTrackerRunDue:
    """M5-1: run_due_trackings 批量执行"""

    def test_run_due_trackings(self, tracker):
        """批量执行到期记录"""
        tid1 = tracker.schedule_tracking("P1", "C1", "B1", metric_before=0.1, days_offset=0)
        tid2 = tracker.schedule_tracking("P2", "C2", "B2", metric_before=0.1, days_offset=0)
        tid3 = tracker.schedule_tracking("P3", "C3", "B3", metric_before=0.1, days_offset=30)

        result = tracker.run_due_trackings()
        assert result["executed"] == 2
        assert result["succeeded"] == 2
        assert result["failed"] == 0

        # tid1/tid2 已跟踪，tid3 仍 pending
        assert tracker.get_tracking(tid1)["status"] == "tracked"
        assert tracker.get_tracking(tid2)["status"] == "tracked"
        assert tracker.get_tracking(tid3)["status"] == "pending"

    def test_run_due_trackings_by_line(self, tracker):
        """按产线过滤批量执行"""
        tracker.schedule_tracking("P1", "C1", "B1", line_id="heat_treatment", days_offset=0)
        tracker.schedule_tracking("P2", "C2", "B2", line_id="welding", days_offset=0)

        result = tracker.run_due_trackings(line_id="heat_treatment")
        assert result["executed"] == 1


class TestDefaultQualityFetcher:
    """M5-1: 默认 quality_fetcher 稳定性"""

    def test_default_fetcher_stable(self):
        """同一 batch_id + line_id 多次查询结果一致"""
        v1 = _default_quality_fetcher("B001", "heat_treatment")
        v2 = _default_quality_fetcher("B001", "heat_treatment")
        assert v1 == v2

    def test_default_fetcher_range(self):
        """默认 fetcher 返回值在 0.02-0.30 范围内"""
        for i in range(100):
            val = _default_quality_fetcher(f"B{i:03d}", "heat_treatment")
            assert 0.02 <= val <= 0.30

    def test_default_fetcher_different_lines(self):
        """不同产线同批次结果不同"""
        v1 = _default_quality_fetcher("B001", "heat_treatment")
        v2 = _default_quality_fetcher("B001", "welding")
        assert v1 != v2


# ===== API 端点测试 =====

class TestEffectAPI:
    """M5-1: /api/v1/effect/* 端点"""

    def test_schedule_effect_tracking(self, client):
        """POST /api/v1/effect/track 创建跟踪记录"""
        resp = client.post("/api/v1/effect/track", json={
            "proposal_id": "P001",
            "case_id": "case_001",
            "batch_id_before": "B001",
            "line_id": "heat_treatment",
            "metric_before": 0.15,
            "user_id": "operator_01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "pending"
        assert data["tracking_id"].startswith("trk_")

    def test_schedule_effect_other_line_forbidden(self, client):
        """operator_01 无权给 welding 创建跟踪 → 403"""
        resp = client.post("/api/v1/effect/track", json={
            "proposal_id": "P001",
            "case_id": "case_001",
            "batch_id_before": "B001",
            "line_id": "welding",
            "user_id": "operator_01",
        })
        assert resp.status_code == 403

    def test_get_effect_tracking(self, client):
        """GET /api/v1/effect/{tracking_id}"""
        # 先创建
        create = client.post("/api/v1/effect/track", json={
            "proposal_id": "P001",
            "case_id": "case_001",
            "batch_id_before": "B001",
            "user_id": "operator_01",
        })
        tracking_id = create.json()["tracking_id"]

        resp = client.get(f"/api/v1/effect/{tracking_id}")
        assert resp.status_code == 200
        assert resp.json()["tracking_id"] == tracking_id

    def test_get_effect_tracking_not_found(self, client):
        """GET 不存在的 tracking_id → 404"""
        resp = client.get("/api/v1/effect/trk_nonexistent")
        assert resp.status_code == 404

    def test_list_effect_trackings(self, client):
        """GET /api/v1/effect 列表"""
        client.post("/api/v1/effect/track", json={
            "proposal_id": "P1", "case_id": "C1", "batch_id_before": "B1",
            "user_id": "operator_01",
        })
        resp = client.get("/api/v1/effect", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_evaluate_effect(self, client):
        """POST /api/v1/effect/{tracking_id}/evaluate 触发跟踪"""
        create = client.post("/api/v1/effect/track", json={
            "proposal_id": "P001",
            "case_id": "case_001",
            "batch_id_before": "B001",
            "metric_before": 0.20,
            "user_id": "operator_01",
        })
        tracking_id = create.json()["tracking_id"]

        resp = client.post(
            f"/api/v1/effect/{tracking_id}/evaluate",
            params={"batch_id_after": "B002", "user_id": "operator_01"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "tracked"
        assert data["batch_id_after"] == "B002"
        assert data["metric_after"] is not None

    def test_effect_stats(self, client):
        """GET /api/v1/effect/stats 统计"""
        client.post("/api/v1/effect/track", json={
            "proposal_id": "P1", "case_id": "C1", "batch_id_before": "B1",
            "metric_before": 0.15, "user_id": "operator_01",
        })
        resp = client.get("/api/v1/effect/stats", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_run_due_requires_admin(self, client):
        """POST /api/v1/effect/run-due 仅 admin 可调用"""
        resp = client.post(
            "/api/v1/effect/run-due",
            params={"user_id": "operator_01"},
        )
        assert resp.status_code == 403

    def test_run_due_admin_ok(self, client):
        """admin 可批量执行"""
        client.post("/api/v1/effect/track", json={
            "proposal_id": "P1", "case_id": "C1", "batch_id_before": "B1",
            "metric_before": 0.1, "user_id": "admin", "line_id": "heat_treatment",
            "days_offset": 0,
        })
        resp = client.post(
            "/api/v1/effect/run-due",
            params={"user_id": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["executed"] >= 1
