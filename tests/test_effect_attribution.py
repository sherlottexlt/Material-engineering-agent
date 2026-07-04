"""
M5-2 效果归因测试

测试三层：
1. MemoryService.update_confidence_from_effect + _improvement_to_effect_score 映射
2. EffectTracker.attribute_effect（幂等、自动 track、案例不存在）+ run_due_trackings 自动归因
3. Evaluator.collect_feedback（反馈持久化 + 聚合更新 confidence）
4. API 端点 /api/v1/effect/{tracking_id}/attribute
"""
import sys
import types

import pytest

# mock langchain_openai.ChatOpenAI（sandbox 中模块存在但缺 ChatOpenAI 属性）
if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = types.ModuleType("langchain_openai")
_mock_lc = sys.modules["langchain_openai"]
if not hasattr(_mock_lc, "ChatOpenAI"):
    _mock_lc.ChatOpenAI = type("ChatOpenAI", (), {"__init__": lambda self, **kw: None})


# ===== 轻量级 Chroma collection 模拟（测试用，无需安装 chromadb）=====


class _FakeChromaCollection:
    """模拟 Chroma collection 的内存实现，覆盖 get/update/add/delete 接口。"""

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
    db_path = tmp_path / "test_attribution.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def memory_with_chroma(tmp_path):
    """带 FakeChromaCollection 的 MemoryService（支持 confidence 更新）"""
    from agent.memory.memory_service import MemoryService
    db_path = tmp_path / "test_attribution_chroma.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = _FakeChromaCollection()
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def tracker(memory):
    """EffectTracker 实例（无 Chroma）"""
    from agent.effect_tracker import EffectTracker
    return EffectTracker(memory)


@pytest.fixture
def tracker_with_chroma(memory_with_chroma):
    """EffectTracker 实例（有 Chroma，支持归因）"""
    from agent.effect_tracker import EffectTracker
    return EffectTracker(memory_with_chroma)


@pytest.fixture
def client(memory_with_chroma):
    """FastAPI 测试客户端（注入临时 memory + tracker）"""
    from fastapi.testclient import TestClient
    from api.routes import app, effect_tracker
    original_db = effect_tracker.db
    original_memory = effect_tracker.memory
    effect_tracker.db = memory_with_chroma.db
    effect_tracker.memory = memory_with_chroma
    yield TestClient(app)
    effect_tracker.db = original_db
    effect_tracker.memory = original_memory


# ===== 1. _improvement_to_effect_score 映射测试 =====


class TestImprovementToEffectScore:
    """改善百分比 → 效果分映射"""

    def test_significant_improvement(self):
        """improvement >= 30 → 1.0"""
        from agent.memory.memory_service import MemoryService
        assert MemoryService._improvement_to_effect_score(30) == 1.0
        assert MemoryService._improvement_to_effect_score(50) == 1.0
        assert MemoryService._improvement_to_effect_score(100) == 1.0

    def test_moderate_improvement(self):
        """10 <= improvement < 30 → 0.7-0.95 线性"""
        from agent.memory.memory_service import MemoryService
        assert MemoryService._improvement_to_effect_score(10) == 0.7
        assert MemoryService._improvement_to_effect_score(30) == 1.0  # 边界
        # 中点 20 → 0.7 + 10/20 * 0.25 = 0.825
        assert MemoryService._improvement_to_effect_score(20) == 0.825

    def test_small_improvement(self):
        """0 <= improvement < 10 → 0.5-0.7 线性"""
        from agent.memory.memory_service import MemoryService
        assert MemoryService._improvement_to_effect_score(0) == 0.5
        assert MemoryService._improvement_to_effect_score(10) == 0.7  # 边界
        assert MemoryService._improvement_to_effect_score(5) == 0.6

    def test_small_negative(self):
        """-10 < improvement < 0 → 0.3-0.5 线性"""
        from agent.memory.memory_service import MemoryService
        assert MemoryService._improvement_to_effect_score(0) == 0.5
        assert MemoryService._improvement_to_effect_score(-5) == 0.4
        # -10 的边界走 <= -10 分支返回 0.1，但 -10 > -10 为 False 所以也走这分支
        assert MemoryService._improvement_to_effect_score(-9) == round(0.5 + (-9) / 10 * 0.2, 3)

    def test_significant_negative(self):
        """improvement <= -10 → 0.1"""
        from agent.memory.memory_service import MemoryService
        assert MemoryService._improvement_to_effect_score(-10) == 0.1
        assert MemoryService._improvement_to_effect_score(-20) == 0.1
        assert MemoryService._improvement_to_effect_score(-50) == 0.1

    def test_monotonic_increasing(self):
        """映射函数单调递增（改善越高，效果分越高）"""
        from agent.memory.memory_service import MemoryService
        values = [-20, -10, -5, 0, 5, 10, 20, 30, 50]
        scores = [MemoryService._improvement_to_effect_score(v) for v in values]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"{values[i]}->{scores[i]} > {values[i+1]}->{scores[i+1]}"


# ===== 2. update_confidence_from_effect 测试 =====


class TestUpdateConfidenceFromEffect:
    """MemoryService.update_confidence_from_effect"""

    def test_update_with_significant_improvement(self, memory_with_chroma):
        """显著改善应提升 confidence"""
        memory_with_chroma._collection.add(
            ids=["case_001"],
            documents=["案例1"],
            metadatas=[{"confidence": 0.5, "defect_type": "crack"}],
        )
        result = memory_with_chroma.update_confidence_from_effect("case_001", 25.0)
        assert result["updated"] is True
        assert result["old_confidence"] == 0.5
        # 25% 改善：effect_score = 0.7 + 15/20*0.25 = 0.8875
        # new = 0.5*0.6 + 0.8875*0.4 + 0.05(奖励) = 0.3 + 0.355 + 0.05 = 0.705
        assert result["new_confidence"] > 0.5
        assert result["effect_score"] == round(0.7 + 15 / 20 * 0.25, 3)

    def test_update_with_negative_improvement(self, memory_with_chroma):
        """反效应降低 confidence"""
        memory_with_chroma._collection.add(
            ids=["case_002"],
            documents=["案例2"],
            metadatas=[{"confidence": 0.7}],
        )
        result = memory_with_chroma.update_confidence_from_effect("case_002", -15.0)
        assert result["updated"] is True
        assert result["old_confidence"] == 0.7
        assert result["new_confidence"] < 0.7
        assert result["effect_score"] == 0.1  # <= -10

    def test_update_with_zero_improvement(self, memory_with_chroma):
        """无改善时 confidence 基本不变（小幅向 0.5 靠拢）"""
        memory_with_chroma._collection.add(
            ids=["case_003"],
            documents=["案例3"],
            metadatas=[{"confidence": 0.5}],
        )
        result = memory_with_chroma.update_confidence_from_effect("case_003", 0.0)
        assert result["updated"] is True
        # 0% 改善：effect_score = 0.5, new = 0.5*0.6 + 0.5*0.4 = 0.5
        assert result["new_confidence"] == 0.5

    def test_update_nonexistent_case(self, memory_with_chroma):
        """案例不存在时返回 updated=False"""
        result = memory_with_chroma.update_confidence_from_effect("nonexistent", 20.0)
        assert result["updated"] is False
        assert result["old_confidence"] is None

    def test_update_without_chroma(self, memory):
        """无 Chroma 时降级返回 updated=False"""
        result = memory.update_confidence_from_effect("case_x", 15.0)
        assert result["updated"] is False
        assert result["old_confidence"] is None

    def test_confidence_bounded(self, memory_with_chroma):
        """confidence 不超出 [0.05, 1.0]"""
        memory_with_chroma._collection.add(
            ids=["case_high"],
            documents=["高置信度"],
            metadatas=[{"confidence": 0.95}],
        )
        # 大幅改善，confidence 不超过 1.0
        result = memory_with_chroma.update_confidence_from_effect("case_high", 50.0)
        assert result["new_confidence"] <= 1.0

        memory_with_chroma._collection.add(
            ids=["case_low"],
            documents=["低置信度"],
            metadatas=[{"confidence": 0.1}],
        )
        # 大幅反效果，confidence 不低于 0.05
        result = memory_with_chroma.update_confidence_from_effect("case_low", -50.0)
        assert result["new_confidence"] >= 0.05

    def test_metadata_records_attribution(self, memory_with_chroma):
        """归因后 metadata 记录 last_effect_improvement 和 last_attributed_at"""
        memory_with_chroma._collection.add(
            ids=["case_meta"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        memory_with_chroma.update_confidence_from_effect("case_meta", 15.0)
        updated = memory_with_chroma._collection.get(ids=["case_meta"])
        meta = updated["metadatas"][0]
        assert "last_effect_improvement" in meta
        assert meta["last_effect_improvement"] == 15.0
        assert "last_attributed_at" in meta


# ===== 3. EffectTracker.attribute_effect 测试 =====


class TestAttributeEffect:
    """EffectTracker.attribute_effect"""

    def test_attribute_after_track(self, tracker_with_chroma, memory_with_chroma):
        """跟踪完成后归因应更新案例 confidence"""
        memory_with_chroma._collection.add(
            ids=["case_attr_1"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        tracking_id = tracker_with_chroma.schedule_tracking(
            proposal_id="P001", case_id="case_attr_1",
            batch_id_before="B001", metric_before=0.20, days_offset=0,
        )
        tracker_with_chroma.track_effect(tracking_id)

        result = tracker_with_chroma.attribute_effect(tracking_id)
        assert result is not None
        assert result["attribution_done"] is True
        assert result["old_confidence"] == 0.5
        assert result["new_confidence"] != 0.5
        assert "attributed_at" in result

    def test_attribute_idempotent(self, tracker_with_chroma, memory_with_chroma):
        """归因幂等：重复调用返回上次结果"""
        memory_with_chroma._collection.add(
            ids=["case_attr_2"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        tracking_id = tracker_with_chroma.schedule_tracking(
            proposal_id="P002", case_id="case_attr_2",
            batch_id_before="B002", metric_before=0.20, days_offset=0,
        )
        tracker_with_chroma.track_effect(tracking_id)

        first = tracker_with_chroma.attribute_effect(tracking_id)
        second = tracker_with_chroma.attribute_effect(tracking_id)
        assert first is not None and second is not None
        assert first["new_confidence"] == second["new_confidence"]
        assert first["attribution_done"] is True

    def test_attribute_auto_tracks_pending(self, tracker_with_chroma, memory_with_chroma):
        """归因时若记录仍 pending，自动先 track"""
        memory_with_chroma._collection.add(
            ids=["case_attr_3"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        tracking_id = tracker_with_chroma.schedule_tracking(
            proposal_id="P003", case_id="case_attr_3",
            batch_id_before="B003", metric_before=0.20, days_offset=0,
        )
        # 不手动 track，直接 attribute
        result = tracker_with_chroma.attribute_effect(tracking_id)
        assert result is not None
        assert result["attribution_done"] is True

    def test_attribute_nonexistent_tracking(self, tracker_with_chroma):
        """不存在的 tracking_id 返回 None"""
        result = tracker_with_chroma.attribute_effect("trk_nonexistent")
        assert result is None

    def test_attribute_records_in_db(self, tracker_with_chroma, memory_with_chroma):
        """归因结果持久化到 effect_tracking 表"""
        memory_with_chroma._collection.add(
            ids=["case_attr_4"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        tracking_id = tracker_with_chroma.schedule_tracking(
            proposal_id="P004", case_id="case_attr_4",
            batch_id_before="B004", metric_before=0.20, days_offset=0,
        )
        tracker_with_chroma.track_effect(tracking_id)
        tracker_with_chroma.attribute_effect(tracking_id)

        rec = tracker_with_chroma.get_tracking(tracking_id)
        assert rec["attribution_done"] == 1
        assert rec["attribution_result"] is not None
        import json
        stored = json.loads(rec["attribution_result"])
        assert stored["case_id"] == "case_attr_4"
        assert stored["attribution_done"] is True


# ===== 4. run_due_trackings 自动归因测试 =====


class TestRunDueAttribution:
    """run_due_trackings 跟踪 + 自动归因"""

    def test_run_due_attributes_tracked(self, tracker_with_chroma, memory_with_chroma):
        """批量跟踪后自动归因，返回 attributed 计数"""
        memory_with_chroma._collection.add(
            ids=["case_due_1", "case_due_2"],
            documents=["案例1", "案例2"],
            metadatas=[{"confidence": 0.5}, {"confidence": 0.6}],
        )
        for i, case_id in enumerate(["case_due_1", "case_due_2"]):
            tracker_with_chroma.schedule_tracking(
                proposal_id=f"PD{i}", case_id=case_id,
                batch_id_before=f"BD{i}", metric_before=0.20, days_offset=0,
            )
        result = tracker_with_chroma.run_due_trackings()
        assert result["executed"] == 2
        assert result["succeeded"] == 2
        assert result["attributed"] == 2

    def test_run_due_attributed_field_in_result(self, tracker_with_chroma):
        """run_due_trackings 返回值包含 attributed 字段"""
        result = tracker_with_chroma.run_due_trackings()
        assert "attributed" in result
        assert result["executed"] == 0
        assert result["attributed"] == 0


# ===== 5. Evaluator.collect_feedback 测试 =====


class TestEvaluatorCollectFeedback:
    """Evaluator.collect_feedback"""

    def test_collect_feedback_without_memory(self):
        """未注入 memory 时返回 saved=False"""
        from agent.evaluator import Evaluator
        from models.entities import UserFeedback
        ev = Evaluator()
        fb = UserFeedback(
            feedback_id="fb_1", proposal_id="P001", user_id="u1",
            action="adopted", score=0.8,
        )
        result = ev.collect_feedback(fb)
        assert result["saved"] is False
        assert "error" in result

    def test_collect_feedback_persists_and_updates(self, memory_with_chroma):
        """注入 memory 后：持久化反馈 + 聚合更新 confidence"""
        from agent.evaluator import Evaluator
        from models.entities import UserFeedback
        # 先添加案例（proposal_id 作为 case_id）
        memory_with_chroma._collection.add(
            ids=["P001"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        ev = Evaluator(memory=memory_with_chroma)
        fb = UserFeedback(
            feedback_id="fb_1", proposal_id="P001", user_id="u1",
            action="adopted", score=0.9,
        )
        result = ev.collect_feedback(fb)
        assert result["saved"] is True
        assert result["feedback_id"] == "fb_1"
        update = result["confidence_update"]
        assert update["feedback_count"] >= 1
        assert update["updated"] is True
        # 高分反馈应提升 confidence
        assert update["new_confidence"] > 0.5

    def test_collect_feedback_multiple_aggregates(self, memory_with_chroma):
        """多条反馈聚合更新（反馈分占 50%，旧 confidence 占 50%）"""
        from agent.evaluator import Evaluator
        from models.entities import UserFeedback
        memory_with_chroma._collection.add(
            ids=["P002"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        ev = Evaluator(memory=memory_with_chroma)
        # 两条反馈
        for i, score in enumerate([0.8, 0.9]):
            fb = UserFeedback(
                feedback_id=f"fb_{i}", proposal_id="P002", user_id=f"u{i}",
                action="adopted", score=score,
            )
            result = ev.collect_feedback(fb)
            assert result["saved"] is True

        update = result["confidence_update"]
        assert update["feedback_count"] == 2


# ===== 6. API 端点测试 =====


class TestAttributionAPI:
    """/api/v1/effect/{tracking_id}/attribute"""

    def test_attribute_api_success(self, client, tracker_with_chroma, memory_with_chroma):
        """API 成功归因"""
        memory_with_chroma._collection.add(
            ids=["case_api_1"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        tracking_id = tracker_with_chroma.schedule_tracking(
            proposal_id="PA1", case_id="case_api_1",
            batch_id_before="BA1", metric_before=0.20, days_offset=0,
        )
        tracker_with_chroma.track_effect(tracking_id)

        resp = client.post(f"/api/v1/effect/{tracking_id}/attribute?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["attribution_done"] is True
        assert "new_confidence" in data

    def test_attribute_api_not_found(self, client):
        """不存在的 tracking_id 返回 404"""
        resp = client.post("/api/v1/effect/trk_nonexistent/attribute?user_id=admin")
        assert resp.status_code == 404

    def test_attribute_api_idempotent(self, client, tracker_with_chroma, memory_with_chroma):
        """API 重复归因幂等"""
        memory_with_chroma._collection.add(
            ids=["case_api_2"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        tracking_id = tracker_with_chroma.schedule_tracking(
            proposal_id="PA2", case_id="case_api_2",
            batch_id_before="BA2", metric_before=0.20, days_offset=0,
        )
        tracker_with_chroma.track_effect(tracking_id)

        r1 = client.post(f"/api/v1/effect/{tracking_id}/attribute?user_id=admin")
        r2 = client.post(f"/api/v1/effect/{tracking_id}/attribute?user_id=admin")
        assert r1.status_code == 200 and r2.status_code == 200
        d1, d2 = r1.json(), r2.json()
        assert d1["new_confidence"] == d2["new_confidence"]

    def test_attribute_api_permission_check(self, client, tracker_with_chroma, memory_with_chroma):
        """非授权用户访问其他产线的 tracking 应 403"""
        memory_with_chroma._collection.add(
            ids=["case_api_3"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        tracking_id = tracker_with_chroma.schedule_tracking(
            proposal_id="PA3", case_id="case_api_3",
            batch_id_before="BA3", line_id="welding",
            metric_before=0.20, days_offset=0,
        )
        tracker_with_chroma.track_effect(tracking_id)
        # operator_01 无 welding 产线权限（默认只有 heat_treatment）
        resp = client.post(f"/api/v1/effect/{tracking_id}/attribute?user_id=operator_01")
        assert resp.status_code == 403
