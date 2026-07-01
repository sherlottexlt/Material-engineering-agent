"""
Memory Service 单元测试
对应 TDD 第 11 节

测试三层记忆：
- 短期记忆 CRUD
- 长期记忆写入与检索
- 置信度更新
- 遗忘机制
- memory_writer 缺陷类型抽取
- search_cases 语义检索接入
"""
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pytest

from agent.memory.memory_service import MemoryService
from agent.nodes.memory_writer import _extract_defect_type, _format_solution
from agent.tools import search_cases, _set_memory_service
from models.entities import CaseRecord, BatchParams, DefectType, ProcessType
from models.state import AgentState


# ===== 轻量级 Chroma collection 模拟（测试用，无需安装 chromadb）=====


class _FakeChromaCollection:
    """模拟 Chroma collection 的内存实现，覆盖 add/query/get/update 接口。

    用关键词重叠度近似语义相关度，足够测试 MemoryService 的写入/检索/更新逻辑。
    """

    def __init__(self):
        self._store: dict[str, dict] = {}  # id -> {document, metadata}

    def add(self, ids, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._store[doc_id] = {
                "document": documents[i] if i < len(documents) else "",
                "metadata": dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
            }

    def query(self, query_texts, n_results):
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        if not self._store or not query_texts:
            return results
        query_text = query_texts[0] or ""
        scored = []
        for doc_id, item in self._store.items():
            doc = item["document"]
            q_tokens = [t for t in query_text.split() if t]
            overlap = sum(1 for t in q_tokens if t in doc)
            # 距离 = 1 - 归一化重叠度
            distance = 1.0 - (overlap / max(len(q_tokens), 1)) if q_tokens else 1.0
            scored.append((distance, doc_id, item))
        scored.sort(key=lambda x: x[0])
        for distance, doc_id, item in scored[:n_results]:
            results["documents"][0].append(item["document"])
            results["metadatas"][0].append(dict(item["metadata"]))
            results["distances"][0].append(distance)
            results["ids"][0].append(doc_id)
        return results

    def get(self, ids):
        metadatas, documents = [], []
        for doc_id in ids:
            if doc_id in self._store:
                metadatas.append(dict(self._store[doc_id]["metadata"]))
                documents.append(self._store[doc_id]["document"])
            else:
                metadatas.append(None)
                documents.append(None)
        return {"metadatas": metadatas, "documents": documents, "ids": ids}

    def update(self, ids, metadatas):
        for i, doc_id in enumerate(ids):
            if doc_id in self._store and i < len(metadatas):
                self._store[doc_id]["metadata"].update(metadatas[i] or {})


@pytest.fixture
def memory(tmp_path):
    """临时 MemoryService 实例（无 Chroma，长期记忆降级）

    注意：chromadb 安装后 MemoryService 会自动初始化 Chroma，
    此 fixture 通过 mock _ensure_chroma 为 no-op + _collection = None 模拟降级模式。
    """
    db_path = tmp_path / "test_memory.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None  # 强制降级模式
    # 阻止 write_semantic/search_semantic 调用时重新初始化 Chroma
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def memory_with_chroma(tmp_path):
    """带 fake Chroma 的 MemoryService（长期记忆可测）"""
    db_path = tmp_path / "test_memory_chroma.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma_fake")
    service._collection = _FakeChromaCollection()
    yield service
    service.close()


@pytest.fixture(autouse=True)
def _reset_tools_memory():
    """每个测试前后清理 tools 模块注入的 MemoryService，避免互相污染"""
    _set_memory_service(None)
    yield
    _set_memory_service(None)


def _make_case(case_id="C001", defect_type=DefectType.HARDNESS_LOW, **kw):
    """构造测试用 CaseRecord"""
    defaults = dict(
        case_id=case_id,
        defect_type=defect_type,
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
    defaults.update(kw)
    return CaseRecord(**defaults)


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
    """长期记忆测试（使用 fake Chroma，不再 skip）"""

    def test_write_and_search_semantic(self, memory_with_chroma):
        """应能写入并检索长期记忆"""
        memory = memory_with_chroma
        case = _make_case(case_id="C001")
        success = memory.write_semantic(case)
        assert success is True

        results = memory.search_semantic("硬度偏低 保温时间")
        assert len(results) > 0
        assert results[0]["metadata"]["defect_type"] == "hardness_low"

    def test_write_semantic_returns_false_without_chroma(self, memory):
        """无 Chroma 时写入应返回 False（降级模式）"""
        case = _make_case(case_id="C001")
        success = memory.write_semantic(case)
        assert success is False

    def test_search_empty_query_returns_empty(self, memory_with_chroma):
        """空库或无匹配查询应返回空列表"""
        results = memory_with_chroma.search_semantic("不存在的查询内容 xyz")
        assert isinstance(results, list)
        assert results == []

    def test_search_semantic_top_k_limit(self, memory_with_chroma):
        """应遵守 top_k 限制"""
        memory = memory_with_chroma
        for i in range(5):
            memory.write_semantic(_make_case(case_id=f"C{i:03d}"))
        results = memory.search_semantic("保温时间", top_k=2)
        assert len(results) <= 2

    def test_write_semantic_metadata_persisted(self, memory_with_chroma):
        """写入的 metadata（confidence/source/created_at）应能检索回"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_META", confidence=0.9, source="manual")
        memory.write_semantic(case)
        results = memory.search_semantic("保温时间")
        assert len(results) == 1
        meta = results[0]["metadata"]
        assert meta["confidence"] == 0.9
        assert meta["source"] == "manual"
        assert "created_at" in meta


class TestConfidenceUpdate:
    """置信度更新测试（使用 fake Chroma，不再 skip）"""

    def test_update_confidence(self, memory_with_chroma):
        """应能更新案例置信度"""
        memory = memory_with_chroma
        case = _make_case(case_id="C002", confidence=0.5)
        memory.write_semantic(case)

        updated = memory.update_confidence("C002", 0.9)
        assert updated is True

        # 验证加权更新：0.5 * 0.7 + 0.9 * 0.3 = 0.62
        results = memory.search_semantic("保温时间")
        meta = results[0]["metadata"]
        assert abs(meta["confidence"] - 0.62) < 0.01

    def test_update_nonexistent_case(self, memory_with_chroma):
        """更新不存在的案例应返回 False"""
        result = memory_with_chroma.update_confidence("nonexistent", 0.9)
        assert result is False

    def test_update_confidence_without_chroma(self, memory):
        """无 Chroma 时更新应返回 False"""
        result = memory.update_confidence("any", 0.9)
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

    def test_cleanup_keeps_high_quality(self, memory):
        """过期但高质的记忆应保留"""
        memory.write_episodic("B001", "hardness_low", "原因", "方案", quality_score=0.8)
        memory.db.execute(
            "UPDATE episodic SET created_at = ? WHERE batch_id = ?",
            (datetime.now() - timedelta(days=40), "B001"),
        )
        memory.db.commit()
        deleted = memory.cleanup_expired()
        assert deleted == 0  # quality_score=0.8 >= 0.3，不清理


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


class TestExtractDefectType:
    """memory_writer 缺陷类型抽取测试"""

    def test_from_defect_record(self):
        """应从 state['defect_record'] 抽取"""
        state = {"defect_record": {"defect_type": "crack"}}
        dtype = _extract_defect_type(state, "未知")
        assert dtype == DefectType.CRACK

    def test_from_defect_history(self):
        """应从 data_result.defect_history 抽取"""
        state = {
            "defect_record": None,
            "data_result": {"defect_history": {"records": [{"defect_type": "deformation"}]}},
        }
        dtype = _extract_defect_type(state, "未知")
        assert dtype == DefectType.DEFORMATION

    def test_from_root_cause_keyword(self):
        """应从 root_cause 关键词推断"""
        state = {"defect_record": None, "data_result": {}}
        assert _extract_defect_type(state, "裂纹扩展") == DefectType.CRACK
        assert _extract_defect_type(state, "组织异常 相变不完全") == DefectType.MICROSTRUCTURE
        assert _extract_defect_type(state, "硬度偏高") == DefectType.HARDNESS_HIGH

    def test_fallback_to_hardness_low(self):
        """无任何信号时兜底 hardness_low"""
        state = {"defect_record": None, "data_result": {}}
        dtype = _extract_defect_type(state, "未知原因")
        assert dtype == DefectType.HARDNESS_LOW

    def test_defect_record_takes_priority(self):
        """defect_record 优先级最高"""
        state = {
            "defect_record": {"defect_type": "crack"},
            "data_result": {"defect_history": {"records": [{"defect_type": "deformation"}]}},
        }
        dtype = _extract_defect_type(state, "硬度偏低")
        assert dtype == DefectType.CRACK


class TestFormatSolution:
    """memory_writer 方案格式化测试"""

    def test_dict_adjustments(self):
        """dict 类型 adjustments 应格式化为可读字符串"""
        result = _format_solution({"holding_time": "+15 分钟", "temperature": "+10℃"})
        assert "holding_time +15 分钟" in result
        assert "temperature +10℃" in result

    def test_empty_dict(self):
        """空 dict 应返回'无调整项'"""
        assert _format_solution({}) == "无调整项"

    def test_none(self):
        """None 应返回'未知'"""
        assert _format_solution(None) == "未知"

    def test_string_passthrough(self):
        """字符串应原样返回"""
        assert _format_solution("见最终回答") == "见最终回答"


class TestSearchCasesSemantic:
    """search_cases 语义检索接入测试"""

    def test_search_returns_semantic_when_memory_has_data(self, memory_with_chroma):
        """有长期记忆时应走 semantic_memory 来源"""
        _set_memory_service(memory_with_chroma)
        memory_with_chroma.write_semantic(_make_case(case_id="C_SEM_1"))

        result = search_cases("保温时间")
        assert result["_source"] == "semantic_memory"
        assert result["total"] >= 1
        assert result["results"][0]["defect_type"] == "hardness_low"

    def test_search_falls_back_to_mock_without_chroma(self, memory):
        """无 Chroma 时应降级到 mock_data"""
        _set_memory_service(memory)
        result = search_cases("硬度偏低")
        assert result["_source"] == "mock_data"
        assert isinstance(result["results"], list)

    def test_search_falls_back_when_empty(self, memory_with_chroma):
        """Chroma 可用但无数据时应降级到 mock_data"""
        _set_memory_service(memory_with_chroma)
        result = search_cases("任意查询")
        assert result["_source"] == "mock_data"

    def test_search_respects_top_k(self, memory_with_chroma):
        """应遵守 top_k 限制"""
        _set_memory_service(memory_with_chroma)
        for i in range(5):
            memory_with_chroma.write_semantic(_make_case(case_id=f"C_K_{i}"))
        result = search_cases("保温时间", top_k=2)
        assert result["total"] <= 2


class _RecordingMemoryService:
    """记录写入调用的 MemoryService 替身，用于隔离测试"""

    def __init__(self):
        self.episodic_writes = []
        self.semantic_writes = []

    def write_episodic(self, **kwargs):
        self.episodic_writes.append(kwargs)
        return "ep_test"

    def write_semantic(self, case):
        self.semantic_writes.append(case)
        return True


class TestMemoryWriterIsolation:
    """工作记忆隔离测试（M3-4）：memory_writer 只副作用，不修改 LangGraph 状态

    通过 memory_writer 函数的 __globals__ 直接操作模块全局命名空间，
    规避 agent/nodes/__init__.py 把模块名覆盖为函数名的问题。
    """

    def _inject_fake(self, monkeypatch, fake):
        """把 fake 注入 memory_writer 模块的 _memory_service 全局变量"""
        from agent.nodes import memory_writer as mw_func
        # _get_memory_service() 检查全局 _memory_service，非 None 时直接返回
        monkeypatch.setitem(mw_func.__globals__, "_memory_service", fake)

    def test_returns_empty_dict(self, monkeypatch):
        """应返回空字典，不修改 state"""
        from agent.nodes import memory_writer as mw_func

        fake = _RecordingMemoryService()
        self._inject_fake(monkeypatch, fake)

        state = {
            "trace_id": "tr_test",
            "batch_id": "B_TEST",
            "decision_result": {"proposals": [{"root_cause": "保温时间不足", "adjustments": {"holding_time": "+15"}}]},
            "data_result": {"batch_params": {"temperature": 850, "holding_time": 90}},
            "final_answer": None,
        }
        result = asyncio.run(mw_func(state))
        assert result == {}

    def test_does_not_mutate_state(self, monkeypatch):
        """调用后 state 内容应保持不变（除被读取外不应增删字段）"""
        from agent.nodes import memory_writer as mw_func
        import copy

        fake = _RecordingMemoryService()
        self._inject_fake(monkeypatch, fake)

        state = {
            "trace_id": "tr_test",
            "batch_id": "B_TEST",
            "decision_result": {"proposals": [{"root_cause": "裂纹扩展", "adjustments": {"temperature": "+10"}}]},
            "data_result": {"batch_params": {"temperature": 830}},
            "final_answer": "测试回答",
        }
        before = copy.deepcopy(state)
        asyncio.run(mw_func(state))
        assert state == before, "memory_writer 不应修改 state"

    def test_writes_to_both_memory_layers(self, monkeypatch):
        """应同时写入短期记忆和长期记忆"""
        from agent.nodes import memory_writer as mw_func

        fake = _RecordingMemoryService()
        self._inject_fake(monkeypatch, fake)

        state = {
            "trace_id": "tr_dual",
            "batch_id": "B_DUAL",
            "decision_result": {"proposals": [{"root_cause": "保温时间不足", "adjustments": {"holding_time": "+15"}}]},
            "data_result": {"batch_params": {"holding_time": 90}},
            "final_answer": None,
        }
        asyncio.run(mw_func(state))

        assert len(fake.episodic_writes) == 1
        assert fake.episodic_writes[0]["batch_id"] == "B_DUAL"
        assert len(fake.semantic_writes) == 1
        assert fake.semantic_writes[0].case_id == "case_tr_dual"

    def test_exception_does_not_propagate(self, monkeypatch):
        """记忆写入异常不应影响主流程（返回空字典）"""
        from agent.nodes import memory_writer as mw_func

        class _ExplodingMemory:
            def write_episodic(self, **kwargs):
                raise RuntimeError("DB 挂了")

            def write_semantic(self, case):
                raise RuntimeError("Chroma 挂了")

        self._inject_fake(monkeypatch, _ExplodingMemory())

        state = {
            "trace_id": "tr_err",
            "batch_id": "B_ERR",
            "decision_result": {"proposals": []},
            "data_result": {},
            "final_answer": "兜底回答",
        }
        result = asyncio.run(mw_func(state))
        assert result == {}, "异常时应静默返回空字典，不污染 state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
