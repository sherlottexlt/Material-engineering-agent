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
    """模拟 Chroma collection 的内存实现，覆盖 add/query/get/update/delete 接口。

    用关键词重叠度近似语义相关度，足够测试 MemoryService 的写入/检索/更新/删除逻辑。
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

    def get(self, ids=None, limit=None, include=None):
        """支持按 ids 查询或全量查询（limit 限制）。

        - ids 提供时：只返回存在的 ID（过滤不存在的，与真实 Chroma 一致）
        - ids 为 None 时：返回全部（受 limit 限制）
        """
        if ids is not None:
            out_ids, out_docs, out_metas = [], [], []
            for doc_id in ids:
                if doc_id in self._store:
                    out_ids.append(doc_id)
                    out_docs.append(self._store[doc_id]["document"])
                    out_metas.append(dict(self._store[doc_id]["metadata"]))
            return {"ids": out_ids, "documents": out_docs, "metadatas": out_metas}

        # 全量查询
        items = list(self._store.items())
        if limit is not None:
            items = items[:limit]
        return {
            "ids": [doc_id for doc_id, _ in items],
            "documents": [item["document"] for _, item in items],
            "metadatas": [dict(item["metadata"]) for _, item in items],
        }

    def update(self, ids, metadatas=None, documents=None):
        """更新 metadata 和/或 document（部分更新）"""
        for i, doc_id in enumerate(ids):
            if doc_id not in self._store:
                continue
            if metadatas is not None and i < len(metadatas) and metadatas[i]:
                self._store[doc_id]["metadata"].update(metadatas[i])
            if documents is not None and i < len(documents):
                self._store[doc_id]["document"] = documents[i]

    def delete(self, ids):
        """删除指定 ids"""
        for doc_id in ids:
            self._store.pop(doc_id, None)


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


# ===== M3-9: 知识冲突检测测试 =====

from agent.memory.memory_service import (
    ConflictRecord, CONFLICT_HARD, CONFLICT_SOFT, CONFLICT_CONFIDENCE,
)


def _make_case_with_params(case_id, temperature=850, holding_time=120, **kw):
    """构造带自定义工艺参数的测试案例"""
    defaults = dict(
        case_id=case_id,
        defect_type=DefectType.HARDNESS_LOW,
        batch_params=BatchParams(
            batch_id="B001",
            process_type=ProcessType.HEAT_TREATMENT,
            temperature=temperature,
            holding_time=holding_time,
            start_time=datetime.now(),
        ),
        root_cause="保温时间不足",
        solution="holding_time +15",
        confidence=0.8,
    )
    defaults.update(kw)
    return CaseRecord(**defaults)


class TestConflictDetection:
    """M3-9 知识冲突检测测试"""

    def test_detect_hard_conflict(self, memory_with_chroma):
        """硬冲突：相同缺陷 + 相似参数 + 不同根因"""
        memory = memory_with_chroma
        case1 = _make_case_with_params("C_HD1", root_cause="保温时间不足", confidence=0.8)
        memory.write_semantic(case1)

        # 案例2：相同缺陷 + 相似参数 + 不同根因
        case2 = _make_case_with_params("C_HD2", root_cause="冷却速率过低", confidence=0.7)
        conflicts = memory.detect_conflicts(case2)

        assert len(conflicts) >= 1
        assert any(c.conflict_type == CONFLICT_HARD for c in conflicts)

    def test_detect_soft_conflict(self, memory_with_chroma):
        """软冲突：相同根因 + 不同方案"""
        memory = memory_with_chroma
        case1 = _make_case_with_params("C_SF1", root_cause="保温时间不足", solution="holding_time +15")
        memory.write_semantic(case1)

        # 案例2：相同根因 + 不同方案
        case2 = _make_case_with_params("C_SF2", root_cause="保温时间不足", solution="holding_time +20", confidence=0.75)
        conflicts = memory.detect_conflicts(case2)

        # 根因相同，不应有 hard 冲突
        assert not any(c.conflict_type == CONFLICT_HARD for c in conflicts)
        # 方案不同，应有 soft 冲突
        assert any(c.conflict_type == CONFLICT_SOFT for c in conflicts)

    def test_detect_confidence_conflict(self, memory_with_chroma):
        """置信度冲突：置信度差异 > 0.3"""
        memory = memory_with_chroma
        case1 = _make_case_with_params("C_CF1", root_cause="保温时间不足", solution="holding_time +15", confidence=0.9)
        memory.write_semantic(case1)

        # 案例2：相同根因 + 相同方案 + 置信度差异大
        case2 = _make_case_with_params("C_CF2", root_cause="保温时间不足", solution="holding_time +15", confidence=0.4)
        conflicts = memory.detect_conflicts(case2)

        assert any(c.conflict_type == CONFLICT_CONFIDENCE for c in conflicts)

    def test_no_conflict_different_defect(self, memory_with_chroma):
        """不同缺陷类型无冲突"""
        memory = memory_with_chroma
        case1 = _make_case_with_params("C_ND1", defect_type=DefectType.HARDNESS_LOW)
        memory.write_semantic(case1)

        case2 = _make_case_with_params(
            "C_ND2", defect_type=DefectType.CRACK,
            root_cause="裂纹扩展", solution="检查原材料",
        )
        conflicts = memory.detect_conflicts(case2)

        assert len(conflicts) == 0

    def test_no_conflict_different_params(self, memory_with_chroma):
        """工艺参数差异大无冲突"""
        memory = memory_with_chroma
        case1 = _make_case_with_params("C_NP1", temperature=850, holding_time=120)
        memory.write_semantic(case1)

        # 温度差 50°C，远超容差 10°C
        case2 = _make_case_with_params(
            "C_NP2", temperature=900, holding_time=120,
            root_cause="温度偏高", solution="temperature -20",
        )
        conflicts = memory.detect_conflicts(case2)

        assert len(conflicts) == 0

    def test_params_similar_old_data(self, memory_with_chroma):
        """旧数据无工艺参数时降级为相似"""
        memory = memory_with_chroma
        # 手动写入旧格式 metadata（无 temperature/holding_time）
        memory._collection.add(
            ids=["C_OLD"],
            documents=["hardness_low\n保温时间不足\nholding_time +15"],
            metadatas=[{"defect_type": "hardness_low", "confidence": 0.8, "source": "manual"}],
        )

        case = _make_case_with_params("C_NEW", root_cause="冷却速率过低", confidence=0.7)
        conflicts = memory.detect_conflicts(case)

        # 旧数据无工艺参数，降级为相似，应检测到 hard 冲突
        assert any(c.conflict_type == CONFLICT_HARD for c in conflicts)

    def test_root_cause_similar(self, memory):
        """根因相似度判断（bigram 重叠）"""
        assert memory._root_cause_similar("保温时间不足", "保温时间不足") is True
        assert memory._root_cause_similar("保温时间不足", "保温时间严重不足") is True
        assert memory._root_cause_similar("保温时间不足", "冷却速率过低") is False
        assert memory._root_cause_similar("", "保温时间不足") is False
        assert memory._root_cause_similar("保温时间不足", "") is False
        assert memory._root_cause_similar("a", "a") is False  # 单字符无 bigram

    def test_extract_field(self, memory):
        """从 document 中提取字段"""
        doc = "hardness_low\n保温时间不足\nholding_time +15"
        assert memory._extract_field(doc, "defect_type") == "hardness_low"
        assert memory._extract_field(doc, "root_cause") == "保温时间不足"
        assert memory._extract_field(doc, "solution") == "holding_time +15"
        assert memory._extract_field(doc, "unknown") == ""
        assert memory._extract_field("only_one_line", "root_cause") == ""

    def test_save_and_list_conflict(self, memory):
        """保存 + 查询冲突记录"""
        now = datetime.now()
        conflict = ConflictRecord(
            conflict_id="cf_test_001",
            new_case_id="C_NEW",
            existing_case_id="C_OLD",
            conflict_type=CONFLICT_HARD,
            description="测试冲突",
            created_at=now,
        )

        ok = memory.save_conflict(conflict)
        assert ok is True

        results = memory.list_conflicts()
        assert len(results) == 1
        assert results[0]["conflict_id"] == "cf_test_001"
        assert results[0]["conflict_type"] == "hard"

    def test_write_semantic_triggers_conflict_detection(self, memory_with_chroma):
        """write_semantic 自动检测冲突并记录到 SQLite"""
        memory = memory_with_chroma
        case1 = _make_case_with_params("C_W1", root_cause="保温时间不足", confidence=0.8)
        memory.write_semantic(case1)

        # 写入冲突案例（不同根因 → hard 冲突）
        case2 = _make_case_with_params("C_W2", root_cause="冷却速率过低", confidence=0.7)
        memory.write_semantic(case2)

        # 应该检测到冲突并记录
        conflicts = memory.list_conflicts()
        assert len(conflicts) >= 1
        assert any(c["conflict_type"] == "hard" for c in conflicts)

    def test_detect_conflicts_without_chroma(self, memory):
        """无 Chroma 时检测返回空列表"""
        case = _make_case_with_params("C_NC1")
        conflicts = memory.detect_conflicts(case)
        assert conflicts == []

    def test_self_not_conflict(self, memory_with_chroma):
        """检测时排除自身（相同 case_id）"""
        memory = memory_with_chroma
        case = _make_case_with_params("C_SELF", root_cause="保温时间不足", confidence=0.8)
        memory.write_semantic(case)

        # 再次检测同一个案例，不应产生冲突
        conflicts = memory.detect_conflicts(case)
        assert len(conflicts) == 0


class TestCaseCRUD:
    """M3-7 案例库 CRUD 测试"""

    def test_get_semantic_case_returns_case(self, memory_with_chroma):
        """应能按 ID 获取案例"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_GET1", root_cause="保温时间不足", solution="holding_time +15")
        memory.write_semantic(case)

        record = memory.get_semantic_case("C_GET1")
        assert record is not None
        assert record["id"] == "C_GET1"
        assert "保温时间不足" in record["document"]
        assert record["metadata"]["defect_type"] == "hardness_low"

    def test_get_semantic_case_nonexistent_returns_none(self, memory_with_chroma):
        """获取不存在的案例应返回 None"""
        memory = memory_with_chroma
        assert memory.get_semantic_case("nonexistent_id") is None

    def test_get_semantic_case_without_chroma_returns_none(self, memory):
        """无 Chroma 时获取应返回 None"""
        assert memory.get_semantic_case("any_id") is None

    def test_update_semantic_root_cause(self, memory_with_chroma):
        """应能只更新根因（保留其他字段）"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_UP1", root_cause="旧根因", solution="旧方案", confidence=0.8)
        memory.write_semantic(case)

        ok = memory.update_semantic("C_UP1", root_cause="新根因")
        assert ok is True

        record = memory.get_semantic_case("C_UP1")
        assert "新根因" in record["document"]
        assert "旧方案" in record["document"]  # solution 保留
        assert record["metadata"]["confidence"] == 0.8  # confidence 保留

    def test_update_semantic_confidence(self, memory_with_chroma):
        """应能只更新置信度"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_UP2", confidence=0.5)
        memory.write_semantic(case)

        ok = memory.update_semantic("C_UP2", confidence=0.95)
        assert ok is True

        record = memory.get_semantic_case("C_UP2")
        assert abs(record["metadata"]["confidence"] - 0.95) < 0.01

    def test_update_semantic_solution(self, memory_with_chroma):
        """应能只更新解决方案"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_UP3", solution="旧方案")
        memory.write_semantic(case)

        ok = memory.update_semantic("C_UP3", solution="新方案")
        assert ok is True

        record = memory.get_semantic_case("C_UP3")
        assert "新方案" in record["document"]
        assert "旧方案" not in record["document"]

    def test_update_semantic_tags_set(self, memory_with_chroma):
        """应能设置标签（存为逗号分隔字符串）"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_TAG1", tags=[])
        memory.write_semantic(case)

        ok = memory.update_semantic("C_TAG1", tags=["紧急", "参考案例"])
        assert ok is True

        record = memory.get_semantic_case("C_TAG1")
        assert record["metadata"]["tags"] == "紧急,参考案例"

    def test_update_semantic_tags_clear(self, memory_with_chroma):
        """空标签列表应清空标签字段"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_TAG2", tags=["紧急"])
        memory.write_semantic(case)
        # 确认标签已写入
        assert memory.get_semantic_case("C_TAG2")["metadata"].get("tags") == "紧急"

        ok = memory.update_semantic("C_TAG2", tags=[])
        assert ok is True

        record = memory.get_semantic_case("C_TAG2")
        assert "tags" not in record["metadata"]

    def test_update_semantic_nonexistent_returns_false(self, memory_with_chroma):
        """更新不存在的案例应返回 False"""
        memory = memory_with_chroma
        assert memory.update_semantic("nonexistent", root_cause="x") is False

    def test_update_semantic_without_chroma_returns_false(self, memory):
        """无 Chroma 时更新应返回 False"""
        assert memory.update_semantic("any", root_cause="x") is False

    def test_delete_semantic_removes_case(self, memory_with_chroma):
        """应能删除案例"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_DEL1")
        memory.write_semantic(case)
        assert memory.get_semantic_case("C_DEL1") is not None

        ok = memory.delete_semantic("C_DEL1")
        assert ok is True
        assert memory.get_semantic_case("C_DEL1") is None

    def test_delete_semantic_nonexistent_returns_true(self, memory_with_chroma):
        """删除不存在的案例应返回 True（幂等，与真实 Chroma 一致）"""
        memory = memory_with_chroma
        assert memory.delete_semantic("nonexistent") is True

    def test_delete_semantic_without_chroma_returns_false(self, memory):
        """无 Chroma 时删除应返回 False"""
        assert memory.delete_semantic("any") is False

    def test_delete_semantic_cleans_conflicts(self, memory_with_chroma):
        """删除案例应清理引用该案例的冲突记录"""
        memory = memory_with_chroma
        # 写入两条相似但根因不同的案例 → 产生 hard 冲突
        case1 = _make_case_with_params("C_DC1", root_cause="保温时间不足", confidence=0.8)
        memory.write_semantic(case1)
        case2 = _make_case_with_params("C_DC2", root_cause="冷却速率过低", confidence=0.7)
        memory.write_semantic(case2)

        conflicts_before = memory.list_conflicts()
        assert len(conflicts_before) >= 1

        # 删除新案例 C_DC2（作为 new_case_id）
        memory.delete_semantic("C_DC2")

        # 引用 C_DC2 的冲突记录应被清理
        conflicts_after = memory.list_conflicts()
        for c in conflicts_after:
            assert c["new_case_id"] != "C_DC2"
            assert c["existing_case_id"] != "C_DC2"

    def test_write_semantic_persists_tags(self, memory_with_chroma):
        """写入案例时 tags 应存为逗号分隔字符串"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_WT", tags=["紧急", "参考", "实验证实"])
        memory.write_semantic(case)

        record = memory.get_semantic_case("C_WT")
        assert record["metadata"]["tags"] == "紧急,参考,实验证实"

    def test_crud_full_cycle(self, memory_with_chroma):
        """完整 CRUD 生命周期：创建 → 读取 → 更新 → 删除"""
        memory = memory_with_chroma

        # Create
        case = _make_case(case_id="C_CRUD", root_cause="初始根因", confidence=0.5, tags=["v1"])
        memory.write_semantic(case)
        assert memory.get_semantic_case("C_CRUD") is not None

        # Read
        record = memory.get_semantic_case("C_CRUD")
        assert record["metadata"]["defect_type"] == "hardness_low"
        assert record["metadata"]["tags"] == "v1"

        # Update（根因 + 置信度 + 标签）
        memory.update_semantic(
            "C_CRUD",
            root_cause="修正根因",
            confidence=0.9,
            tags=["v2", "已修正"],
        )
        record = memory.get_semantic_case("C_CRUD")
        assert "修正根因" in record["document"]
        assert abs(record["metadata"]["confidence"] - 0.9) < 0.01
        assert record["metadata"]["tags"] == "v2,已修正"

        # Delete
        memory.delete_semantic("C_CRUD")
        assert memory.get_semantic_case("C_CRUD") is None


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
