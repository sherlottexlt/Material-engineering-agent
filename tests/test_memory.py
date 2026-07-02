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
from agent.nodes.memory_writer import (
    _extract_defect_type,
    _extract_root_cause_tags,
    _assess_confidence,
    _format_solution,
)
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

    def query(self, query_texts, n_results, where=None):
        """语义检索（可选 where 过滤，模拟 Chroma metadata where）

        Args:
            query_texts: 查询文本列表
            n_results: 返回数量
            where: metadata 过滤条件 dict，如 {"line_id": "welding"}
        """
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        if not self._store or not query_texts:
            return results
        query_text = query_texts[0] or ""
        scored = []
        for doc_id, item in self._store.items():
            # M4-9: where 过滤（模拟 Chroma metadata where）
            if where:
                meta = item.get("metadata", {})
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue
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
        """更新 metadata 和/或 document

        注意：Chroma 的 update 是替换整个 metadata/document，不是合并。
        这与真实 Chroma 行为一致（update_semantic 依赖此语义来移除字段）。
        """
        for i, doc_id in enumerate(ids):
            if doc_id not in self._store:
                continue
            if metadatas is not None and i < len(metadatas):
                self._store[doc_id]["metadata"] = dict(metadatas[i])
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


class TestExtractRootCauseTags:
    """M3-6: 根因标签自动抽取测试"""

    def test_single_root_cause_holding_time(self):
        """单根因：保温时间不足"""
        tags = _extract_root_cause_tags("保温时间不足")
        assert "保温时间不足" in tags
        assert "holding_time" in tags

    def test_single_root_cause_cooling_rate(self):
        """单根因：冷却速率过低"""
        tags = _extract_root_cause_tags("冷却速率过低")
        assert "冷却速率过低" in tags
        assert "cooling_rate" in tags

    def test_single_root_cause_temperature(self):
        """单根因：温度偏低"""
        tags = _extract_root_cause_tags("温度偏低")
        assert "温度偏低" in tags
        assert "temperature" in tags

    def test_multi_root_cause_combo(self):
        """多根因组合：保温时间不足+冷却速率过低"""
        tags = _extract_root_cause_tags("保温时间不足+冷却速率过低")
        assert "保温时间不足" in tags
        assert "holding_time" in tags
        assert "冷却速率过低" in tags
        assert "cooling_rate" in tags

    def test_multi_root_cause_three_way(self):
        """三根因组合（虽不常见，但应支持）"""
        tags = _extract_root_cause_tags("保温时间不足+冷却速率过低+温度偏低")
        assert "保温时间不足" in tags
        assert "冷却速率过低" in tags
        assert "温度偏低" in tags

    def test_synonym_time_insufficient(self):
        """同义词：时间不足 → 保温时间不足"""
        tags = _extract_root_cause_tags("时间不足")
        assert "保温时间不足" in tags
        assert "holding_time" in tags

    def test_synonym_time_not_enough(self):
        """同义词：时间不够 → 保温时间不足"""
        tags = _extract_root_cause_tags("时间不够")
        assert "保温时间不足" in tags

    def test_fullwidth_plus_separator(self):
        """全角＋分隔符应正常解析"""
        tags = _extract_root_cause_tags("保温时间不足＋冷却速率过低")
        assert "保温时间不足" in tags
        assert "冷却速率过低" in tags

    def test_empty_string(self):
        """空字符串返回空列表"""
        assert _extract_root_cause_tags("") == []

    def test_none_input(self):
        """None 输入返回空列表"""
        assert _extract_root_cause_tags(None) == []

    def test_unknown_root_cause(self):
        """未知根因返回空列表"""
        assert _extract_root_cause_tags("未知原因") == []

    def test_no_duplicates(self):
        """重复关键词不应产生重复标签"""
        tags = _extract_root_cause_tags("保温时间不足+保温时间不足")
        assert tags.count("保温时间不足") == 1
        assert tags.count("holding_time") == 1


class TestAssessConfidence:
    """M3-6: 置信度自动评估测试"""

    def test_full_signals_high_confidence(self):
        """全信号齐全应得高置信度（0.9 上限）"""
        state = {"defect_record": {"defect_type": "hardness_low"}}
        batch_params = {"temperature": 844, "holding_time": 80, "cooling_rate": 5.7}
        proposals = [{"root_cause": "保温时间不足"}]
        confidence = _assess_confidence(state, "保温时间不足", batch_params, proposals)
        # 0.3 + 0.3(3 参数) + 0.2(根因明确) + 0.1(proposals) + 0.1(defect_record) = 1.0 → 上限 0.9
        assert confidence == 0.9

    def test_minimal_signals_low_confidence(self):
        """无任何信号应得基础值 0.3"""
        state = {}
        confidence = _assess_confidence(state, "", {}, [])
        assert confidence == 0.3

    def test_data_completeness(self):
        """数据完整性：每个参数 +0.1"""
        state = {}
        # 1 个参数
        confidence_1 = _assess_confidence(state, "", {"temperature": 844}, [])
        assert confidence_1 == 0.4  # 0.3 + 0.1
        # 2 个参数
        confidence_2 = _assess_confidence(
            state, "", {"temperature": 844, "holding_time": 80}, []
        )
        assert confidence_2 == 0.5  # 0.3 + 0.2
        # 3 个参数
        confidence_3 = _assess_confidence(
            state, "", {"temperature": 844, "holding_time": 80, "cooling_rate": 5.7}, []
        )
        assert confidence_3 == 0.6  # 0.3 + 0.3

    def test_root_cause_explicitness(self):
        """根因明确性：含标准关键词 +0.2"""
        state = {}
        batch_params = {}
        confidence = _assess_confidence(state, "保温时间不足", batch_params, [])
        assert confidence == 0.5  # 0.3 + 0.2

    def test_root_cause_unknown(self):
        """未知根因不加分"""
        state = {}
        confidence = _assess_confidence(state, "未知原因", {}, [])
        assert confidence == 0.3

    def test_proposals_existence(self):
        """有 proposals +0.1"""
        state = {}
        confidence = _assess_confidence(state, "", {}, [{"root_cause": "x"}])
        assert confidence == 0.4  # 0.3 + 0.1

    def test_defect_record_existence(self):
        """有 defect_record +0.1"""
        state = {"defect_record": {"defect_type": "hardness_low"}}
        confidence = _assess_confidence(state, "", {}, [])
        assert confidence == 0.4  # 0.3 + 0.1

    def test_capped_at_0_9(self):
        """置信度上限 0.9"""
        state = {"defect_record": {"defect_type": "hardness_low"}}
        batch_params = {"temperature": 844, "holding_time": 80, "cooling_rate": 5.7}
        proposals = [{"root_cause": "保温时间不足"}]
        confidence = _assess_confidence(state, "保温时间不足", batch_params, proposals)
        assert confidence <= 0.9

    def test_multi_root_cause_gets_explicitness_bonus(self):
        """多根因组合也享受明确性加分"""
        state = {}
        confidence = _assess_confidence(state, "保温时间不足+冷却速率过低", {}, [])
        assert confidence == 0.5  # 0.3 + 0.2


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


# ===== M3-10: 反馈驱动权重更新测试 =====


class TestAggregateFeedbackUpdate:
    """M3-10: 反馈聚合置信度更新测试"""

    def test_no_feedback_returns_not_updated(self, memory_with_chroma):
        """无反馈时返回 updated=False"""
        memory = memory_with_chroma
        memory.write_semantic(_make_case(case_id="C_NO_FB", confidence=0.5))
        result = memory.aggregate_feedback_update("C_NO_FB")
        assert result["updated"] is False
        assert result["feedback_count"] == 0

    def test_single_feedback_updates_confidence(self, memory_with_chroma):
        """单条反馈应更新置信度"""
        memory = memory_with_chroma
        memory.write_semantic(_make_case(case_id="C_SINGLE", confidence=0.5))
        memory.write_feedback("fb_1", "C_SINGLE", "u1", "adopted", 0.9)

        result = memory.aggregate_feedback_update("C_SINGLE")
        assert result["updated"] is True
        assert result["feedback_count"] == 1
        assert abs(result["old_confidence"] - 0.5) < 0.01
        # new_conf = 0.5 * 0.5 + 0.9 * 0.5 = 0.7
        assert abs(result["new_confidence"] - 0.7) < 0.01

        # 验证已持久化
        case = memory.get_semantic_case("C_SINGLE")
        assert abs(case["metadata"]["confidence"] - 0.7) < 0.01

    def test_multiple_feedbacks_aggregated(self, memory_with_chroma):
        """多条反馈应聚合加权"""
        memory = memory_with_chroma
        memory.write_semantic(_make_case(case_id="C_MULTI", confidence=0.5))
        memory.write_feedback("fb_1", "C_MULTI", "u1", "adopted", 0.9)
        memory.write_feedback("fb_2", "C_MULTI", "u2", "rejected", 0.1)

        result = memory.aggregate_feedback_update("C_MULTI")
        assert result["updated"] is True
        assert result["feedback_count"] == 2
        # 两条都是新反馈（weight=1.0），aggregated = (0.9+0.1)/2 = 0.5
        # new_conf = 0.5*0.5 + 0.5*0.5 = 0.5
        assert abs(result["new_confidence"] - 0.5) < 0.01

    def test_time_decay_weighting(self, memory_with_chroma):
        """旧反馈权重应降低，新反馈占主导"""
        memory = memory_with_chroma
        memory.write_semantic(_make_case(case_id="C_DECAY", confidence=0.5))
        # 旧反馈（90天前，score=0.1）
        memory.write_feedback("fb_old", "C_DECAY", "u1", "rejected", 0.1)
        memory.db.execute(
            "UPDATE feedback SET created_at = ? WHERE feedback_id = ?",
            (datetime.now() - timedelta(days=90), "fb_old"),
        )
        memory.db.commit()
        # 新反馈（score=0.9）
        memory.write_feedback("fb_new", "C_DECAY", "u2", "adopted", 0.9)

        result = memory.aggregate_feedback_update("C_DECAY", days=90)
        assert result["updated"] is True
        # 旧反馈 weight=0.1，新反馈 weight=1.0
        # aggregated ≈ 0.827，new_conf ≈ 0.664
        assert result["new_confidence"] > 0.6  # 新反馈占主导

    def test_nonexistent_case_not_updated(self, memory_with_chroma):
        """案例不存在时返回 updated=False"""
        memory = memory_with_chroma
        memory.write_feedback("fb_1", "C_GHOST", "u1", "adopted", 0.9)
        result = memory.aggregate_feedback_update("C_GHOST")
        assert result["updated"] is False
        assert result["feedback_count"] == 1

    def test_without_chroma_not_updated(self, memory):
        """无 Chroma 时返回 updated=False"""
        memory.write_feedback("fb_1", "C001", "u1", "adopted", 0.9)
        result = memory.aggregate_feedback_update("C001")
        assert result["updated"] is False


class TestBatchUpdateConfidence:
    """M3-10: 批量反馈置信度更新测试"""

    def test_batch_update_multiple_cases(self, memory_with_chroma):
        """批量更新多个案例"""
        memory = memory_with_chroma
        memory.write_semantic(_make_case(case_id="C_B1", confidence=0.5))
        memory.write_semantic(_make_case(case_id="C_B2", confidence=0.5))
        memory.write_feedback("fb_1", "C_B1", "u1", "adopted", 0.9)
        memory.write_feedback("fb_2", "C_B2", "u2", "rejected", 0.1)
        memory.write_feedback("fb_3", "C_B1", "u3", "adopted", 0.8)

        result = memory.batch_update_confidence_from_feedback(days=30)
        assert result["processed_cases"] == 2
        assert result["total_feedback"] == 3
        assert result["updated"] == 2
        assert result["skipped"] == 0

    def test_batch_no_feedback(self, memory_with_chroma):
        """无反馈时返回空统计"""
        result = memory_with_chroma.batch_update_confidence_from_feedback(days=30)
        assert result["processed_cases"] == 0
        assert result["total_feedback"] == 0
        assert result["updated"] == 0


# ===== M3-13: 遗忘机制扩展测试 =====


class TestCleanupAll:
    """M3-13: 扩展清理测试"""

    def test_default_no_feedback_cleanup(self, memory):
        """默认不清理反馈"""
        memory.write_episodic("B001", "hardness_low", "原因", "方案")
        memory.write_feedback("fb_1", "P001", "u1", "adopted", 0.9)

        # 让短期记忆过期低质
        memory.db.execute(
            "UPDATE episodic SET created_at = ?, quality_score = 0.1 WHERE batch_id = ?",
            (datetime.now() - timedelta(days=40), "B001"),
        )
        memory.db.commit()

        result = memory.cleanup_all()
        assert result["episodic_deleted"] == 1
        assert result["feedback_deleted"] == 0
        # 反馈仍在
        assert len(memory.query_feedback(days=30)) == 1

    def test_cleanup_with_feedback(self, memory):
        """also_cleanup_feedback=True 清理过期反馈"""
        memory.write_feedback("fb_old", "P001", "u1", "adopted", 0.9)
        memory.write_feedback("fb_new", "P002", "u2", "rejected", 0.1)

        # fb_old 过期（retention_days*2 = 60 天）
        memory.db.execute(
            "UPDATE feedback SET created_at = ? WHERE feedback_id = ?",
            (datetime.now() - timedelta(days=70), "fb_old"),
        )
        memory.db.commit()

        result = memory.cleanup_all(also_cleanup_feedback=True)
        assert result["feedback_deleted"] == 1
        # fb_new 仍在
        remaining = memory.query_feedback(days=100)
        assert len(remaining) == 1
        assert remaining[0]["feedback_id"] == "fb_new"


class TestArchiveLowQuality:
    """M3-13: 低质长期记忆归档测试"""

    def test_archive_low_quality_old_cases(self, memory_with_chroma, tmp_path):
        """应归档低质且陈旧的案例"""
        memory = memory_with_chroma
        archive_file = tmp_path / "archive.json"

        # 低质陈旧案例（应归档）
        memory.write_semantic(_make_case(
            case_id="C_OLD_LOW", confidence=0.1,
            created_at=datetime.now() - timedelta(days=100),
        ))
        # 高质量案例（应保留）
        memory.write_semantic(_make_case(
            case_id="C_GOOD", confidence=0.9,
            created_at=datetime.now() - timedelta(days=100),
        ))
        # 低质但新（应保留）
        memory.write_semantic(_make_case(case_id="C_NEW_LOW", confidence=0.1))

        result = memory.archive_low_quality_semantic(
            min_confidence=0.3, min_age_days=90, archive_path=archive_file,
        )

        assert result["archived"] == 1
        assert result["remaining"] == 2
        assert result["archive_total"] == 1
        # 低质陈旧案例已移除
        assert memory.get_semantic_case("C_OLD_LOW") is None
        assert memory.get_semantic_case("C_GOOD") is not None
        assert memory.get_semantic_case("C_NEW_LOW") is not None

        # 验证归档文件内容
        import json
        archived = json.loads(archive_file.read_text(encoding="utf-8"))
        assert len(archived) == 1
        assert archived[0]["id"] == "C_OLD_LOW"
        assert "archived_at" in archived[0]

    def test_archive_keeps_high_quality(self, memory_with_chroma, tmp_path):
        """高质量案例不应被归档"""
        memory = memory_with_chroma
        memory.write_semantic(_make_case(
            case_id="C_OLD_GOOD", confidence=0.9,
            created_at=datetime.now() - timedelta(days=100),
        ))

        result = memory.archive_low_quality_semantic(
            min_confidence=0.3, min_age_days=90,
            archive_path=tmp_path / "archive.json",
        )

        assert result["archived"] == 0
        assert result["remaining"] == 1
        assert memory.get_semantic_case("C_OLD_GOOD") is not None

    def test_archive_keeps_new_low_quality(self, memory_with_chroma, tmp_path):
        """低质但新的案例不应被归档"""
        memory = memory_with_chroma
        memory.write_semantic(_make_case(case_id="C_NEW_LOW", confidence=0.1))

        result = memory.archive_low_quality_semantic(
            min_confidence=0.3, min_age_days=90,
            archive_path=tmp_path / "archive.json",
        )

        assert result["archived"] == 0
        assert memory.get_semantic_case("C_NEW_LOW") is not None

    def test_archive_accumulates(self, memory_with_chroma, tmp_path):
        """多次归档应累积到同一文件"""
        memory = memory_with_chroma
        archive_file = tmp_path / "archive.json"

        # 第一次归档
        memory.write_semantic(_make_case(
            case_id="C_A1", confidence=0.1,
            created_at=datetime.now() - timedelta(days=100),
        ))
        memory.archive_low_quality_semantic(
            min_confidence=0.3, min_age_days=90, archive_path=archive_file,
        )

        # 第二次归档
        memory.write_semantic(_make_case(
            case_id="C_A2", confidence=0.1,
            created_at=datetime.now() - timedelta(days=100),
        ))
        result = memory.archive_low_quality_semantic(
            min_confidence=0.3, min_age_days=90, archive_path=archive_file,
        )

        assert result["archived"] == 1
        assert result["archive_total"] == 2

    def test_archive_without_chroma(self, memory, tmp_path):
        """无 Chroma 时返回空统计"""
        result = memory.archive_low_quality_semantic(
            archive_path=tmp_path / "archive.json",
        )
        assert result["archived"] == 0
        assert result["remaining"] == 0
        assert result["archive_total"] == 0

    def test_archive_empty_library(self, memory_with_chroma, tmp_path):
        """空库归档返回 0"""
        result = memory_with_chroma.archive_low_quality_semantic(
            archive_path=tmp_path / "archive.json",
        )
        assert result["archived"] == 0
        assert result["remaining"] == 0


class TestArchiveStats:
    """M3-13: 归档统计测试"""

    def test_stats_empty_archive(self, memory, tmp_path):
        """空归档文件返回零统计"""
        result = memory.get_archive_stats(archive_path=tmp_path / "nonexistent.json")
        assert result["total"] == 0
        assert result["by_defect_type"] == {}
        assert result["avg_confidence"] == 0.0

    def test_stats_with_archived_cases(self, memory_with_chroma, tmp_path):
        """应正确统计归档案例"""
        memory = memory_with_chroma
        archive_file = tmp_path / "archive.json"

        for i in range(3):
            memory.write_semantic(_make_case(
                case_id=f"C_ARC_{i}", confidence=0.1,
                created_at=datetime.now() - timedelta(days=100),
            ))

        memory.archive_low_quality_semantic(
            min_confidence=0.3, min_age_days=90, archive_path=archive_file,
        )

        stats = memory.get_archive_stats(archive_path=archive_file)
        assert stats["total"] == 3
        assert stats["by_defect_type"].get("hardness_low") == 3
        assert stats["avg_confidence"] < 0.3  # 都是低质


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


# ===== M4-9: 多产线隔离测试 =====


class TestMultiLineIsolation:
    """M4-9: 多产线数据隔离测试

    验证 line_id 贯穿 episodic / semantic / feedback / conflicts 四层记忆，
    不同产线的数据互不干扰。
    """

    def test_episodic_isolation_by_line_id(self, memory):
        """短期记忆应按 line_id 隔离：写入两条不同产线，按 line_id 查询只返回对应产线"""
        memory.write_episodic("B001", "hardness_low", "原因A", "方案A", line_id="heat_treatment")
        memory.write_episodic("B002", "porosity", "原因B", "方案B", line_id="welding")

        # 按 heat_treatment 查询，只返回 1 条
        ht_results = memory.query_episodic(line_id="heat_treatment", days=365)
        assert len(ht_results) == 1
        assert ht_results[0]["line_id"] == "heat_treatment"
        assert ht_results[0]["batch_id"] == "B001"

        # 按 welding 查询，只返回 1 条
        wd_results = memory.query_episodic(line_id="welding", days=365)
        assert len(wd_results) == 1
        assert wd_results[0]["line_id"] == "welding"
        assert wd_results[0]["batch_id"] == "B002"

        # 不传 line_id 查询全部
        all_results = memory.query_episodic(days=365)
        assert len(all_results) == 2

    def test_episodic_default_line_id(self, memory):
        """write_episodic 不传 line_id 时默认 heat_treatment"""
        memory.write_episodic("B001", "hardness_low", "原因A", "方案A")
        results = memory.query_episodic(line_id="heat_treatment", days=365)
        assert len(results) == 1
        assert results[0]["line_id"] == "heat_treatment"

    def test_feedback_isolation_by_line_id(self, memory):
        """用户反馈应按 line_id 隔离"""
        memory.write_feedback("fb1", "P001", "u1", "adopted", 0.9, line_id="heat_treatment")
        memory.write_feedback("fb2", "P002", "u2", "rejected", 0.2, line_id="welding")

        ht = memory.query_feedback(line_id="heat_treatment", days=365)
        assert len(ht) == 1
        assert ht[0]["feedback_id"] == "fb1"

        wd = memory.query_feedback(line_id="welding", days=365)
        assert len(wd) == 1
        assert wd[0]["feedback_id"] == "fb2"

        all_fb = memory.query_feedback(days=365)
        assert len(all_fb) == 2

    def test_semantic_isolation_by_line_id(self, memory_with_chroma):
        """长期记忆（Chroma）应按 line_id 隔离检索"""
        memory = memory_with_chroma
        # 写入两个不同产线的案例，缺陷类型相同
        case_ht = _make_case(
            case_id="C_HT",
            root_cause="保温时间不足",
            line_id="heat_treatment",
        )
        case_wd = _make_case(
            case_id="C_WD",
            root_cause="保温时间不足",
            line_id="welding",
        )
        memory.write_semantic(case_ht)
        memory.write_semantic(case_wd)

        # 按 heat_treatment 检索，只命中 C_HT
        ht_hits = memory.search_semantic("保温时间不足", top_k=10, line_id="heat_treatment")
        ht_ids = [h["id"] for h in ht_hits]
        assert "C_HT" in ht_ids
        assert "C_WD" not in ht_ids, "welding 案例不应出现在 heat_treatment 检索结果中"

        # 按 welding 检索，只命中 C_WD
        wd_hits = memory.search_semantic("保温时间不足", top_k=10, line_id="welding")
        wd_ids = [h["id"] for h in wd_hits]
        assert "C_WD" in wd_ids
        assert "C_HT" not in wd_ids

        # 不传 line_id 检索全部
        all_hits = memory.search_semantic("保温时间不足", top_k=10)
        all_ids = [h["id"] for h in all_hits]
        assert "C_HT" in all_ids
        assert "C_WD" in all_ids

    def test_semantic_metadata_contains_line_id(self, memory_with_chroma):
        """write_semantic 写入的 Chroma metadata 应包含 line_id 字段"""
        memory = memory_with_chroma
        case = _make_case(case_id="C_META", line_id="welding")
        memory.write_semantic(case)

        results = memory.search_semantic("保温时间不足", top_k=5, line_id="welding")
        assert len(results) > 0
        assert results[0]["metadata"]["line_id"] == "welding"

    def test_detect_conflicts_isolation_by_line_id(self, memory_with_chroma):
        """冲突检测应按 line_id 隔离：不同产线的相似案例不互相冲突"""
        memory = memory_with_chroma
        # 先写入 heat_treatment 案例
        case_ht = _make_case(
            case_id="C_HT_EXIST",
            root_cause="保温时间不足",
            solution="holding_time +15",
            confidence=0.8,
            line_id="heat_treatment",
        )
        memory.write_semantic(case_ht)

        # 再写入 welding 案例，根因相同但应不与 heat_treatment 冲突
        case_wd = _make_case(
            case_id="C_WD_NEW",
            root_cause="保温时间不足",
            solution="holding_time +30",  # 不同方案，理论上会触发软冲突
            confidence=0.8,
            line_id="welding",
        )
        # detect_conflicts 内部会按 case.line_id 过滤检索
        conflicts = memory.detect_conflicts(case_wd)
        # 因为 welding 产线没有其他案例，所以不应检测到冲突
        assert len(conflicts) == 0, "不同产线案例不应触发冲突"

    def test_detect_conflicts_same_line_triggers_conflict(self, memory_with_chroma):
        """同产线内相似案例应触发冲突（验证隔离逻辑没误伤同产线检测）"""
        memory = memory_with_chroma
        case1 = _make_case(
            case_id="C_HT_1",
            root_cause="保温时间不足",
            solution="holding_time +15",
            confidence=0.8,
            line_id="heat_treatment",
        )
        memory.write_semantic(case1)

        # 同产线、相似参数、不同方案 → 应触发软冲突
        case2 = _make_case(
            case_id="C_HT_2",
            root_cause="保温时间不足",
            solution="holding_time +30",
            confidence=0.8,
            line_id="heat_treatment",
        )
        conflicts = memory.detect_conflicts(case2)
        assert len(conflicts) > 0, "同产线相似案例应触发冲突"
        # 验证冲突记录的 line_id 字段
        for c in conflicts:
            assert c.line_id == "heat_treatment"

    def test_save_and_list_conflicts_isolation(self, memory):
        """冲突记录的保存与查询应按 line_id 隔离"""
        from agent.memory.memory_service import ConflictRecord

        now = datetime.now()
        cf_ht = ConflictRecord(
            conflict_id="cf_ht_1",
            new_case_id="C_HT_NEW",
            existing_case_id="C_HT_OLD",
            conflict_type="soft",
            description="热处理产线冲突",
            created_at=now,
            line_id="heat_treatment",
        )
        cf_wd = ConflictRecord(
            conflict_id="cf_wd_1",
            new_case_id="C_WD_NEW",
            existing_case_id="C_WD_OLD",
            conflict_type="hard",
            description="焊接产线冲突",
            created_at=now,
            line_id="welding",
        )
        memory.save_conflict(cf_ht)
        memory.save_conflict(cf_wd)

        ht = memory.list_conflicts(line_id="heat_treatment")
        assert len(ht) == 1
        assert ht[0]["conflict_id"] == "cf_ht_1"
        assert ht[0]["line_id"] == "heat_treatment"

        wd = memory.list_conflicts(line_id="welding")
        assert len(wd) == 1
        assert wd[0]["conflict_id"] == "cf_wd_1"

        all_cf = memory.list_conflicts()
        assert len(all_cf) == 2

    def test_search_cases_tool_isolation(self, memory_with_chroma):
        """search_cases 工具函数应按 line_id 隔离"""
        _set_memory_service(memory_with_chroma)
        memory = memory_with_chroma

        case_ht = _make_case(
            case_id="C_TOOL_HT",
            root_cause="保温时间不足",
            line_id="heat_treatment",
        )
        case_wd = _make_case(
            case_id="C_TOOL_WD",
            root_cause="保温时间不足",
            line_id="welding",
        )
        memory.write_semantic(case_ht)
        memory.write_semantic(case_wd)

        # 按 welding 检索
        result_wd = search_cases("保温时间不足", top_k=10, line_id="welding")
        ids_wd = [r["record_id"] for r in result_wd.get("results", [])]
        assert "C_TOOL_WD" in ids_wd
        assert "C_TOOL_HT" not in ids_wd

        # 按 heat_treatment 检索
        result_ht = search_cases("保温时间不足", top_k=10, line_id="heat_treatment")
        ids_ht = [r["record_id"] for r in result_ht.get("results", [])]
        assert "C_TOOL_HT" in ids_ht
        assert "C_TOOL_WD" not in ids_ht

    def test_memory_writer_passes_line_id(self, monkeypatch):
        """memory_writer 应从 state 读取 line_id 并传给 write_episodic / CaseRecord"""
        from agent.nodes import memory_writer as mw_func

        class _LineCaptureMemory:
            """捕获 write_episodic 和 write_semantic 调用参数的 fake"""
            def __init__(self):
                self.episodic_kwargs = None
                self.semantic_case = None

            def write_episodic(self, **kwargs):
                self.episodic_kwargs = kwargs

            def write_semantic(self, case):
                self.semantic_case = case
                return True

        fake = _LineCaptureMemory()
        monkeypatch.setitem(mw_func.__globals__, "_memory_service", fake)

        state = {
            "trace_id": "tr_line",
            "batch_id": "B_LINE",
            "line_id": "welding",  # M4-9: 产线ID
            "decision_result": {"proposals": [{"root_cause": "保温时间不足", "adjustments": {"holding_time": "+15"}}]},
            "data_result": {"batch_params": {"holding_time": 90}},
            "final_answer": None,
        }
        asyncio.run(mw_func(state))

        # 验证 write_episodic 收到了 line_id
        assert fake.episodic_kwargs is not None
        assert fake.episodic_kwargs.get("line_id") == "welding"

        # 验证 CaseRecord 的 line_id 字段
        assert fake.semantic_case is not None
        assert fake.semantic_case.line_id == "welding"

    def test_memory_writer_default_line_id(self, monkeypatch):
        """memory_writer 在 state 无 line_id 时应回退到默认值 heat_treatment"""
        from agent.nodes import memory_writer as mw_func

        class _LineCaptureMemory:
            def __init__(self):
                self.episodic_kwargs = None
                self.semantic_case = None

            def write_episodic(self, **kwargs):
                self.episodic_kwargs = kwargs

            def write_semantic(self, case):
                self.semantic_case = case
                return True

        fake = _LineCaptureMemory()
        monkeypatch.setitem(mw_func.__globals__, "_memory_service", fake)

        state = {
            "trace_id": "tr_default",
            "batch_id": "B_DEFAULT",
            # 不传 line_id，应回退到默认值
            "decision_result": {"proposals": [{"root_cause": "保温时间不足", "adjustments": {"holding_time": "+15"}}]},
            "data_result": {"batch_params": {"holding_time": 90}},
            "final_answer": None,
        }
        asyncio.run(mw_func(state))

        assert fake.episodic_kwargs.get("line_id") == "heat_treatment"
        assert fake.semantic_case.line_id == "heat_treatment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
