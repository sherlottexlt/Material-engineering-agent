"""
M5-7 知识主动学习测试

测试三层：
1. ActiveLearner 候选识别 + 问题生成 + 规则提炼 + 手册写入 + 提交回答闭环
2. MemoryService 数据层方法（save/list/get/update/defect_frequency/learning_stats）
3. API 端点（/api/v1/learning/* 共 6 个）
"""
import json
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


# ===== Fixtures =====


@pytest.fixture
def memory(tmp_path):
    """临时 MemoryService（无 Chroma）"""
    from agent.memory.memory_service import MemoryService
    db_path = tmp_path / "test_active_learning.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def handbook_index(tmp_path):
    """临时 handbook_index.json（空索引，避免污染真实索引）"""
    index_path = tmp_path / "handbook_index.json"
    index = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "source_files": [],
        "total_chunks": 0,
        "chunks": [],
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return index_path


@pytest.fixture
def learner(memory, handbook_index):
    """ActiveLearner 实例（用临时 handbook 索引）"""
    from agent.active_learner import ActiveLearner
    return ActiveLearner(memory=memory, handbook_index_path=handbook_index)


@pytest.fixture
def client(memory, handbook_index):
    """FastAPI 测试客户端（注入临时 memory + learner + handbook 索引）"""
    from fastapi.testclient import TestClient
    from api.routes import app, active_learner
    import api.routes as routes_module

    orig_learner_memory = active_learner.memory
    orig_learner_db = active_learner.db
    orig_learner_handbook = active_learner.handbook_index_path
    orig_routes_memory = routes_module.memory

    active_learner.memory = memory
    active_learner.db = memory.db
    active_learner.handbook_index_path = handbook_index
    routes_module.memory = memory

    yield TestClient(app)

    active_learner.memory = orig_learner_memory
    active_learner.db = orig_learner_db
    active_learner.handbook_index_path = orig_learner_handbook
    routes_module.memory = orig_routes_memory


# ===== 辅助函数 =====


def _seed_episodic(
    memory,
    record_id="rec_001",
    batch_id="B001",
    defect_type="hardness_low",
    root_cause="淬火温度不足导致奥氏体化不充分",
    solution="将淬火温度提升至 860℃ 并保温 90 分钟",
    line_id="heat_treatment",
    quality_score=0.5,
    days_ago=0,
):
    """注入一条 episodic 记录"""
    created_at = datetime.now() - timedelta(days=days_ago)
    memory.db.execute(
        "INSERT INTO episodic VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (record_id, batch_id, defect_type, root_cause, solution,
         created_at, quality_score, line_id),
    )
    memory.db.commit()
    return record_id


def _seed_failure_case(memory, case_id="case_fail_1", line_id="heat_treatment",
                       category="low_confidence", status="open"):
    """注入一条 failure_cases 记录"""
    from agent.failure_case_collector import FailureCaseCollector
    collector = FailureCaseCollector(memory)
    collector._save_failure(
        case_id=case_id,
        tracking_id=None,
        line_id=line_id,
        category=category,
        confidence=0.1 if category == "low_confidence" else None,
        improvement_pct=None,
        failure_reason="confidence=0.1 < 0.3",
    )
    # 可选：更新 status
    if status != "open":
        memory.db.execute(
            "UPDATE failure_cases SET status = ? WHERE case_id = ?",
            (status, case_id),
        )
        memory.db.commit()


# ===== 1. 候选识别（TestIdentifyCandidates）=====


class TestIdentifyCandidates:
    """identify_candidates 候选识别测试"""

    def test_empty_database(self, learner):
        """空库识别应返回 0 候选"""
        result = learner.identify_candidates()
        assert result["total"] == 0
        assert result["by_source"] == {"low_quality": 0, "failure_case": 0, "high_frequency": 0}
        assert result["skipped_duplicate"] == 0
        assert result["candidate_ids"] == []

    def test_low_quality_source(self, learner, memory):
        """低质量案例应被识别为候选"""
        _seed_episodic(memory, record_id="rec_low", quality_score=0.2)
        result = learner.identify_candidates(low_quality_threshold=0.4)

        assert result["total"] >= 1
        assert result["by_source"]["low_quality"] >= 1
        assert len(result["candidate_ids"]) >= 1

    def test_failure_case_source(self, learner, memory):
        """失败案例应被识别为候选"""
        _seed_failure_case(memory, case_id="case_fail_1", status="open")
        result = learner.identify_candidates(days=30)

        assert result["total"] >= 1
        assert result["by_source"]["failure_case"] >= 1

    def test_high_frequency_source(self, learner, memory):
        """高频缺陷应被识别为候选（>= 3 次）"""
        # 注入 4 条同 defect_type 的案例
        for i in range(4):
            _seed_episodic(memory, record_id=f"rec_hf_{i}", defect_type="hardness_low",
                           quality_score=0.8)
        result = learner.identify_candidates(days=365, frequency_min_count=3)

        assert result["by_source"]["high_frequency"] >= 1

    def test_idempotent(self, learner, memory):
        """重复识别应跳过已存在的候选"""
        _seed_episodic(memory, record_id="rec_idem", quality_score=0.2)

        # 第一次识别
        r1 = learner.identify_candidates(low_quality_threshold=0.4)
        assert r1["total"] >= 1

        # 第二次识别应跳过
        r2 = learner.identify_candidates(low_quality_threshold=0.4)
        assert r2["total"] == 0
        assert r2["skipped_duplicate"] >= 1

    def test_line_id_filter(self, learner, memory):
        """line_id 过滤应只识别指定产线"""
        _seed_episodic(memory, record_id="rec_ht", line_id="heat_treatment", quality_score=0.2)
        _seed_episodic(memory, record_id="rec_wd", line_id="welding", quality_score=0.2)
        result = learner.identify_candidates(line_id="heat_treatment", low_quality_threshold=0.4)

        assert result["total"] == 1
        # 验证候选的 source_id 是 rec_ht
        candidates = learner.list_candidates()
        assert candidates[0]["source_id"] == "rec_ht"


# ===== 2. 问题生成（TestGenerateQuestion）=====


class TestGenerateQuestion:
    """generate_question 问题生成测试"""

    def test_low_quality_question(self, learner):
        """低质量案例问题应包含案例ID和缺陷类型"""
        question = learner.generate_question({
            "source_type": "low_quality",
            "source_id": "rec_001",
            "defect_type": "hardness_low",
            "record": {"root_cause": "", "solution": ""},
        })
        assert "rec_001" in question
        assert "hardness_low" in question
        assert "低质量案例学习" in question

    def test_failure_case_question(self, learner):
        """失败案例问题应包含失败类别和原因"""
        question = learner.generate_question({
            "source_type": "failure_case",
            "source_id": "fail_001",
            "defect_type": "low_confidence",
            "failure": {
                "category": "low_confidence",
                "failure_reason": "confidence=0.1 < 0.3",
                "confidence": 0.1,
            },
        })
        assert "fail_001" in question
        assert "low_confidence" in question
        assert "失败案例学习" in question
        assert "0.1" in question  # confidence 值

    def test_high_frequency_question(self, learner):
        """高频缺陷问题应包含次数和天数"""
        question = learner.generate_question({
            "source_type": "high_frequency",
            "source_id": "hardness_low",
            "defect_type": "hardness_low",
            "frequency": {"count": 5, "defect_type": "hardness_low"},
            "days": 30,
        })
        assert "hardness_low" in question
        assert "5" in question  # 次数
        assert "30" in question  # 天数
        assert "高频缺陷学习" in question


# ===== 3. 规则提炼 + 手册写入（TestExtractRuleAndAddToHandbook）=====


class TestExtractRuleAndAddToHandbook:
    """extract_rule + add_rule_to_handbook 测试"""

    def test_extract_rule_format(self, learner):
        """提炼的规则应包含标题、来源、问题、回答"""
        candidate = {
            "candidate_id": "lc_test",
            "source_type": "low_quality",
            "defect_type": "hardness_low",
            "line_id": "heat_treatment",
            "question": "案例 rec_001 信息不完整...",
        }
        answer = "根因是淬火温度不足，建议提升至 860℃"
        rule = learner.extract_rule(candidate, answer)

        assert "## 主动学习规则 - hardness_low" in rule
        assert "低质量案例补全" in rule
        assert "heat_treatment" in rule
        assert "案例 rec_001 信息不完整" in rule
        assert "根因是淬火温度不足" in rule

    def test_add_rule_to_handbook(self, learner, handbook_index):
        """规则应写入 handbook_index.json"""
        rule_text = "## 主动学习规则 - hardness_low\n\n测试规则内容"
        result = learner.add_rule_to_handbook(rule_text, "hardness_low")

        assert result["success"] is True
        assert result["chunk_id"].startswith("al_")
        assert result["total_chunks"] == 1

        # 验证文件已写入
        with open(handbook_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        assert len(index["chunks"]) == 1
        assert index["chunks"][0]["content"] == rule_text
        assert index["total_chunks"] == 1

    def test_add_multiple_rules(self, learner, handbook_index):
        """多次写入手册应追加 chunk"""
        for i in range(3):
            result = learner.add_rule_to_handbook(f"规则 {i}", f"defect_{i}")
            assert result["success"] is True

        with open(handbook_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        assert len(index["chunks"]) == 3
        assert index["total_chunks"] == 3
        # chunk_id 应递增
        chunk_ids = [c["chunk_id"] for c in index["chunks"]]
        assert len(set(chunk_ids)) == 3  # 唯一

    def test_add_rule_to_nonexistent_path(self, learner, tmp_path):
        """handbook 索引不存在时应创建新文件"""
        from agent.active_learner import ActiveLearner
        new_path = tmp_path / "new_dir" / "handbook.json"
        learner2 = ActiveLearner(memory=learner.memory, handbook_index_path=new_path)

        result = learner2.add_rule_to_handbook("测试规则", "test_defect")
        assert result["success"] is True
        assert new_path.exists()


# ===== 4. 提交回答闭环（TestSubmitAnswer）=====


class TestSubmitAnswer:
    """submit_answer 提交回答闭环测试"""

    def test_submit_with_auto_extract(self, learner, memory, handbook_index):
        """提交回答 + auto_extract 应提炼规则并写入手册"""
        _seed_episodic(memory, record_id="rec_answer", quality_score=0.2)
        # 识别候选
        learner.identify_candidates(low_quality_threshold=0.4)
        candidates = learner.list_candidates(status="pending")
        assert len(candidates) >= 1
        cid = candidates[0]["candidate_id"]

        # 提交回答
        result = learner.submit_answer(
            candidate_id=cid,
            answer="根因是淬火温度不足 810℃ 低于标准 840℃，建议提升至 860℃",
            auto_extract=True,
        )

        assert result["success"] is True
        assert result["status"] == "learned"
        assert result["rule_extracted"] is True
        assert result["handbook_updated"] is True
        assert result["chunk_id"].startswith("al_")

        # 验证候选状态已更新
        candidate = learner.get_candidate(cid)
        assert candidate["status"] == "learned"
        assert candidate["answer"] is not None
        assert candidate["rule_text"] is not None
        assert candidate["rule_added_to_handbook"] == 1

    def test_submit_without_auto_extract(self, learner, memory):
        """auto_extract=False 仅保存回答不提炼规则"""
        _seed_episodic(memory, record_id="rec_no_auto", quality_score=0.2)
        learner.identify_candidates(low_quality_threshold=0.4)
        cid = learner.list_candidates(status="pending")[0]["candidate_id"]

        result = learner.submit_answer(
            candidate_id=cid,
            answer="测试回答",
            auto_extract=False,
        )

        assert result["success"] is True
        assert result["status"] == "answered"
        assert result["rule_extracted"] is False
        assert result["handbook_updated"] is False

        candidate = learner.get_candidate(cid)
        assert candidate["status"] == "answered"
        assert candidate["rule_text"] is None

    def test_submit_nonexistent_candidate(self, learner):
        """提交到不存在的候选应失败"""
        result = learner.submit_answer("nonexistent", "answer")
        assert result["success"] is False
        assert result["status"] == "not_found"

    def test_submit_to_already_learned(self, learner, memory):
        """已 learned 的候选不能再次提交"""
        _seed_episodic(memory, record_id="rec_learned", quality_score=0.2)
        learner.identify_candidates(low_quality_threshold=0.4)
        cid = learner.list_candidates(status="pending")[0]["candidate_id"]

        # 第一次提交
        learner.submit_answer(cid, "回答", auto_extract=True)
        # 第二次提交应失败
        result = learner.submit_answer(cid, "再次回答", auto_extract=True)
        assert result["success"] is False


# ===== 5. 查询代理（TestQueryProxy）=====


class TestQueryProxy:
    """list/get/skip/stats 查询代理测试"""

    def test_list_candidates_empty(self, learner):
        """空库列表返回空"""
        assert learner.list_candidates() == []

    def test_list_candidates_filter(self, learner, memory):
        """列表支持 status 过滤"""
        _seed_episodic(memory, record_id="rec_filter", quality_score=0.2)
        learner.identify_candidates(low_quality_threshold=0.4)

        pending = learner.list_candidates(status="pending")
        assert len(pending) >= 1
        assert all(c["status"] == "pending" for c in pending)

        learned = learner.list_candidates(status="learned")
        assert len(learned) == 0

    def test_get_candidate(self, learner, memory):
        """查询单条候选"""
        _seed_episodic(memory, record_id="rec_get", quality_score=0.2)
        learner.identify_candidates(low_quality_threshold=0.4)
        cid = learner.list_candidates()[0]["candidate_id"]

        candidate = learner.get_candidate(cid)
        assert candidate is not None
        assert candidate["candidate_id"] == cid

    def test_get_candidate_nonexistent(self, learner):
        """查询不存在的候选返回 None"""
        assert learner.get_candidate("nonexistent") is None

    def test_skip_candidate(self, learner, memory):
        """跳过候选"""
        _seed_episodic(memory, record_id="rec_skip", quality_score=0.2)
        learner.identify_candidates(low_quality_threshold=0.4)
        cid = learner.list_candidates(status="pending")[0]["candidate_id"]

        result = learner.skip_candidate(cid)
        assert result["success"] is True
        assert result["status"] == "skipped"

        candidate = learner.get_candidate(cid)
        assert candidate["status"] == "skipped"

    def test_skip_nonexistent(self, learner):
        """跳过不存在的候选应失败"""
        result = learner.skip_candidate("nonexistent")
        assert result["success"] is False

    def test_skip_already_answered(self, learner, memory):
        """已 answered 的候选不能跳过"""
        _seed_episodic(memory, record_id="rec_skip_answered", quality_score=0.2)
        learner.identify_candidates(low_quality_threshold=0.4)
        cid = learner.list_candidates(status="pending")[0]["candidate_id"]

        # 先提交回答
        learner.submit_answer(cid, "回答", auto_extract=False)
        # 再尝试跳过应失败
        result = learner.skip_candidate(cid)
        assert result["success"] is False

    def test_learning_stats(self, learner, memory):
        """学习统计"""
        _seed_episodic(memory, record_id="rec_stats", quality_score=0.2)
        learner.identify_candidates(low_quality_threshold=0.4)

        stats = learner.get_learning_stats(days=365)
        assert stats["total"] >= 1
        assert stats["by_status"]["pending"] >= 1
        assert stats["by_source"]["low_quality"] >= 1
        assert stats["rules_extracted"] == 0
        assert stats["rules_added_to_handbook"] == 0


# ===== 6. 数据层方法（TestMemoryLayerMethods）=====


class TestMemoryLayerMethods:
    """MemoryService M5-7 新增方法测试"""

    def test_save_and_get_candidate(self, memory):
        """保存并查询候选"""
        ok = memory.save_learning_candidate(
            candidate_id="lc_test_1",
            source_type="low_quality",
            source_id="rec_001",
            line_id="heat_treatment",
            defect_type="hardness_low",
            question="测试问题",
        )
        assert ok is True

        candidate = memory.get_learning_candidate("lc_test_1")
        assert candidate is not None
        assert candidate["source_type"] == "low_quality"
        assert candidate["question"] == "测试问题"
        assert candidate["status"] == "pending"

    def test_get_nonexistent_candidate(self, memory):
        """查询不存在的候选返回 None"""
        assert memory.get_learning_candidate("nonexistent") is None

    def test_list_candidates_with_filters(self, memory):
        """列表支持多条件过滤"""
        memory.save_learning_candidate(
            candidate_id="lc_1", source_type="low_quality", source_id="s1",
            line_id="heat_treatment", defect_type="d1", question="q1"
        )
        memory.save_learning_candidate(
            candidate_id="lc_2", source_type="failure_case", source_id="s2",
            line_id="welding", defect_type="d2", question="q2"
        )

        # status 过滤
        all_pending = memory.list_learning_candidates(status="pending")
        assert len(all_pending) == 2

        # source_type 过滤
        low_quality_only = memory.list_learning_candidates(source_type="low_quality")
        assert len(low_quality_only) == 1
        assert low_quality_only[0]["candidate_id"] == "lc_1"

        # line_id 过滤
        ht_only = memory.list_learning_candidates(line_id="heat_treatment")
        assert len(ht_only) == 1
        assert ht_only[0]["candidate_id"] == "lc_1"

    def test_update_candidate(self, memory):
        """更新候选字段"""
        memory.save_learning_candidate(
            candidate_id="lc_upd", source_type="low_quality", source_id="s1",
            line_id="heat_treatment", defect_type="d1", question="q1"
        )

        ok = memory.update_learning_candidate(
            candidate_id="lc_upd",
            status="answered",
            answer="专家回答",
        )
        assert ok is True

        candidate = memory.get_learning_candidate("lc_upd")
        assert candidate["status"] == "answered"
        assert candidate["answer"] == "专家回答"
        assert candidate["answered_at"] is not None

    def test_update_candidate_learned(self, memory):
        """更新为 learned 应记录 learned_at"""
        memory.save_learning_candidate(
            candidate_id="lc_learned", source_type="low_quality", source_id="s1",
            line_id="heat_treatment", defect_type="d1", question="q1"
        )
        memory.update_learning_candidate(
            candidate_id="lc_learned",
            status="learned",
            rule_text="规则文本",
            rule_added_to_handbook=1,
        )

        candidate = memory.get_learning_candidate("lc_learned")
        assert candidate["status"] == "learned"
        assert candidate["rule_text"] == "规则文本"
        assert candidate["rule_added_to_handbook"] == 1
        assert candidate["learned_at"] is not None

    def test_defect_frequency(self, memory):
        """缺陷频次统计"""
        _seed_episodic(memory, record_id="rec_f1", defect_type="hardness_low", days_ago=0)
        _seed_episodic(memory, record_id="rec_f2", defect_type="hardness_low", days_ago=0)
        _seed_episodic(memory, record_id="rec_f3", defect_type="hardness_low", days_ago=0)
        _seed_episodic(memory, record_id="rec_f4", defect_type="crack", days_ago=0)

        freqs = memory.get_defect_frequency(days=365, top_n=5)
        assert len(freqs) >= 2
        # hardness_low 出现 3 次，应排第一
        assert freqs[0]["defect_type"] == "hardness_low"
        assert freqs[0]["count"] == 3

    def test_defect_frequency_line_filter(self, memory):
        """频次统计支持 line_id 过滤"""
        _seed_episodic(memory, record_id="rec_ht", defect_type="hardness_low",
                       line_id="heat_treatment", days_ago=0)
        _seed_episodic(memory, record_id="rec_wd", defect_type="porosity",
                       line_id="welding", days_ago=0)

        ht_freqs = memory.get_defect_frequency(line_id="heat_treatment", days=365)
        assert len(ht_freqs) == 1
        assert ht_freqs[0]["defect_type"] == "hardness_low"

    def test_learning_stats_empty(self, memory):
        """空库学习统计"""
        stats = memory.get_learning_stats(days=365)
        assert stats["total"] == 0
        assert stats["by_status"] == {"pending": 0, "answered": 0, "learned": 0, "skipped": 0}
        assert stats["rules_extracted"] == 0


# ===== 7. API 端点（TestLearningAPI）=====


class TestLearningAPI:
    """M5-7 API 端点测试"""

    def test_identify_endpoint_admin(self, client, memory):
        """admin 触发识别应成功"""
        _seed_episodic(memory, record_id="rec_api_id", quality_score=0.2)
        resp = client.post("/api/v1/learning/identify?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total"] >= 1

    def test_identify_endpoint_operator_forbidden(self, client):
        """operator 触发识别应 403"""
        resp = client.post("/api/v1/learning/identify?user_id=operator_01")
        assert resp.status_code == 403

    def test_candidates_endpoint_admin_all(self, client, memory):
        """admin 列出全部候选"""
        _seed_episodic(memory, record_id="rec_list", quality_score=0.2)
        client.post("/api/v1/learning/identify?user_id=admin")

        resp = client.get("/api/v1/learning/candidates?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] >= 1

    def test_candidates_endpoint_status_filter(self, client, memory):
        """candidates 支持 status 过滤"""
        _seed_episodic(memory, record_id="rec_status", quality_score=0.2)
        client.post("/api/v1/learning/identify?user_id=admin")

        resp = client.get("/api/v1/learning/candidates?user_id=admin&status=pending")
        data = resp.json()
        assert all(c["status"] == "pending" for c in data["candidates"])

    def test_candidate_detail_endpoint(self, client, memory):
        """查询单条候选详情"""
        _seed_episodic(memory, record_id="rec_detail", quality_score=0.2)
        client.post("/api/v1/learning/identify?user_id=admin")
        cid = client.get("/api/v1/learning/candidates?user_id=admin").json()["candidates"][0]["candidate_id"]

        resp = client.get(f"/api/v1/learning/candidates/{cid}?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["candidate"]["candidate_id"] == cid

    def test_candidate_detail_not_found(self, client):
        """不存在的候选应 404"""
        resp = client.get("/api/v1/learning/candidates/nonexistent?user_id=admin")
        assert resp.status_code == 404

    def test_answer_endpoint_with_extract(self, client, memory, handbook_index):
        """提交回答 + auto_extract 应写入手册"""
        _seed_episodic(memory, record_id="rec_answer_api", quality_score=0.2)
        client.post("/api/v1/learning/identify?user_id=admin")
        cid = client.get("/api/v1/learning/candidates?user_id=admin").json()["candidates"][0]["candidate_id"]

        resp = client.post(
            f"/api/v1/learning/answer/{cid}",
            json={"answer": "根因是温度不足，建议提升至 860℃", "auto_extract": True, "user_id": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "learned"
        assert data["handbook_updated"] is True
        assert data["chunk_id"].startswith("al_")

        # 验证手册已写入
        with open(handbook_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        assert len(index["chunks"]) >= 1

    def test_answer_endpoint_operator_forbidden(self, client, memory):
        """operator 提交回答应 403"""
        _seed_episodic(memory, record_id="rec_forbidden", quality_score=0.2)
        client.post("/api/v1/learning/identify?user_id=admin")
        cid = client.get("/api/v1/learning/candidates?user_id=admin").json()["candidates"][0]["candidate_id"]

        resp = client.post(
            f"/api/v1/learning/answer/{cid}",
            json={"answer": "测试", "auto_extract": False, "user_id": "operator_01"},
        )
        assert resp.status_code == 403

    def test_skip_endpoint(self, client, memory):
        """跳过候选"""
        _seed_episodic(memory, record_id="rec_skip_api", quality_score=0.2)
        client.post("/api/v1/learning/identify?user_id=admin")
        cid = client.get("/api/v1/learning/candidates?user_id=admin").json()["candidates"][0]["candidate_id"]

        resp = client.post(f"/api/v1/learning/skip/{cid}?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "skipped"

    def test_stats_endpoint(self, client, memory):
        """学习统计端点"""
        _seed_episodic(memory, record_id="rec_stats_api", quality_score=0.2)
        client.post("/api/v1/learning/identify?user_id=admin")

        resp = client.get("/api/v1/learning/stats?user_id=admin&days=365")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["stats"]["total"] >= 1
        assert data["stats"]["by_status"]["pending"] >= 1
