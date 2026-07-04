"""
M5-6 案例质量评分测试

测试三层：
1. CaseQualityScorer 单案例评分（4 维子分） + 批量评分
2. MemoryService 数据层方法（update_quality_score / get_quality_distribution / get_low_quality_cases）
3. API 端点（/api/v1/cases/quality/* 共 4 个）
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


# ===== Fixtures =====


@pytest.fixture
def memory(tmp_path):
    """临时 MemoryService（无 Chroma）"""
    from agent.memory.memory_service import MemoryService
    db_path = tmp_path / "test_quality.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def scorer(memory):
    """CaseQualityScorer 实例"""
    from agent.case_quality_scorer import CaseQualityScorer
    return CaseQualityScorer(memory)


@pytest.fixture
def client(memory):
    """FastAPI 测试客户端（注入临时 memory + scorer）"""
    from fastapi.testclient import TestClient
    from api.routes import app, quality_scorer
    orig_scorer_memory = quality_scorer.memory
    orig_scorer_db = quality_scorer.db

    quality_scorer.memory = memory
    quality_scorer.db = memory.db

    # 同时让 routes.memory 指向临时 memory（用于单案例详情端点的直接 db 查询）
    from api.routes import memory as routes_memory
    import api.routes as routes_module
    orig_routes_memory = routes_memory
    routes_module.memory = memory

    yield TestClient(app)

    quality_scorer.memory = orig_scorer_memory
    quality_scorer.db = orig_scorer_db
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
    days_ago=0,
):
    """注入一条 episodic 记录"""
    created_at = datetime.now() - timedelta(days=days_ago)
    memory.db.execute(
        "INSERT INTO episodic VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (record_id, batch_id, defect_type, root_cause, solution,
         created_at, 0.5, line_id),
    )
    memory.db.commit()
    return record_id


def _seed_failure_case(memory, case_id, line_id="heat_treatment"):
    """注入一条 failure_cases 记录（让验证状态降分）"""
    from agent.failure_case_collector import FailureCaseCollector
    collector = FailureCaseCollector(memory)
    collector._save_failure(
        case_id=case_id,
        tracking_id=None,
        line_id=line_id,
        category="low_confidence",
        confidence=0.1,
        improvement_pct=None,
        failure_reason="confidence=0.1 < 0.3",
    )


# ===== 1. 单案例评分（TestScoreCase）=====


class TestScoreCase:
    """score_case 单案例评分测试"""

    def test_full_quality_case(self, scorer):
        """完整案例应得高分（4 维都满）"""
        record = {
            "record_id": "rec_full",
            "batch_id": "B001",
            "defect_type": "hardness_low",
            "root_cause": "淬火温度不足导致奥氏体化不充分" * 2,  # > 20 字符
            "solution": "将淬火温度提升至 860℃ 并保温 90 分钟" * 2,  # > 20 + 含可操作关键词
            "created_at": datetime.now(),
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)

        assert result["record_id"] == "rec_full"
        assert result["old_score"] == 0.5
        assert 0.7 <= result["new_score"] <= 1.0  # 高分
        assert result["dimensions"]["completeness"] == 1.0
        assert result["dimensions"]["reusability"] == 1.0  # defect_type 标准 + solution 可操作
        assert result["dimensions"]["timeliness"] >= 0.99  # 刚创建
        assert result["dimensions"]["validation"] == 0.5  # 无失败记录
        assert len(result["reasons"]) > 0

    def test_empty_fields_case(self, scorer):
        """全空字段案例应得低分"""
        record = {
            "record_id": "rec_empty",
            "batch_id": "B002",
            "defect_type": "",
            "root_cause": "",
            "solution": "",
            "created_at": datetime.now(),
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)

        assert result["new_score"] < 0.3  # 低分
        assert result["dimensions"]["completeness"] == 0.0
        assert result["dimensions"]["reusability"] == 0.0
        assert result["dimensions"]["validation"] == 0.5  # 无失败记录仍有基线
        assert "root_cause 为空" in result["reasons"]
        assert "solution 为空" in result["reasons"]
        assert "defect_type 为空" in result["reasons"]

    def test_short_root_cause(self, scorer):
        """root_cause 偏短（< 20 字符）应扣完整度分"""
        record = {
            "record_id": "rec_short",
            "batch_id": "B003",
            "defect_type": "hardness_low",
            "root_cause": "温度低",  # 3 字符 < 20
            "solution": "将淬火温度提升至 860℃ 并保温 90 分钟" * 2,
            "created_at": datetime.now(),
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)
        # root_cause 非空 +0.5, 长度不达标 +0
        # solution 非空 +0.5 + 长度达标 +0.3 = 0.8
        # 总分 1.3 / 1.6 = 0.8125
        assert result["dimensions"]["completeness"] == 0.8125
        assert any("偏短" in r for r in result["reasons"])

    def test_nonstandard_defect_type(self, scorer):
        """非标准 defect_type 应扣可复用性分"""
        record = {
            "record_id": "rec_nonstd",
            "batch_id": "B004",
            "defect_type": "weird_unknown_defect_xyz",  # 不含标准关键词
            "root_cause": "淬火温度不足导致奥氏体化不充分" * 2,
            "solution": "将淬火温度提升至 860℃" * 2,
            "created_at": datetime.now(),
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)
        # defect_type 非空 +0.4, 非标准 +0
        # solution 含可操作关键词（"提升"/"温度"） +0.3
        # 总 0.7
        assert result["dimensions"]["reusability"] == 0.7
        assert any("非标准" in r for r in result["reasons"])

    def test_solution_without_actionable_keyword(self, scorer):
        """solution 缺可操作关键词应扣可复用性分"""
        record = {
            "record_id": "rec_noaction",
            "batch_id": "B005",
            "defect_type": "hardness_low",
            "root_cause": "淬火温度不足导致奥氏体化不充分" * 2,
            "solution": "这是一个没有任何可执行动作的描述性文字" * 2,  # 无关键词
            "created_at": datetime.now(),
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)
        # defect_type 标准 +0.4+0.3, solution 无可操作关键词 +0
        # 总 0.7
        assert result["dimensions"]["reusability"] == 0.7
        assert any("缺可操作关键词" in r for r in result["reasons"])

    def test_old_case_timeliness(self, scorer):
        """旧案例时效性应接近 0"""
        record = {
            "record_id": "rec_old",
            "batch_id": "B006",
            "defect_type": "hardness_low",
            "root_cause": "淬火温度不足" * 5,
            "solution": "提升温度" * 5,
            "created_at": datetime.now() - timedelta(days=400),  # > 1 年
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)
        assert result["dimensions"]["timeliness"] == 0.0  # > 365 天衰减到 0
        assert any("年龄 400" in r for r in result["reasons"])

    def test_case_with_failure_record(self, scorer, memory):
        """有失败记录的案例验证状态应为 0"""
        record_id = _seed_episodic(memory, record_id="rec_fail")
        _seed_failure_case(memory, case_id=record_id)

        record = {
            "record_id": record_id,
            "batch_id": "B001",
            "defect_type": "hardness_low",
            "root_cause": "淬火温度不足导致奥氏体化不充分" * 2,
            "solution": "将淬火温度提升至 860℃" * 2,
            "created_at": datetime.now(),
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)
        assert result["dimensions"]["validation"] == 0.0  # 0.5 - 0.5 = 0
        assert any("失败记录" in r for r in result["reasons"])

    def test_score_case_clamped_to_range(self, scorer):
        """评分应限制在 [0, 1]"""
        record = {
            "record_id": "rec_range",
            "batch_id": "B007",
            "defect_type": "hardness_low",
            "root_cause": "测试" * 20,
            "solution": "测试" * 20,
            "created_at": datetime.now(),
            "quality_score": 0.5,
            "line_id": "heat_treatment",
        }
        result = scorer.score_case(record)
        assert 0.0 <= result["new_score"] <= 1.0


# ===== 2. 批量评分（TestScoreAll）=====


class TestScoreAll:
    """score_all 批量评分测试"""

    def test_empty_database(self, scorer):
        """空库批量评分应返回 0"""
        result = scorer.score_all()
        assert result["total"] == 0
        assert result["updated"] == 0
        assert result["skipped"] == 0
        assert result["avg_score"] == 0.0
        assert result["by_tier"] == {"high": 0, "medium": 0, "low": 0}
        assert result["sample"] == []

    def test_single_case_update(self, scorer, memory):
        """单案例批量评分应更新 quality_score 字段"""
        _seed_episodic(memory, record_id="rec_s1", root_cause="原因" * 20,
                       solution="提升温度至 860℃" * 5)
        result = scorer.score_all()

        assert result["total"] == 1
        assert result["updated"] == 1
        assert result["skipped"] == 0
        assert result["avg_score"] > 0

        # 验证数据库已更新
        cur = memory.db.execute(
            "SELECT quality_score FROM episodic WHERE record_id = ?", ("rec_s1",)
        )
        db_score = cur.fetchone()[0]
        assert db_score == result["sample"][0]["new_score"]

    def test_dry_run_no_write(self, scorer, memory):
        """dry_run=True 仅评分不写入"""
        _seed_episodic(memory, record_id="rec_dry")
        result = scorer.score_all(dry_run=True)

        assert result["total"] == 1
        assert result["updated"] == 0  # dry_run 不写入
        assert result["skipped"] == 1

        # 验证数据库仍是默认值 0.5
        cur = memory.db.execute(
            "SELECT quality_score FROM episodic WHERE record_id = ?", ("rec_dry",)
        )
        assert cur.fetchone()[0] == 0.5

    def test_multiple_cases_by_tier(self, scorer, memory):
        """多案例评分后 by_tier 统计正确"""
        # 高分案例
        _seed_episodic(memory, record_id="rec_h1",
                       root_cause="淬火温度不足导致奥氏体化不充分" * 3,
                       solution="将淬火温度提升至 860℃ 并保温 90 分钟" * 3)
        # 中分案例（字段部分缺失）
        _seed_episodic(memory, record_id="rec_m1",
                       root_cause="温度低",
                       solution="提温",
                       days_ago=30)
        # 低分案例（全空）
        _seed_episodic(memory, record_id="rec_l1",
                       root_cause="", solution="", defect_type="")

        result = scorer.score_all()
        assert result["total"] == 3
        assert result["updated"] == 3
        assert result["by_tier"]["high"] + result["by_tier"]["medium"] + result["by_tier"]["low"] == 3

    def test_limit_truncation(self, scorer, memory):
        """limit 应截断处理数量"""
        for i in range(5):
            _seed_episodic(memory, record_id=f"rec_lmt_{i}")
        result = scorer.score_all(limit=2)
        assert result["total"] == 2  # 截断

    def test_line_id_filter(self, scorer, memory):
        """line_id 过滤应只评分指定产线"""
        _seed_episodic(memory, record_id="rec_ht", line_id="heat_treatment")
        _seed_episodic(memory, record_id="rec_wd", line_id="welding")
        result = scorer.score_all(line_id="heat_treatment")
        assert result["total"] == 1
        assert result["sample"][0]["record_id"] == "rec_ht"


# ===== 3. 数据层方法（TestMemoryQualityMethods）=====


class TestMemoryQualityMethods:
    """MemoryService M5-6 新增方法测试"""

    def test_update_quality_score(self, memory):
        """update_quality_score 应更新字段"""
        _seed_episodic(memory, record_id="rec_u1")
        ok = memory.update_quality_score("rec_u1", 0.85)
        assert ok is True

        cur = memory.db.execute(
            "SELECT quality_score FROM episodic WHERE record_id = ?", ("rec_u1",)
        )
        assert cur.fetchone()[0] == 0.85

    def test_update_quality_score_clamped(self, memory):
        """评分应限制在 [0, 1]"""
        _seed_episodic(memory, record_id="rec_u2")
        memory.update_quality_score("rec_u2", 1.5)
        cur = memory.db.execute(
            "SELECT quality_score FROM episodic WHERE record_id = ?", ("rec_u2",)
        )
        assert cur.fetchone()[0] == 1.0

        memory.update_quality_score("rec_u2", -0.5)
        cur = memory.db.execute(
            "SELECT quality_score FROM episodic WHERE record_id = ?", ("rec_u2",)
        )
        assert cur.fetchone()[0] == 0.0

    def test_update_quality_score_nonexistent(self, memory):
        """不存在的 record_id 应返回 False"""
        ok = memory.update_quality_score("nonexistent", 0.5)
        assert ok is False

    def test_get_quality_distribution_empty(self, memory):
        """空库分布统计"""
        stats = memory.get_quality_distribution()
        assert stats["total"] == 0
        assert stats["by_tier"] == {"high": 0, "medium": 0, "low": 0}
        assert stats["avg_score"] == 0.0

    def test_get_quality_distribution_by_tier(self, memory):
        """分布统计分档正确"""
        _seed_episodic(memory, record_id="rec_h", days_ago=0)
        _seed_episodic(memory, record_id="rec_m", days_ago=0)
        _seed_episodic(memory, record_id="rec_l", days_ago=0)
        memory.update_quality_score("rec_h", 0.85)
        memory.update_quality_score("rec_m", 0.55)
        memory.update_quality_score("rec_l", 0.20)

        stats = memory.get_quality_distribution(days=365)
        assert stats["total"] == 3
        assert stats["by_tier"]["high"] == 1
        assert stats["by_tier"]["medium"] == 1
        assert stats["by_tier"]["low"] == 1
        assert abs(stats["avg_score"] - round((0.85 + 0.55 + 0.20) / 3, 4)) < 0.01
        assert stats["min_score"] == 0.20
        assert stats["max_score"] == 0.85

    def test_get_quality_distribution_line_filter(self, memory):
        """分布统计支持 line_id 过滤"""
        _seed_episodic(memory, record_id="rec_ht", line_id="heat_treatment")
        _seed_episodic(memory, record_id="rec_wd", line_id="welding")
        stats = memory.get_quality_distribution(line_id="heat_treatment", days=365)
        assert stats["total"] == 1

    def test_get_low_quality_cases(self, memory):
        """get_low_quality_cases 返回低于阈值的案例"""
        _seed_episodic(memory, record_id="rec_low1", days_ago=0)
        _seed_episodic(memory, record_id="rec_low2", days_ago=0)
        _seed_episodic(memory, record_id="rec_high", days_ago=0)
        memory.update_quality_score("rec_low1", 0.20)
        memory.update_quality_score("rec_low2", 0.35)
        memory.update_quality_score("rec_high", 0.85)

        cases = memory.get_low_quality_cases(threshold=0.4)
        assert len(cases) == 2
        # 按升序排列
        assert cases[0]["quality_score"] <= cases[1]["quality_score"]
        assert all(c["quality_score"] < 0.4 for c in cases)

    def test_get_low_quality_cases_empty(self, memory):
        """无低质量案例时返回空列表"""
        _seed_episodic(memory, record_id="rec_hi", days_ago=0)
        memory.update_quality_score("rec_hi", 0.9)
        cases = memory.get_low_quality_cases(threshold=0.4)
        assert cases == []


# ===== 4. API 端点（TestQualityAPI）=====


class TestQualityAPI:
    """M5-6 API 端点测试"""

    def test_score_endpoint_admin(self, client, memory):
        """admin 触发评分应成功"""
        _seed_episodic(memory, record_id="rec_api_1",
                       root_cause="淬火温度不足" * 10,
                       solution="提升温度至 860℃" * 5)
        resp = client.post(
            "/api/v1/cases/quality/score?user_id=admin&days=365"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["dry_run"] is False
        assert data["total"] == 1
        assert data["updated"] == 1

    def test_score_endpoint_non_admin_forbidden(self, client):
        """非 admin 触发评分应 403"""
        resp = client.post(
            "/api/v1/cases/quality/score?user_id=operator_01"
        )
        assert resp.status_code == 403

    def test_score_endpoint_dry_run(self, client, memory):
        """dry_run=True 不写入数据库"""
        _seed_episodic(memory, record_id="rec_api_2")
        resp = client.post(
            "/api/v1/cases/quality/score?user_id=admin&dry_run=true"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["updated"] == 0

    def test_stats_endpoint_admin_all(self, client, memory):
        """admin 查看全部产线统计"""
        _seed_episodic(memory, record_id="rec_s1", line_id="heat_treatment")
        _seed_episodic(memory, record_id="rec_s2", line_id="welding")
        memory.update_quality_score("rec_s1", 0.8)
        memory.update_quality_score("rec_s2", 0.3)

        resp = client.get("/api/v1/cases/quality/stats?user_id=admin&days=365")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["stats"]["total"] == 2

    def test_stats_endpoint_line_filter(self, client, memory):
        """非 admin 用户查看自己产线统计"""
        _seed_episodic(memory, record_id="rec_ht", line_id="heat_treatment")
        _seed_episodic(memory, record_id="rec_wd", line_id="welding")
        memory.update_quality_score("rec_ht", 0.8)
        memory.update_quality_score("rec_wd", 0.5)

        # operator_01 只有 heat_treatment 权限（见 config/line_access.yaml）
        resp = client.get(
            "/api/v1/cases/quality/stats?user_id=operator_01&days=365"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["total"] == 1  # 只看到 heat_treatment

    def test_stats_endpoint_forbidden_line(self, client):
        """无权访问的产线应 403"""
        # operator_01 无 welding 权限
        resp = client.get(
            "/api/v1/cases/quality/stats?user_id=operator_01&line_id=welding"
        )
        assert resp.status_code == 403

    def test_low_endpoint_admin(self, client, memory):
        """admin 获取低质量案例"""
        _seed_episodic(memory, record_id="rec_low", days_ago=0)
        _seed_episodic(memory, record_id="rec_high", days_ago=0)
        memory.update_quality_score("rec_low", 0.2)
        memory.update_quality_score("rec_high", 0.9)

        resp = client.get("/api/v1/cases/quality/low?user_id=admin&threshold=0.4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert data["cases"][0]["record_id"] == "rec_low"

    def test_low_endpoint_threshold_filter(self, client, memory):
        """threshold 参数过滤"""
        _seed_episodic(memory, record_id="rec_a", days_ago=0)
        _seed_episodic(memory, record_id="rec_b", days_ago=0)
        memory.update_quality_score("rec_a", 0.5)
        memory.update_quality_score("rec_b", 0.2)

        # threshold=0.3 只返回 rec_b
        resp = client.get("/api/v1/cases/quality/low?user_id=admin&threshold=0.3")
        data = resp.json()
        assert data["count"] == 1
        assert data["cases"][0]["record_id"] == "rec_b"

    def test_quality_detail_endpoint(self, client, memory):
        """单案例评分详情"""
        _seed_episodic(memory, record_id="rec_detail",
                       root_cause="淬火温度不足" * 10,
                       solution="提升温度至 860℃" * 5)
        resp = client.get("/api/v1/cases/rec_detail/quality?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["quality"]["record_id"] == "rec_detail"
        assert "dimensions" in data["quality"]
        assert "reasons" in data["quality"]

    def test_quality_detail_not_found(self, client):
        """不存在的案例应 404"""
        resp = client.get("/api/v1/cases/nonexistent/quality?user_id=admin")
        assert resp.status_code == 404

    def test_quality_detail_forbidden_line(self, client, memory):
        """无权访问的案例产线应 403"""
        _seed_episodic(memory, record_id="rec_wd_only", line_id="welding")
        # operator_01 无 welding 权限
        resp = client.get("/api/v1/cases/rec_wd_only/quality?user_id=operator_01")
        assert resp.status_code == 403
