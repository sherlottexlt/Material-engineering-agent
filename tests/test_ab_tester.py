"""
M5-8 A/B 测试框架测试

测试三层：
1. ABTestFramework 实验管理 + 分配 + 指标 + 分析 + 采集
2. MemoryService 数据层方法（save/list/get/update + ab_stats）
3. API 端点（/api/v1/abtest/* 共 9 个）
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
    db_path = tmp_path / "test_ab_test.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def tester(memory):
    """ABTestFramework 实例（无外部依赖）"""
    from agent.ab_tester import ABTestFramework
    return ABTestFramework(memory=memory)


@pytest.fixture
def running_experiment(tester):
    """创建并启动一个实验"""
    result = tester.create_experiment(
        name="test_exp",
        description="测试实验",
        line_id="heat_treatment",
        variant_a_config={"version": "v1"},
        variant_b_config={"version": "v2"},
    )
    exp_id = result["experiment_id"]
    tester.start_experiment(exp_id)
    return exp_id


@pytest.fixture
def client(memory):
    """FastAPI 测试客户端（注入临时 memory + tester）"""
    from fastapi.testclient import TestClient
    from api.routes import app, ab_tester
    import api.routes as routes_module

    orig_memory = ab_tester.memory
    orig_db = ab_tester.db
    orig_routes_memory = routes_module.memory

    ab_tester.memory = memory
    ab_tester.db = memory.db
    routes_module.memory = memory

    yield TestClient(app)

    ab_tester.memory = orig_memory
    ab_tester.db = orig_db
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


# ===== 1. 实验创建（TestCreateExperiment）=====


class TestCreateExperiment:
    """create_experiment 测试"""

    def test_create_minimal(self, tester):
        """最小参数创建"""
        result = tester.create_experiment(name="exp1", description="测试")
        assert result["success"] is True
        assert result["experiment_id"].startswith("exp_")

    def test_create_with_configs(self, tester):
        """带 A/B 配置创建"""
        result = tester.create_experiment(
            name="exp2",
            description="带配置",
            variant_a_config={"prompt_version": "v1"},
            variant_b_config={"prompt_version": "v2"},
        )
        exp_id = result["experiment_id"]
        exp = tester.get_experiment(exp_id)
        assert exp["variant_a_config"]["prompt_version"] == "v1"
        assert exp["variant_b_config"]["prompt_version"] == "v2"

    def test_create_with_metrics(self, tester):
        """自定义指标名"""
        result = tester.create_experiment(
            name="exp3",
            description="自定义指标",
            metric_names=["quality_score", "custom_metric"],
        )
        exp = tester.get_experiment(result["experiment_id"])
        assert "quality_score" in exp["metric_names"]
        assert "custom_metric" in exp["metric_names"]

    def test_get_experiment(self, tester):
        """查询实验详情"""
        result = tester.create_experiment(name="exp4", description="查询测试")
        exp = tester.get_experiment(result["experiment_id"])
        assert exp is not None
        assert exp["name"] == "exp4"
        assert exp["status"] == "draft"

    def test_get_nonexistent(self, tester):
        """查询不存在的实验返回 None"""
        assert tester.get_experiment("nonexistent") is None


# ===== 2. 启停实验（TestStartStopExperiment）=====


class TestStartStopExperiment:
    """start_experiment / stop_experiment 测试"""

    def test_start_draft(self, tester):
        """draft → running"""
        result = tester.create_experiment(name="exp_start")
        exp_id = result["experiment_id"]
        start_result = tester.start_experiment(exp_id)
        assert start_result["success"] is True
        assert start_result["status"] == "running"

        exp = tester.get_experiment(exp_id)
        assert exp["status"] == "running"
        assert exp["started_at"] is not None

    def test_start_nonexistent(self, tester):
        """启动不存在的实验失败"""
        result = tester.start_experiment("nonexistent")
        assert result["success"] is False

    def test_start_already_running(self, tester, running_experiment):
        """已 running 不能再启动"""
        result = tester.start_experiment(running_experiment)
        assert result["success"] is False

    def test_stop_running(self, tester, running_experiment):
        """running → stopped"""
        result = tester.stop_experiment(running_experiment)
        assert result["success"] is True
        assert result["status"] == "stopped"

        exp = tester.get_experiment(running_experiment)
        assert exp["status"] == "stopped"
        assert exp["stopped_at"] is not None

    def test_stop_non_running(self, tester):
        """非 running 状态不能停止"""
        result = tester.create_experiment(name="exp_stop")
        exp_id = result["experiment_id"]
        stop_result = tester.stop_experiment(exp_id)
        assert stop_result["success"] is False


# ===== 3. 分配（TestAssignment）=====


class TestAssignment:
    """assign / get_assignment / list_assignments 测试"""

    def test_assign_deterministic(self, tester, running_experiment):
        """同 case_id 多次分配结果一致"""
        r1 = tester.assign(running_experiment, "case_001")
        r2 = tester.assign(running_experiment, "case_001")
        assert r1["success"] is True
        assert r2["success"] is True
        assert r1["variant"] == r2["variant"]
        assert r2["reassigned"] is True  # 第二次返回已有分配

    def test_assign_idempotent(self, tester, running_experiment):
        """已分配返回原记录"""
        r1 = tester.assign(running_experiment, "case_idem")
        assignment_id_1 = r1["assignment_id"]
        r2 = tester.assign(running_experiment, "case_idem")
        assert r2["assignment_id"] == assignment_id_1

    def test_assign_nonexistent_experiment(self, tester):
        """实验不存在"""
        result = tester.assign("nonexistent", "case_001")
        assert result["success"] is False

    def test_assign_not_running(self, tester):
        """非 running 状态不能分配"""
        result = tester.create_experiment(name="exp_not_running")
        exp_id = result["experiment_id"]
        assign_result = tester.assign(exp_id, "case_001")
        assert assign_result["success"] is False

    def test_assign_distribution(self, tester, running_experiment):
        """多个 case_id 分配均匀（约 50/50）"""
        variants = []
        for i in range(20):
            r = tester.assign(running_experiment, f"case_dist_{i}")
            variants.append(r["variant"])
        a_count = variants.count("a")
        b_count = variants.count("b")
        # 至少各有 5 个（20 个里允许 5-15 范围）
        assert a_count >= 5
        assert b_count >= 5

    def test_get_assignment(self, tester, running_experiment):
        """查询已有分配"""
        tester.assign(running_experiment, "case_get")
        assignment = tester.get_assignment(running_experiment, "case_get")
        assert assignment is not None
        assert assignment["case_id"] == "case_get"
        assert assignment["variant"] in ("a", "b")


# ===== 4. 指标记录（TestRecordMetric）=====


class TestRecordMetric:
    """record_metric / record_metric_batch / list_metrics 测试"""

    def test_record_single(self, tester, running_experiment):
        """单条记录"""
        tester.assign(running_experiment, "case_metric")
        result = tester.record_metric(
            experiment_id=running_experiment,
            case_id="case_metric",
            metric_name="quality_score",
            metric_value=0.85,
        )
        assert result["success"] is True
        assert result["metric_id"].startswith("mtc_")

    def test_record_batch(self, tester, running_experiment):
        """批量记录"""
        tester.assign(running_experiment, "case_batch_1")
        tester.assign(running_experiment, "case_batch_2")
        result = tester.record_metric_batch(
            experiment_id=running_experiment,
            metrics=[
                {"case_id": "case_batch_1", "metric_name": "quality_score", "metric_value": 0.8},
                {"case_id": "case_batch_2", "metric_name": "quality_score", "metric_value": 0.6},
                {"case_id": "case_unassigned", "metric_name": "quality_score", "metric_value": 0.5},
            ],
        )
        assert result["success"] is True
        assert result["recorded"] == 2
        assert result["failed"] == 1

    def test_record_nonexistent_case(self, tester, running_experiment):
        """未分配的 case 失败"""
        result = tester.record_metric(
            experiment_id=running_experiment,
            case_id="case_unassigned",
            metric_name="quality_score",
            metric_value=0.5,
        )
        assert result["success"] is False

    def test_record_invalid_value(self, tester, running_experiment):
        """非数值失败"""
        tester.assign(running_experiment, "case_invalid")
        result = tester.record_metric(
            experiment_id=running_experiment,
            case_id="case_invalid",
            metric_name="quality_score",
            metric_value="not_a_number",
        )
        assert result["success"] is False

    def test_list_metrics(self, tester, running_experiment):
        """列出指标"""
        tester.assign(running_experiment, "case_list_1")
        tester.assign(running_experiment, "case_list_2")
        tester.record_metric(running_experiment, "case_list_1", "quality_score", 0.8)
        tester.record_metric(running_experiment, "case_list_2", "quality_score", 0.6)

        metrics = tester.list_metrics(running_experiment)
        assert len(metrics) == 2

        # 按 variant 过滤
        variant_a = [m for m in metrics if m["variant"] == "a"]
        variant_b = [m for m in metrics if m["variant"] == "b"]
        assert len(variant_a) + len(variant_b) == 2


# ===== 5. 分析（TestAnalyze）=====


class TestAnalyze:
    """analyze 测试"""

    def test_analyze_empty(self, tester, running_experiment):
        """无数据分析"""
        result = tester.analyze(running_experiment)
        assert result["success"] is True
        assert result["sample_sizes"]["a"] == 0
        assert result["sample_sizes"]["b"] == 0
        assert result["metrics"] == {}

        assert "无指标数据" in result["conclusion"]

    def test_analyze_single_variant(self, tester, running_experiment):
        """只有一组数据（无法 t 检验）"""
        # 强制分配到 a 组（用 mock）
        tester.memory.save_ab_assignment(
            assignment_id="asg_test_a1",
            experiment_id=running_experiment,
            variant="a",
            case_id="case_only_a_1",
            line_id="heat_treatment",
        )
        tester.memory.save_ab_assignment(
            assignment_id="asg_test_a2",
            experiment_id=running_experiment,
            variant="a",
            case_id="case_only_a_2",
            line_id="heat_treatment",
        )
        tester.record_metric(running_experiment, "case_only_a_1", "quality_score", 0.8)
        tester.record_metric(running_experiment, "case_only_a_2", "quality_score", 0.7)

        result = tester.analyze(running_experiment)
        assert result["success"] is True
        assert "quality_score" in result["metrics"]
        assert result["metrics"]["quality_score"]["a"]["n"] == 2
        assert result["metrics"]["quality_score"]["b"]["n"] == 0
        # b 样本不足，不显著
        assert result["metrics"]["quality_score"]["significant"] is False

    def test_analyze_significant_b(self, tester, running_experiment):
        """B 组显著优（quality_score 越大越好）"""
        # 注入大量数据：A 组 0.5，B 组 0.9
        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_sig_a_{i}",
                experiment_id=running_experiment,
                variant="a",
                case_id=f"case_sig_a_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_sig_a_{i}", "quality_score", 0.5)

        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_sig_b_{i}",
                experiment_id=running_experiment,
                variant="b",
                case_id=f"case_sig_b_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_sig_b_{i}", "quality_score", 0.9)

        result = tester.analyze(running_experiment)
        metric = result["metrics"]["quality_score"]
        assert metric["a"]["mean"] == 0.5
        assert metric["b"]["mean"] == 0.9
        assert metric["diff"] > 0
        assert metric["significant"] is True
        assert metric["winner"] == "b"
        assert "B 组" in result["conclusion"]

    def test_analyze_significant_a(self, tester, running_experiment):
        """A 组显著优"""
        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_sa_{i}",
                experiment_id=running_experiment,
                variant="a",
                case_id=f"case_sa_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_sa_{i}", "quality_score", 0.9)

        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_sb_{i}",
                experiment_id=running_experiment,
                variant="b",
                case_id=f"case_sb_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_sb_{i}", "quality_score", 0.5)

        result = tester.analyze(running_experiment)
        metric = result["metrics"]["quality_score"]
        assert metric["significant"] is True
        assert metric["winner"] == "a"

    def test_analyze_no_difference(self, tester, running_experiment):
        """无显著差异（两组相同）"""
        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_nd_a_{i}",
                experiment_id=running_experiment,
                variant="a",
                case_id=f"case_nd_a_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_nd_a_{i}", "quality_score", 0.7)

        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_nd_b_{i}",
                experiment_id=running_experiment,
                variant="b",
                case_id=f"case_nd_b_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_nd_b_{i}", "quality_score", 0.7)

        result = tester.analyze(running_experiment)
        metric = result["metrics"]["quality_score"]
        # 两组完全相同，t 检验 se=0，mean 相同 → p=1.0
        assert metric["significant"] is False
        assert metric["winner"] == "no_difference"

    def test_analyze_failure_rate_special(self, tester, running_experiment):
        """failure_rate 越小越好（特殊处理 winner）"""
        # A 组失败率 0.8（高），B 组失败率 0.2（低）→ B 组优
        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_fr_a_{i}",
                experiment_id=running_experiment,
                variant="a",
                case_id=f"case_fr_a_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_fr_a_{i}", "failure_rate", 0.8)

        for i in range(30):
            tester.memory.save_ab_assignment(
                assignment_id=f"asg_fr_b_{i}",
                experiment_id=running_experiment,
                variant="b",
                case_id=f"case_fr_b_{i}",
                line_id="heat_treatment",
            )
            tester.record_metric(running_experiment, f"case_fr_b_{i}", "failure_rate", 0.2)

        result = tester.analyze(running_experiment)
        metric = result["metrics"]["failure_rate"]
        assert metric["significant"] is True
        # B 组失败率低 → B 组优
        assert metric["winner"] == "b"

    def test_analyze_nonexistent(self, tester):
        """分析不存在的实验"""
        result = tester.analyze("nonexistent")
        assert result["success"] is False


# ===== 6. 从模块采集指标（TestCollectFromModules）=====


class TestCollectFromModules:
    """collect_metrics_from_quality/effect/failures 测试"""

    def test_collect_from_quality_no_scorer(self, tester, running_experiment):
        """未注入 quality_scorer 失败"""
        result = tester.collect_metrics_from_quality(running_experiment)
        assert result["success"] is False
        assert "quality_scorer" in result["error"]

    def test_collect_from_quality(self, memory, running_experiment):
        """注入 quality_scorer 后采集"""
        from agent.ab_tester import ABTestFramework
        from agent.case_quality_scorer import CaseQualityScorer

        # 创建带 quality_scorer 的 tester
        tester = ABTestFramework(memory=memory, quality_scorer=CaseQualityScorer(memory))

        # 注入案例 + 分配
        _seed_episodic(memory, record_id="case_q_1", quality_score=0.5)
        tester.assign(running_experiment, "case_q_1")

        result = tester.collect_metrics_from_quality(running_experiment)
        assert result["success"] is True
        assert result["recorded"] >= 1

        # 验证指标已记录
        metrics = tester.list_metrics(running_experiment, metric_name="quality_score")
        assert len(metrics) >= 1

    def test_collect_from_effect_no_tracker(self, tester, running_experiment):
        """未注入 effect_tracker 失败"""
        result = tester.collect_metrics_from_effect(running_experiment)
        assert result["success"] is False
        assert "effect_tracker" in result["error"]

    def test_collect_from_failures_no_collector(self, tester, running_experiment):
        """未注入 failure_collector 失败"""
        result = tester.collect_metrics_from_failures(running_experiment)
        assert result["success"] is False
        assert "failure_collector" in result["error"]


# ===== 7. 数据层方法（TestMemoryLayer）=====


class TestMemoryLayer:
    """MemoryService M5-8 新增方法测试"""

    def test_save_and_get_experiment(self, memory):
        """保存并查询实验"""
        memory.save_ab_experiment(
            experiment_id="exp_test_1",
            name="测试实验",
            description="数据层测试",
            line_id="heat_treatment",
            variant_a_config='{"version":"v1"}',
            variant_b_config='{"version":"v2"}',
            metric_names='["quality_score"]',
            sample_size_target=50,
            created_by="admin",
        )
        exp = memory.get_ab_experiment("exp_test_1")
        assert exp is not None
        assert exp["name"] == "测试实验"
        assert exp["status"] == "draft"

    def test_list_experiments_filter(self, memory):
        """列出实验支持过滤"""
        memory.save_ab_experiment(
            experiment_id="exp_f1", name="exp1", description="", line_id="heat_treatment",
            variant_a_config="{}", variant_b_config="{}", metric_names="[]",
            sample_size_target=100, created_by="admin",
        )
        memory.save_ab_experiment(
            experiment_id="exp_f2", name="exp2", description="", line_id="welding",
            variant_a_config="{}", variant_b_config="{}", metric_names="[]",
            sample_size_target=100, created_by="admin",
        )

        # 全部
        all_exps = memory.list_ab_experiments()
        assert len(all_exps) == 2

        # 按 line_id 过滤
        ht_only = memory.list_ab_experiments(line_id="heat_treatment")
        assert len(ht_only) == 1
        assert ht_only[0]["experiment_id"] == "exp_f1"

        # 按 line_id list 过滤
        multi = memory.list_ab_experiments(line_id=["heat_treatment", "welding"])
        assert len(multi) == 2

    def test_save_and_get_assignment(self, memory):
        """保存并查询分配"""
        memory.save_ab_assignment(
            assignment_id="asg_test_1",
            experiment_id="exp_assign",
            variant="a",
            case_id="case_001",
            line_id="heat_treatment",
        )
        assignment = memory.get_ab_assignment("exp_assign", "case_001")
        assert assignment is not None
        assert assignment["variant"] == "a"

    def test_list_assignments_filter(self, memory):
        """列出分配支持 variant 过滤"""
        for i in range(5):
            memory.save_ab_assignment(
                assignment_id=f"asg_l_{i}",
                experiment_id="exp_list",
                variant="a" if i < 3 else "b",
                case_id=f"case_l_{i}",
                line_id="heat_treatment",
            )
        a_only = memory.list_ab_assignments("exp_list", variant="a")
        assert len(a_only) == 3
        b_only = memory.list_ab_assignments("exp_list", variant="b")
        assert len(b_only) == 2

    def test_update_experiment_status(self, memory):
        """更新实验状态"""
        memory.save_ab_experiment(
            experiment_id="exp_upd", name="upd", description="", line_id="heat_treatment",
            variant_a_config="{}", variant_b_config="{}", metric_names="[]",
            sample_size_target=100, created_by="admin",
        )
        memory.update_ab_experiment_status("exp_upd", "running")
        exp = memory.get_ab_experiment("exp_upd")
        assert exp["status"] == "running"
        assert exp["started_at"] is not None

        memory.update_ab_experiment_status("exp_upd", "stopped")
        exp = memory.get_ab_experiment("exp_upd")
        assert exp["status"] == "stopped"
        assert exp["stopped_at"] is not None

    def test_ab_stats(self, memory):
        """A/B 测试总体统计"""
        memory.save_ab_experiment(
            experiment_id="exp_s1", name="s1", description="", line_id="heat_treatment",
            variant_a_config="{}", variant_b_config="{}", metric_names="[]",
            sample_size_target=100, created_by="admin",
        )
        memory.save_ab_assignment(
            assignment_id="asg_s1", experiment_id="exp_s1", variant="a",
            case_id="case_s1", line_id="heat_treatment",
        )
        memory.save_ab_metric(
            metric_id="mtc_s1", experiment_id="exp_s1", assignment_id="asg_s1",
            variant="a", metric_name="quality_score", metric_value=0.8,
        )
        stats = memory.get_ab_stats()
        assert "by_status" in stats
        assert stats["total_assignments"] >= 1
        assert stats["total_metrics"] >= 1


# ===== 8. API 端点（TestABTestAPI）=====


class TestABTestAPI:
    """M5-8 API 端点测试"""

    def test_create_experiment_admin(self, client):
        """admin 创建实验"""
        resp = client.post(
            "/api/v1/abtest/experiments",
            json={"name": "api_exp", "description": "API 测试", "user_id": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["experiment_id"].startswith("exp_")

    def test_create_experiment_operator_forbidden(self, client):
        """operator 创建实验 403"""
        resp = client.post(
            "/api/v1/abtest/experiments",
            json={"name": "api_exp_op", "user_id": "operator_01"},
        )
        assert resp.status_code == 403

    def test_list_experiments_admin(self, client):
        """admin 列出全部实验"""
        client.post(
            "/api/v1/abtest/experiments",
            json={"name": "exp_list_1", "user_id": "admin"},
        )
        resp = client.get("/api/v1/abtest/experiments?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] >= 1

    def test_start_experiment(self, client):
        """启动实验"""
        create_resp = client.post(
            "/api/v1/abtest/experiments",
            json={"name": "exp_start_api", "user_id": "admin"},
        )
        exp_id = create_resp.json()["experiment_id"]

        resp = client.post(f"/api/v1/abtest/experiments/{exp_id}/start?user_id=admin")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_assign_endpoint(self, client):
        """分配端点"""
        create_resp = client.post(
            "/api/v1/abtest/experiments",
            json={"name": "exp_assign_api", "user_id": "admin"},
        )
        exp_id = create_resp.json()["experiment_id"]
        client.post(f"/api/v1/abtest/experiments/{exp_id}/start?user_id=admin")

        resp = client.post(
            "/api/v1/abtest/assign",
            json={"experiment_id": exp_id, "case_id": "case_api_1", "user_id": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["variant"] in ("a", "b")

    def test_record_metric_endpoint(self, client):
        """记录指标端点"""
        create_resp = client.post(
            "/api/v1/abtest/experiments",
            json={"name": "exp_metric_api", "user_id": "admin"},
        )
        exp_id = create_resp.json()["experiment_id"]
        client.post(f"/api/v1/abtest/experiments/{exp_id}/start?user_id=admin")
        client.post(
            "/api/v1/abtest/assign",
            json={"experiment_id": exp_id, "case_id": "case_metric_api", "user_id": "admin"},
        )

        resp = client.post(
            "/api/v1/abtest/metrics",
            json={
                "experiment_id": exp_id,
                "case_id": "case_metric_api",
                "metric_name": "quality_score",
                "metric_value": 0.85,
                "user_id": "admin",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_analyze_endpoint(self, client):
        """分析端点"""
        create_resp = client.post(
            "/api/v1/abtest/experiments",
            json={"name": "exp_analyze_api", "user_id": "admin"},
        )
        exp_id = create_resp.json()["experiment_id"]

        resp = client.get(f"/api/v1/abtest/experiments/{exp_id}/analyze?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "conclusion" in data

    def test_metrics_endpoint(self, client):
        """查询指标端点"""
        create_resp = client.post(
            "/api/v1/abtest/experiments",
            json={"name": "exp_metrics_api", "user_id": "admin"},
        )
        exp_id = create_resp.json()["experiment_id"]

        resp = client.get(f"/api/v1/abtest/experiments/{exp_id}/metrics?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "metrics" in data

    def test_get_experiment_not_found(self, client):
        """查询不存在的实验 404"""
        resp = client.get("/api/v1/abtest/experiments/nonexistent?user_id=admin")
        assert resp.status_code == 404
