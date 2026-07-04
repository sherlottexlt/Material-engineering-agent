"""
M5-5 Prompt 自动优化测试

测试三层：
1. PromptOptimizer 失败模式分析 + 优化生成 + 应用 + 回滚
2. _merge_optimization_rule 合并规则（新增/替换）
3. API 端点（/api/v1/prompts/* 共 6 个）
"""
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

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
    db_path = tmp_path / "test_prompt_opt.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = None
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def prompts_file(tmp_path):
    """临时 prompts.yaml（复制项目原版，避免污染真实配置）"""
    src = Path(__file__).parent.parent / "config" / "prompts.yaml"
    dst = tmp_path / "prompts.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


@pytest.fixture
def optimizer(memory, prompts_file, tmp_path):
    """PromptOptimizer 实例（用临时 prompts 文件 + 临时 history 目录）"""
    from agent.prompt_optimizer import PromptOptimizer
    history_dir = tmp_path / "prompts_history"
    return PromptOptimizer(
        memory=memory,
        prompts_path=prompts_file,
        history_dir=history_dir,
    )


@pytest.fixture
def client(memory, prompts_file, tmp_path):
    """FastAPI 测试客户端（注入临时 memory + optimizer + collector + tracker）"""
    from fastapi.testclient import TestClient
    from api.routes import (
        app, prompt_optimizer, failure_collector, effect_tracker,
    )
    # 备份并替换 optimizer 的 memory/db/prompts_path/history_dir
    orig_opt_memory = prompt_optimizer.memory
    orig_opt_db = prompt_optimizer.db
    orig_opt_prompts = prompt_optimizer.prompts_path
    orig_opt_history = prompt_optimizer.history_dir
    orig_collector_memory = failure_collector.memory
    orig_collector_db = failure_collector.db
    orig_tracker_memory = effect_tracker.memory
    orig_tracker_db = effect_tracker.db

    prompt_optimizer.memory = memory
    prompt_optimizer.db = memory.db
    prompt_optimizer.prompts_path = prompts_file
    prompt_optimizer.history_dir = tmp_path / "prompts_history"
    failure_collector.memory = memory
    failure_collector.db = memory.db
    effect_tracker.memory = memory
    effect_tracker.db = memory.db

    yield TestClient(app)

    prompt_optimizer.memory = orig_opt_memory
    prompt_optimizer.db = orig_opt_db
    prompt_optimizer.prompts_path = orig_opt_prompts
    prompt_optimizer.history_dir = orig_opt_history
    failure_collector.memory = orig_collector_memory
    failure_collector.db = orig_collector_db
    effect_tracker.memory = orig_tracker_memory
    effect_tracker.db = orig_tracker_db


# ===== 辅助函数 =====


def _seed_failure(memory, case_id, category="low_confidence", line_id="heat_treatment",
                  failure_reason="confidence=0.1 < 0.3"):
    """注入失败案例到 failure_cases 表"""
    from agent.failure_case_collector import FailureCaseCollector
    collector = FailureCaseCollector(memory)
    collector._save_failure(
        case_id=case_id,
        tracking_id=None,
        line_id=line_id,
        category=category,
        confidence=0.1 if category == "low_confidence" else None,
        improvement_pct=-15.0 if category == "negative_effect" else None,
        failure_reason=failure_reason,
    )


# ===== 1. 失败模式分析测试 =====


class TestAnalyzeFailurePatterns:
    """analyze_failure_patterns: 从 failure_cases 分析失败模式"""

    def test_no_failures_returns_empty(self, optimizer, memory):
        """无失败案例时返回空列表"""
        patterns = optimizer.analyze_failure_patterns()
        assert patterns == []

    def test_low_confidence_maps_to_decision_and_review(self, optimizer, memory):
        """low_confidence 应映射到 decision + review 两个角色"""
        _seed_failure(memory, "c1", category="low_confidence")
        _seed_failure(memory, "c2", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        roles = {p["role"] for p in patterns}
        assert "decision" in roles
        assert "review" in roles
        # 每个角色的 failure_count 应为 2
        for p in patterns:
            assert p["failure_count"] == 2
            assert p["failure_category"] == "low_confidence"

    def test_rejected_feedback_maps_to_decision_only(self, optimizer, memory):
        """rejected_feedback 只映射到 decision"""
        _seed_failure(memory, "c1", category="rejected_feedback",
                      failure_reason="action=rejected, score=0.2")
        patterns = optimizer.analyze_failure_patterns()
        roles = {p["role"] for p in patterns}
        assert roles == {"decision"}

    def test_negative_effect_maps_to_decision_only(self, optimizer, memory):
        """negative_effect 只映射到 decision"""
        _seed_failure(memory, "c1", category="negative_effect",
                      failure_reason="improvement_pct=-15.0% < 0")
        patterns = optimizer.analyze_failure_patterns()
        roles = {p["role"] for p in patterns}
        assert roles == {"decision"}

    def test_mixed_categories(self, optimizer, memory):
        """混合类别应分别映射"""
        _seed_failure(memory, "c1", category="low_confidence")
        _seed_failure(memory, "c2", category="rejected_feedback",
                      failure_reason="action=rejected")
        _seed_failure(memory, "c3", category="negative_effect",
                      failure_reason="improvement_pct=-10%")
        patterns = optimizer.analyze_failure_patterns()
        # low_confidence → decision + review；rejected_feedback → decision；
        # negative_effect → decision
        roles = {p["role"] for p in patterns}
        assert "decision" in roles
        assert "review" in roles
        # decision 应有 3 个 pattern（每个 category 一个，failure_count=1）
        decision_patterns = [p for p in patterns if p["role"] == "decision"]
        assert len(decision_patterns) == 3
        # 每个 pattern 的 failure_category 应不同
        categories = {p["failure_category"] for p in decision_patterns}
        assert categories == {"low_confidence", "rejected_feedback", "negative_effect"}

    def test_sample_failures_included(self, optimizer, memory):
        """失败模式应包含样本失败案例"""
        _seed_failure(memory, "c1", category="low_confidence",
                      failure_reason="confidence=0.1")
        patterns = optimizer.analyze_failure_patterns()
        assert len(patterns) > 0
        assert "sample_failures" in patterns[0]
        assert len(patterns[0]["sample_failures"]) <= 3

    def test_line_id_filter(self, optimizer, memory):
        """line_id 过滤"""
        _seed_failure(memory, "c1", category="low_confidence", line_id="heat_treatment")
        _seed_failure(memory, "c2", category="low_confidence", line_id="welding")
        patterns = optimizer.analyze_failure_patterns(line_id="heat_treatment")
        # 只统计 heat_treatment 的失败
        for p in patterns:
            assert p["failure_count"] == 1


# ===== 2. 优化生成测试 =====


class TestGenerateOptimization:
    """generate_optimization: 生成优化记录"""

    def test_generates_draft(self, optimizer, memory):
        """生成 status=draft 的优化记录"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")

        opt = optimizer.generate_optimization(decision_pattern)
        assert opt is not None
        assert opt["status"] == "draft"
        assert opt["role"] == "decision"
        assert opt["failure_category"] == "low_confidence"
        assert opt["version"] == 1
        assert opt["old_prompt"]
        assert opt["new_prompt"]
        assert len(opt["new_prompt"]) > len(opt["old_prompt"])  # 新 prompt 更长
        assert opt["change_summary"]

    def test_new_prompt_contains_optimization_marker(self, optimizer, memory):
        """新 prompt 应包含 M5-5 自动优化规则标记"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")

        opt = optimizer.generate_optimization(decision_pattern)
        assert "【M5-5 自动优化规则" in opt["new_prompt"]

    def test_idempotent_when_applied_exists(self, optimizer, memory):
        """同 role+category 已有 applied 记录时，generate 不重复生成"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")

        # 第一次生成 + 应用
        opt1 = optimizer.generate_optimization(decision_pattern)
        optimizer.apply_optimization(opt1["optimization_id"])

        # 第二次生成（应返回已 applied 的记录，不新建）
        opt2 = optimizer.generate_optimization(decision_pattern)
        assert opt2["optimization_id"] == opt1["optimization_id"]
        assert opt2["status"] == "applied"

        # 数据库只有 1 条记录
        all_opts = optimizer.list_optimizations()
        assert len(all_opts) == 1

    def test_force_generates_new_even_if_applied(self, optimizer, memory):
        """force=True 时强制生成新记录"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")

        opt1 = optimizer.generate_optimization(decision_pattern)
        optimizer.apply_optimization(opt1["optimization_id"])

        opt2 = optimizer.generate_optimization(decision_pattern, force=True)
        assert opt2["optimization_id"] != opt1["optimization_id"]
        assert opt2["version"] == 2
        assert opt2["status"] == "draft"

    def test_version_increments(self, optimizer, memory):
        """版本号应递增"""
        _seed_failure(memory, "c1", category="low_confidence")
        _seed_failure(memory, "c2", category="rejected_feedback",
                      failure_reason="action=rejected")
        patterns = optimizer.analyze_failure_patterns()

        versions = []
        for pattern in patterns:
            opt = optimizer.generate_optimization(pattern)
            if opt:
                versions.append(opt["version"])
        # 版本号应递增且唯一
        assert len(set(versions)) == len(versions)
        assert versions == sorted(versions)

    def test_unknown_role_returns_none(self, optimizer, memory):
        """未知角色应返回 None"""
        pattern = {
            "role": "unknown_role",
            "failure_category": "low_confidence",
            "failure_count": 1,
            "trigger_reason": "test",
            "sample_failures": [],
            "sample_failure_reasons": "test",
        }
        opt = optimizer.generate_optimization(pattern)
        assert opt is None


# ===== 3. 规则合并测试 =====


class TestMergeOptimizationRule:
    """_merge_optimization_rule: 合并优化规则到原 prompt"""

    def test_appends_new_rule(self, optimizer):
        """原 prompt 无规则区块时应追加"""
        old = "你是测试 Agent。\n规则：\n1. 规则一\n"
        rule = "\n\n【M5-5 自动优化规则（基于 1 条失败）】\n1. 新规则\n"
        new = optimizer._merge_optimization_rule(old, rule)
        assert "【M5-5 自动优化规则" in new
        assert new.startswith("你是测试 Agent")
        assert len(new) > len(old)

    def test_replaces_existing_rule(self, optimizer):
        """原 prompt 已有规则区块时应替换（幂等）"""
        old = (
            "你是测试 Agent。\n规则：\n1. 规则一\n"
            "\n\n【M5-5 自动优化规则（基于 1 条失败）】\n1. 旧规则\n2. 旧规则二\n"
        )
        rule = "\n\n【M5-5 自动优化规则（基于 5 条失败）】\n1. 新规则\n"
        new = optimizer._merge_optimization_rule(old, rule)
        # 旧规则应被替换
        assert "旧规则" not in new
        assert "新规则" in new
        # 基础 prompt 应保留
        assert "你是测试 Agent" in new
        # 新 prompt 长度应短于含旧规则的 prompt（因为新规则更短）
        assert len(new) < len(old)

    def test_preserves_original_prompt_content(self, optimizer):
        """合并规则不应破坏原 prompt 内容"""
        old = "你是【决策 Agent】。\n职责：综合信息\n规则：\n1. 输出方案\n"
        rule = "\n\n【M5-5 自动优化规则（基于 1 条失败）】\n1. 新规则\n"
        new = optimizer._merge_optimization_rule(old, rule)
        assert "你是【决策 Agent】" in new
        assert "职责：综合信息" in new
        assert "1. 输出方案" in new


# ===== 4. 应用与回滚测试 =====


class TestApplyOptimization:
    """apply_optimization: 应用优化"""

    def test_apply_draft_to_applied(self, optimizer, memory, prompts_file):
        """应用 draft 记录：状态 → applied，prompts.yaml 更新"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")
        opt = optimizer.generate_optimization(decision_pattern)

        # 应用前 prompts.yaml 中 decision_agent 应无优化规则
        with open(prompts_file, "r", encoding="utf-8") as f:
            before = yaml.safe_load(f)
        assert "【M5-5 自动优化规则" not in before["decision_agent"]

        # 应用
        result = optimizer.apply_optimization(opt["optimization_id"])
        assert result is not None
        assert result["status"] == "applied"
        assert result["snapshot_path"]
        assert Path(result["snapshot_path"]).exists()

        # 应用后 prompts.yaml 中 decision_agent 应包含优化规则
        with open(prompts_file, "r", encoding="utf-8") as f:
            after = yaml.safe_load(f)
        assert "【M5-5 自动优化规则" in after["decision_agent"]

    def test_apply_non_draft_returns_none(self, optimizer, memory):
        """应用非 draft 记录应返回 None"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")
        opt = optimizer.generate_optimization(decision_pattern)
        optimizer.apply_optimization(opt["optimization_id"])

        # 重复应用应失败
        result = optimizer.apply_optimization(opt["optimization_id"])
        assert result is None

    def test_apply_nonexistent_returns_none(self, optimizer):
        """应用不存在的记录应返回 None"""
        result = optimizer.apply_optimization("opt_nonexistent")
        assert result is None

    def test_snapshot_backup_created(self, optimizer, memory):
        """应用时应创建快照备份文件"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")
        opt = optimizer.generate_optimization(decision_pattern)

        optimizer.apply_optimization(opt["optimization_id"])
        applied = optimizer.get_optimization(opt["optimization_id"])
        snapshot_path = Path(applied["snapshot_path"])
        assert snapshot_path.exists()
        # 快照应是有效的 yaml
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot = yaml.safe_load(f)
        assert "decision_agent" in snapshot

    def test_clears_lru_cache(self, optimizer, memory):
        """应用后应清除 load_prompts 的 lru_cache"""
        from agent.utils import load_prompts
        # 先填充 cache
        load_prompts.cache_clear()
        load_prompts()
        assert load_prompts.cache_info().currsize > 0

        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")
        opt = optimizer.generate_optimization(decision_pattern)
        optimizer.apply_optimization(opt["optimization_id"])

        # cache 应被清除
        assert load_prompts.cache_info().currsize == 0


class TestRollbackOptimization:
    """rollback_optimization: 回滚优化"""

    def test_rollback_applied_to_rolled_back(self, optimizer, memory, prompts_file):
        """回滚 applied 记录：状态 → rolled_back，prompts.yaml 恢复"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")
        opt = optimizer.generate_optimization(decision_pattern)

        # 记录原始 prompt
        with open(prompts_file, "r", encoding="utf-8") as f:
            original = yaml.safe_load(f)
        original_prompt = original["decision_agent"]

        # 应用 → 回滚
        optimizer.apply_optimization(opt["optimization_id"])
        optimizer.rollback_optimization(opt["optimization_id"])

        # prompts.yaml 应恢复原状
        with open(prompts_file, "r", encoding="utf-8") as f:
            restored = yaml.safe_load(f)
        assert restored["decision_agent"] == original_prompt

        # 状态应为 rolled_back
        record = optimizer.get_optimization(opt["optimization_id"])
        assert record["status"] == "rolled_back"

    def test_rollback_non_applied_returns_none(self, optimizer, memory):
        """回滚非 applied 记录应返回 None"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")
        opt = optimizer.generate_optimization(decision_pattern)
        # 不应用直接回滚
        result = optimizer.rollback_optimization(opt["optimization_id"])
        assert result is None

    def test_rollback_nonexistent_returns_none(self, optimizer):
        """回滚不存在的记录应返回 None"""
        result = optimizer.rollback_optimization("opt_nonexistent")
        assert result is None


# ===== 5. 查询测试 =====


class TestListAndGetOptimizations:
    """list_optimizations / get_optimization"""

    def test_list_empty(self, optimizer):
        """空表返回空列表"""
        assert optimizer.list_optimizations() == []

    def test_list_all(self, optimizer, memory):
        """列出全部"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        for p in patterns:
            optimizer.generate_optimization(p)
        all_opts = optimizer.list_optimizations()
        # low_confidence → decision + review
        assert len(all_opts) == 2

    def test_filter_by_role(self, optimizer, memory):
        """按角色过滤"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        for p in patterns:
            optimizer.generate_optimization(p)
        decision_opts = optimizer.list_optimizations(role="decision")
        assert len(decision_opts) == 1
        assert all(o["role"] == "decision" for o in decision_opts)

    def test_filter_by_status(self, optimizer, memory):
        """按状态过滤"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        for p in patterns:
            opt = optimizer.generate_optimization(p)
        # 全部是 draft
        drafts = optimizer.list_optimizations(status="draft")
        assert len(drafts) == 2
        applied = optimizer.list_optimizations(status="applied")
        assert len(applied) == 0

    def test_get_existing(self, optimizer, memory):
        """查询单条"""
        _seed_failure(memory, "c1", category="low_confidence")
        patterns = optimizer.analyze_failure_patterns()
        decision_pattern = next(p for p in patterns if p["role"] == "decision")
        opt = optimizer.generate_optimization(decision_pattern)
        result = optimizer.get_optimization(opt["optimization_id"])
        assert result is not None
        assert result["optimization_id"] == opt["optimization_id"]
        assert result["role"] == "decision"

    def test_get_nonexistent(self, optimizer):
        """查询不存在返回 None"""
        assert optimizer.get_optimization("opt_xxx") is None


class TestGetCurrentPrompts:
    """get_current_prompts"""

    def test_get_all(self, optimizer):
        """查看全部 prompts"""
        prompts = optimizer.get_current_prompts()
        assert "decision_agent" in prompts
        assert "review_agent" in prompts
        assert "planner" in prompts

    def test_get_by_role(self, optimizer):
        """按角色查看"""
        prompts = optimizer.get_current_prompts(role="decision")
        assert "decision_agent" in prompts
        assert "review_agent" not in prompts

    def test_unknown_role_returns_empty(self, optimizer):
        """未知角色返回空 dict"""
        prompts = optimizer.get_current_prompts(role="unknown")
        assert prompts == {}


# ===== 6. API 端点测试 =====


class TestPromptsAPI:
    """/api/v1/prompts/* 端点测试"""

    def test_optimize_requires_admin(self, client):
        """非 admin 不能触发优化"""
        resp = client.post("/api/v1/prompts/optimize?user_id=operator_01")
        assert resp.status_code == 403

    def test_optimize_no_failures(self, client):
        """无失败案例时返回无需优化"""
        resp = client.post("/api/v1/prompts/optimize?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["patterns_found"] == 0
        assert data["optimizations_generated"] == 0

    def test_optimize_generates_drafts(self, client, memory):
        """有失败案例时生成 draft 优化"""
        _seed_failure(memory, "c1", category="low_confidence")
        resp = client.post("/api/v1/prompts/optimize?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["patterns_found"] == 2  # decision + review
        assert data["optimizations_generated"] == 2
        assert data["optimizations_applied"] == 0
        for opt in data["optimizations"]:
            assert opt["status"] == "draft"

    def test_optimize_with_apply(self, client, memory):
        """apply=true 时自动应用"""
        _seed_failure(memory, "c1", category="low_confidence")
        resp = client.post("/api/v1/prompts/optimize?user_id=admin&apply=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["optimizations_applied"] == 2
        assert len(data["applied_ids"]) == 2

    def test_list_optimizations_admin(self, client, memory):
        """admin 列出优化历史"""
        _seed_failure(memory, "c1", category="low_confidence")
        client.post("/api/v1/prompts/optimize?user_id=admin")
        resp = client.get("/api/v1/prompts/optimizations?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        # 列表不应包含完整 prompt 文本
        for r in data["records"]:
            assert "old_prompt" not in r
            assert "new_prompt" not in r

    def test_list_optimizations_non_admin(self, client):
        """非 admin 不能查看"""
        resp = client.get("/api/v1/prompts/optimizations?user_id=operator_01")
        assert resp.status_code == 403

    def test_get_optimization_by_id(self, client, memory):
        """按 ID 查询（含完整 prompt）"""
        _seed_failure(memory, "c1", category="low_confidence")
        client.post("/api/v1/prompts/optimize?user_id=admin")
        list_resp = client.get("/api/v1/prompts/optimizations?user_id=admin")
        opt_id = list_resp.json()["records"][0]["optimization_id"]

        resp = client.get(f"/api/v1/prompts/optimizations/{opt_id}?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["optimization_id"] == opt_id
        assert "old_prompt" in data
        assert "new_prompt" in data

    def test_get_optimization_not_found(self, client):
        """查询不存在应 404"""
        resp = client.get("/api/v1/prompts/optimizations/opt_xxx?user_id=admin")
        assert resp.status_code == 404

    def test_apply_optimization(self, client, memory, prompts_file):
        """手动应用 draft"""
        _seed_failure(memory, "c1", category="low_confidence")
        client.post("/api/v1/prompts/optimize?user_id=admin")
        list_resp = client.get("/api/v1/prompts/optimizations?user_id=admin")
        opt_id = list_resp.json()["records"][0]["optimization_id"]

        resp = client.post(f"/api/v1/prompts/apply/{opt_id}?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "applied"

    def test_rollback_optimization(self, client, memory, prompts_file):
        """回滚 applied"""
        _seed_failure(memory, "c1", category="low_confidence")
        client.post("/api/v1/prompts/optimize?user_id=admin&apply=true")
        list_resp = client.get("/api/v1/prompts/optimizations?user_id=admin&status=applied")
        opt_id = list_resp.json()["records"][0]["optimization_id"]

        resp = client.post(f"/api/v1/prompts/rollback/{opt_id}?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "rolled_back"

    def test_get_current_prompts(self, client):
        """查看当前 prompts"""
        resp = client.get("/api/v1/prompts/current?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompts" in data
        assert "decision_agent" in data["prompts"]

    def test_get_current_prompts_by_role(self, client):
        """按角色查看当前 prompts"""
        resp = client.get("/api/v1/prompts/current?user_id=admin&role=decision")
        assert resp.status_code == 200
        data = resp.json()
        assert "decision_agent" in data["prompts"]
        assert "review_agent" not in data["prompts"]

    def test_get_current_prompts_non_admin(self, client):
        """非 admin 不能查看"""
        resp = client.get("/api/v1/prompts/current?user_id=operator_01")
        assert resp.status_code == 403
