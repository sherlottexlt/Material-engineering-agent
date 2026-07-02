"""
M4-11 跨产线统一看板测试

测试两层：
1. MemoryService.get_line_stats() 单产线统计方法
2. /api/v1/dashboard/overview API 端点（含权限过滤）
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    from api.routes import app
    return TestClient(app)


@pytest.fixture
def memory():
    """MemoryService 测试实例"""
    from agent.memory.memory_service import MemoryService
    svc = MemoryService()
    # 写入测试数据：两条产线各 1 条短期记忆 + 1 条反馈
    svc.write_episodic("B_DASH_HT", "hardness_low", "保温不足", "延长保温",
                       line_id="heat_treatment")
    svc.write_episodic("B_DASH_WD", "porosity", "电流过大", "降低电流",
                       line_id="welding")
    svc.write_feedback("fb_dash_ht", "P_DASH_HT", "u1", "adopted", 0.9,
                       line_id="heat_treatment")
    svc.write_feedback("fb_dash_wd", "P_DASH_WD", "u2", "rejected", 0.3,
                       line_id="welding")
    yield svc
    # 清理测试数据
    try:
        svc.db.execute("DELETE FROM episodic WHERE batch_id LIKE 'B_DASH_%'")
        svc.db.execute("DELETE FROM feedback WHERE feedback_id LIKE 'fb_dash_%'")
        svc.db.commit()
    except Exception:
        pass


# ===== MemoryService.get_line_stats 单元测试 =====


class TestGetLineStats:
    """M4-11: MemoryService.get_line_stats 方法"""

    def test_returns_correct_structure(self, memory):
        """返回结构应包含全部 KPI 字段"""
        stats = memory.get_line_stats("heat_treatment", days=30)
        expected_keys = {
            "line_id", "episodic_count", "feedback_count", "conflict_count",
            "semantic_count", "defect_distribution", "action_distribution",
            "adoption_rate", "avg_confidence",
        }
        assert set(stats.keys()) >= expected_keys
        assert stats["line_id"] == "heat_treatment"

    def test_counts_episodic_correctly(self, memory):
        """应正确统计短期记忆数（按 line_id 过滤）"""
        ht_stats = memory.get_line_stats("heat_treatment", days=30)
        wd_stats = memory.get_line_stats("welding", days=30)
        assert ht_stats["episodic_count"] >= 1
        assert wd_stats["episodic_count"] >= 1

    def test_counts_feedback_correctly(self, memory):
        """应正确统计反馈数"""
        ht_stats = memory.get_line_stats("heat_treatment", days=30)
        assert ht_stats["feedback_count"] >= 1
        # heat_treatment 有 1 条 adopted 反馈，采纳率应为 100%
        assert ht_stats["adoption_rate"] >= 0.9

    def test_defect_distribution(self, memory):
        """缺陷分布应包含缺陷类型"""
        stats = memory.get_line_stats("heat_treatment", days=30)
        dist = stats["defect_distribution"]
        assert "hardness_low" in dist

    def test_action_distribution(self, memory):
        """反馈动作分布应包含动作类型"""
        ht_stats = memory.get_line_stats("heat_treatment", days=30)
        assert "adopted" in ht_stats["action_distribution"]
        wd_stats = memory.get_line_stats("welding", days=30)
        assert "rejected" in wd_stats["action_distribution"]

    def test_adoption_rate_calculation(self, memory):
        """采纳率应正确计算（adopted / total）"""
        # welding 产线 1 条 rejected 反馈，采纳率应为 0
        wd_stats = memory.get_line_stats("welding", days=30)
        if wd_stats["feedback_count"] > 0:
            adopted = wd_stats["action_distribution"].get("adopted", 0)
            expected = round(adopted / wd_stats["feedback_count"], 3)
            assert wd_stats["adoption_rate"] == expected

    def test_empty_line_returns_zeros(self, memory):
        """无数据的产线应返回 0"""
        stats = memory.get_line_stats("nonexistent_line", days=30)
        assert stats["episodic_count"] == 0
        assert stats["feedback_count"] == 0
        assert stats["conflict_count"] == 0
        assert stats["defect_distribution"] == {}
        assert stats["adoption_rate"] == 0.0

    def test_days_filter(self, memory):
        """days=0 应过滤掉所有数据（近 0 天）"""
        stats = memory.get_line_stats("heat_treatment", days=0)
        # days=0 时 since=now，所有记录都早于 now，应返回 0
        assert stats["episodic_count"] == 0
        assert stats["feedback_count"] == 0


# ===== /api/v1/dashboard/overview 端点测试 =====


class TestDashboardOverviewEndpoint:
    """M4-11: /api/v1/dashboard/overview 端点"""

    def test_admin_sees_all_lines(self, client):
        """admin 应看到全部产线"""
        resp = client.get("/api/v1/dashboard/overview", params={"user_id": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert "lines" in data
        assert "totals" in data
        line_ids = [l["line_id"] for l in data["lines"]]
        assert "heat_treatment" in line_ids
        assert "welding" in line_ids

    def test_operator_sees_only_own_line(self, client):
        """operator_01 应只看到 heat_treatment"""
        resp = client.get("/api/v1/dashboard/overview", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        data = resp.json()
        line_ids = [l["line_id"] for l in data["lines"]]
        assert line_ids == ["heat_treatment"]
        assert "welding" not in line_ids

    def test_operator_02_sees_welding(self, client):
        """operator_02 应只看到 welding"""
        resp = client.get("/api/v1/dashboard/overview", params={"user_id": "operator_02"})
        assert resp.status_code == 200
        line_ids = [resp.json()["lines"][0]["line_id"]]
        assert line_ids == ["welding"]

    def test_supervisor_sees_both_lines(self, client):
        """supervisor_01 应看到 heat_treatment 和 welding"""
        resp = client.get("/api/v1/dashboard/overview", params={"user_id": "supervisor_01"})
        assert resp.status_code == 200
        line_ids = {l["line_id"] for l in resp.json()["lines"]}
        assert line_ids == {"heat_treatment", "welding"}

    def test_unknown_user_sees_nothing(self, client):
        """未知用户应看到 0 条产线"""
        resp = client.get("/api/v1/dashboard/overview", params={"user_id": "ghost"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["lines"] == []
        assert data["totals"]["total_episodic"] == 0

    def test_totals_aggregation(self, client):
        """totals 应正确聚合各产线数据"""
        resp = client.get("/api/v1/dashboard/overview", params={
            "user_id": "admin", "days": 365
        })
        data = resp.json()
        totals = data["totals"]
        expected_keys = {
            "total_episodic", "total_feedback", "total_conflicts",
            "total_semantic", "overall_adoption_rate", "overall_avg_confidence",
        }
        assert set(totals.keys()) >= expected_keys
        # totals 应等于各产线之和
        sum_episodic = sum(l["episodic_count"] for l in data["lines"])
        assert totals["total_episodic"] == sum_episodic

    def test_days_param_filtering(self, client):
        """days 参数应过滤统计数据"""
        resp_365 = client.get("/api/v1/dashboard/overview", params={
            "user_id": "admin", "days": 365
        })
        resp_1 = client.get("/api/v1/dashboard/overview", params={
            "user_id": "admin", "days": 1
        })
        # 365 天应 >= 1 天的数据量
        assert (resp_365.json()["totals"]["total_episodic"]
                >= resp_1.json()["totals"]["total_episodic"])

    def test_line_stats_has_name(self, client):
        """每条产线统计应包含 name 字段（来自产线配置）"""
        resp = client.get("/api/v1/dashboard/overview", params={"user_id": "admin"})
        for line in resp.json()["lines"]:
            assert "name" in line
            assert line["name"]  # 非空


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
