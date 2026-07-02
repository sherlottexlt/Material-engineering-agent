"""
M4-10 多租户权限隔离测试

测试两层：
1. 权限加载器（agent/utils.py: load_line_access/get_user_lines/can_access_line/...）
2. API 端点权限校验（api/routes.py: _check_line_access + 各端点）

权限矩阵（config/line_access.yaml）：
- admin:           ["*"]           全产线，可读写删
- supervisor_01:   [heat_treatment, welding]  跨产线，可读写
- operator_01:     [heat_treatment] 单产线，可读写
- operator_02:     [welding]        单产线，可读写
- 未知用户:        default_role=operator，无 allowed_lines → 无权访问任何产线
"""
import pytest
from fastapi.testclient import TestClient


# ===== 权限加载器单元测试 =====


class TestAccessLoader:
    """M4-10: 权限配置加载器单元测试"""

    def test_load_line_access_exists(self):
        """line_access.yaml 应能正常加载"""
        from agent.utils import load_line_access
        access = load_line_access()
        assert isinstance(access, dict)
        assert "roles" in access
        assert "users" in access

    def test_admin_role_all_lines(self):
        """admin 角色应有全部产线访问权限（通配符 *）"""
        from agent.utils import get_user_lines, can_access_line
        assert "*" in get_user_lines("admin")
        assert can_access_line("admin", "heat_treatment")
        assert can_access_line("admin", "welding")
        assert can_access_line("admin", "any_unknown_line")  # 通配符覆盖

    def test_supervisor_multi_line(self):
        """supervisor_01 应能访问 heat_treatment 和 welding，但不能访问其他"""
        from agent.utils import can_access_line, get_user_lines
        lines = get_user_lines("supervisor_01")
        assert "heat_treatment" in lines
        assert "welding" in lines
        assert can_access_line("supervisor_01", "heat_treatment")
        assert can_access_line("supervisor_01", "welding")
        assert not can_access_line("supervisor_01", "rolling")

    def test_operator_single_line(self):
        """operator_01 只能访问 heat_treatment"""
        from agent.utils import can_access_line, get_user_lines
        assert get_user_lines("operator_01") == ["heat_treatment"]
        assert can_access_line("operator_01", "heat_treatment")
        assert not can_access_line("operator_01", "welding")

    def test_operator_02_welding_only(self):
        """operator_02 只能访问 welding"""
        from agent.utils import can_access_line
        assert can_access_line("operator_02", "welding")
        assert not can_access_line("operator_02", "heat_treatment")

    def test_unknown_user_default_role(self):
        """未知用户走 default_role=operator，无 allowed_lines → 无权访问"""
        from agent.utils import get_user_role, can_access_line
        assert get_user_role("unknown_user") == "operator"
        # operator 角色无 allowed_lines，未知用户应无权访问任何产线
        assert not can_access_line("unknown_user", "heat_treatment")
        assert not can_access_line("unknown_user", "welding")

    def test_get_user_permissions_admin(self):
        """admin 权限：can_write=True, can_delete=True"""
        from agent.utils import get_user_permissions
        perms = get_user_permissions("admin")
        assert perms["role"] == "admin"
        assert perms["can_write"] is True
        assert perms["can_delete"] is True
        assert "*" in perms["allowed_lines"]

    def test_get_user_permissions_operator(self):
        """operator 权限：can_write=True, can_delete=False"""
        from agent.utils import get_user_permissions
        perms = get_user_permissions("operator_01")
        assert perms["role"] == "operator"
        assert perms["can_write"] is True
        assert perms["can_delete"] is False

    def test_get_user_permissions_supervisor(self):
        """supervisor 权限：can_write=True, can_delete=False"""
        from agent.utils import get_user_permissions
        perms = get_user_permissions("supervisor_01")
        assert perms["role"] == "supervisor"
        assert perms["can_write"] is True
        assert perms["can_delete"] is False


# ===== API 端点权限校验测试 =====


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    from api.routes import app
    return TestClient(app)


class TestAuthPermissionsEndpoint:
    """M4-10: /api/v1/auth/permissions 端点"""

    def test_get_admin_permissions(self, client):
        """查询 admin 权限"""
        resp = client.get("/api/v1/auth/permissions", params={"user_id": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "admin"
        assert data["role"] == "admin"
        assert "*" in data["allowed_lines"]
        assert data["can_write"] is True
        assert data["can_delete"] is True
        assert "heat_treatment" in data["available_lines"]

    def test_get_operator_permissions(self, client):
        """查询 operator 权限"""
        resp = client.get("/api/v1/auth/permissions", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "operator"
        assert data["allowed_lines"] == ["heat_treatment"]
        assert data["can_delete"] is False

    def test_get_unknown_user_permissions(self, client):
        """未知用户走 default_role"""
        resp = client.get("/api/v1/auth/permissions", params={"user_id": "ghost"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "operator"
        assert data["allowed_lines"] == []


class TestLinesEndpoint:
    """M4-10: /api/v1/lines 端点"""

    def test_admin_sees_all_lines(self, client):
        """admin 可见全部产线"""
        resp = client.get("/api/v1/lines", params={"user_id": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2  # heat_treatment + welding
        line_ids = [l["line_id"] for l in data["lines"]]
        assert "heat_treatment" in line_ids
        assert "welding" in line_ids

    def test_operator_sees_only_own_line(self, client):
        """operator_01 只能见到 heat_treatment"""
        resp = client.get("/api/v1/lines", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["lines"][0]["line_id"] == "heat_treatment"

    def test_operator_02_sees_welding_only(self, client):
        """operator_02 只能见到 welding"""
        resp = client.get("/api/v1/lines", params={"user_id": "operator_02"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["lines"][0]["line_id"] == "welding"

    def test_line_summary_has_config_fields(self, client):
        """产线摘要应包含配置字段"""
        resp = client.get("/api/v1/lines", params={"user_id": "admin"})
        data = resp.json()
        line = data["lines"][0]
        assert "name" in line
        assert "process_type" in line
        assert "material" in line
        assert "defect_types" in line


class TestCasesEndpointAccessControl:
    """M4-10: /api/v1/cases 端点权限隔离"""

    def test_admin_sees_all_cases(self, client):
        """admin 可见全部产线案例（无 line_id 过滤）"""
        resp = client.get("/api/v1/cases", params={"user_id": "admin"})
        assert resp.status_code == 200
        # 不验证具体数量，只验证端点可访问
        assert "total" in resp.json()

    def test_operator_filtered_to_own_line(self, client):
        """operator_01 不传 line_id 时只返回 heat_treatment 案例"""
        resp = client.get("/api/v1/cases", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        data = resp.json()
        # 所有返回的案例 line_id 都应是 heat_treatment
        for case in data["cases"]:
            assert case.get("line_id") == "heat_treatment"

    def test_operator_cannot_access_other_line(self, client):
        """operator_01 无权访问 welding 产线 → 403"""
        resp = client.get("/api/v1/cases", params={
            "user_id": "operator_01", "line_id": "welding"
        })
        assert resp.status_code == 403
        assert "无权访问" in resp.json()["detail"]

    def test_operator_can_access_own_line(self, client):
        """operator_01 可访问 heat_treatment 产线"""
        resp = client.get("/api/v1/cases", params={
            "user_id": "operator_01", "line_id": "heat_treatment"
        })
        assert resp.status_code == 200

    def test_unknown_user_no_access(self, client):
        """未知用户无权访问任何产线"""
        resp = client.get("/api/v1/cases", params={
            "user_id": "ghost", "line_id": "heat_treatment"
        })
        assert resp.status_code == 403


class TestFeedbackEndpointAccessControl:
    """M4-10: /api/v1/feedback 端点写权限校验"""

    def test_operator_cannot_feedback_other_line(self, client):
        """operator_01 无权给 welding 产线提交反馈 → 403"""
        resp = client.post("/api/v1/feedback", json={
            "proposal_id": "P_TEST",
            "user_id": "operator_01",
            "action": "adopted",
            "score": 0.9,
            "line_id": "welding",
        })
        assert resp.status_code == 403

    def test_operator_can_feedback_own_line(self, client):
        """operator_01 可给 heat_treatment 产线提交反馈"""
        resp = client.post("/api/v1/feedback", json={
            "proposal_id": "P_TEST_OK",
            "user_id": "operator_01",
            "action": "adopted",
            "score": 0.9,
            "line_id": "heat_treatment",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestCaseCRUDAccessControl:
    """M4-10: 案例库 CRUD 权限校验"""

    def test_create_case_wrong_line_forbidden(self, client):
        """operator_01 无权向 welding 产线写入案例 → 403"""
        resp = client.post("/api/v1/memory/cases", json={
            "defect_type": "porosity",
            "batch_id": "B_TEST",
            "root_cause": "测试根因",
            "line_id": "welding",
            "user_id": "operator_01",
        })
        assert resp.status_code == 403

    def test_create_case_own_line_allowed(self, client):
        """operator_01 可向 heat_treatment 产线写入案例"""
        resp = client.post("/api/v1/memory/cases", json={
            "defect_type": "hardness_low",
            "batch_id": "B_TEST_OK",
            "root_cause": "保温时间不足",
            "line_id": "heat_treatment",
            "user_id": "operator_01",
        })
        # 200 或 503（Chroma 不可用时）都算权限校验通过
        assert resp.status_code in (200, 503)

    def test_admin_can_create_any_line(self, client):
        """admin 可向任意产线写入案例"""
        resp = client.post("/api/v1/memory/cases", json={
            "defect_type": "porosity",
            "batch_id": "B_ADMIN",
            "root_cause": "admin 写入测试",
            "line_id": "welding",
            "user_id": "admin",
        })
        assert resp.status_code in (200, 503)


class TestMemoryQueryAccessControl:
    """M4-10: 记忆查询端点权限隔离"""

    def test_list_episodic_operator_filtered(self, client):
        """operator_01 查询 episodic 只返回 heat_treatment"""
        resp = client.get("/api/v1/memory/episodic", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        for rec in resp.json()["records"]:
            assert rec.get("line_id") == "heat_treatment"

    def test_list_episodic_cross_line_forbidden(self, client):
        """operator_01 无权查询 welding episodic → 403"""
        resp = client.get("/api/v1/memory/episodic", params={
            "user_id": "operator_01", "line_id": "welding"
        })
        assert resp.status_code == 403

    def test_list_feedback_operator_filtered(self, client):
        """operator_01 查询 feedback 只返回 heat_treatment"""
        resp = client.get("/api/v1/memory/feedback", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        for rec in resp.json()["records"]:
            assert rec.get("line_id") == "heat_treatment"

    def test_list_conflicts_operator_filtered(self, client):
        """operator_01 查询 conflicts 只返回 heat_treatment"""
        resp = client.get("/api/v1/memory/conflicts", params={"user_id": "operator_01"})
        assert resp.status_code == 200
        for rec in resp.json()["records"]:
            assert rec.get("line_id") == "heat_treatment"

    def test_supervisor_sees_both_lines(self, client):
        """supervisor_01 可见 heat_treatment 和 welding 两条产线"""
        resp = client.get("/api/v1/memory/episodic", params={"user_id": "supervisor_01"})
        assert resp.status_code == 200
        # 不验证具体数量，只验证不报 403
        line_ids = {r.get("line_id") for r in resp.json()["records"]}
        # 应只包含 supervisor 有权限的产线
        assert line_ids.issubset({"heat_treatment", "welding"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
