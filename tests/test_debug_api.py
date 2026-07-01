"""
M2-12 调试工具 API 端点测试

测试 /api/v1/debug/flows 端点（不需要 LLM 调用）。
/debug/run 和 /debug/trace 因需要 LLM 调用或预存数据，仅测试错误路径。
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    from api.routes import app
    return TestClient(app)


class TestDebugFlows:
    """测试 /api/v1/debug/flows 端点"""

    def test_list_flows(self, client):
        """列出可用流程"""
        resp = client.get("/api/v1/debug/flows")
        assert resp.status_code == 200
        data = resp.json()
        assert "flows" in data
        flows = data["flows"]
        assert len(flows) >= 5

        # 检查必要字段
        for flow in flows:
            assert "name" in flow
            assert "description" in flow
            assert "mode" in flow
            assert "parallel_agents" in flow
            assert "enable_arbitrate" in flow

    def test_contains_parallel_flow(self, client):
        """包含 parallel 流程"""
        resp = client.get("/api/v1/debug/flows")
        flows = resp.json()["flows"]
        names = [f["name"] for f in flows]
        assert "parallel" in names

    def test_contains_quick_flow(self, client):
        """包含 quick 流程"""
        resp = client.get("/api/v1/debug/flows")
        flows = resp.json()["flows"]
        names = [f["name"] for f in flows]
        assert "quick" in names

    def test_parallel_flow_has_correct_mode(self, client):
        """parallel 流程 mode=parallel"""
        resp = client.get("/api/v1/debug/flows")
        flows = resp.json()["flows"]
        parallel = next(f for f in flows if f["name"] == "parallel")
        assert parallel["mode"] == "parallel"
        assert "data" in parallel["parallel_agents"]
        assert parallel["enable_arbitrate"] is True

    def test_quick_flow_has_no_arbitrate(self, client):
        """quick 流程 enable_arbitrate=False"""
        resp = client.get("/api/v1/debug/flows")
        flows = resp.json()["flows"]
        quick = next(f for f in flows if f["name"] == "quick")
        assert quick["enable_arbitrate"] is False


class TestDebugTraceNotFound:
    """测试 /api/v1/debug/trace/{trace_id} 错误路径"""

    def test_trace_not_found(self, client):
        """不存在的 trace_id 返回 404"""
        resp = client.get("/api/v1/debug/trace/nonexistent_trace_id")
        assert resp.status_code == 404


class TestDebugRunValidation:
    """测试 /api/v1/debug/run 请求验证"""

    def test_run_without_batch_id(self, client):
        """缺少 batch_id 返回 422（验证错误）"""
        resp = client.post("/api/v1/debug/run", json={"query": "test"})
        assert resp.status_code == 422  # Pydantic 验证失败

    def test_run_with_valid_request_structure(self, client):
        """有效请求结构（不实际执行，只验证参数解析）"""
        # 注意：实际执行需要 LLM，这里只验证请求能被接受
        # 由于会实际调用 LLM，我们只检查请求格式正确后是否会进入执行
        # 如果 LLM 不可用，会返回 500
        resp = client.post("/api/v1/debug/run", json={
            "query": "测试查询",
            "batch_id": "B001",
            "flow_name": "quick",
            "max_replan": 1,
        })
        # 可能是 200（成功）或 500（LLM 不可用），不应该是 422（验证错误）
        assert resp.status_code in (200, 500)
