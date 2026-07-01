"""
M1-17 工具联调单测

把 scripts/tool_integration_test.py 的关键断言抽成 pytest 用例，
方便 CI 跑（不依赖 loguru 输出，不写日志文件，纯断言）。
"""
import asyncio
import sys
import types
from pathlib import Path

import pytest

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===== 注入 mcp 模块 mock =====


@pytest.fixture(scope="module", autouse=True)
def _inject_mcp_mock():
    """注入 mcp 模块 mock，让 mcp_servers/*_server.py 可以 import"""
    if "mcp" not in sys.modules:
        class _MockServer:
            def __init__(self, name: str = "mock"):
                self.name = name
                self._list_tools = None
                self._call_tool = None

            def list_tools(self, func=None):
                if func is None:
                    def decorator(f):
                        self._list_tools = f
                        return f
                    return decorator
                self._list_tools = func
                return func

            def call_tool(self, func=None):
                if func is None:
                    def decorator(f):
                        self._call_tool = f
                        return f
                    return decorator
                self._call_tool = func
                return func

            def create_initialization_options(self):
                return types.SimpleNamespace()

            async def run(self, *args, **kwargs):
                pass

        class _TextContent:
            def __init__(self, type: str = "text", text: str = ""):
                self.type = type
                self.text = text

        class _Tool:
            def __init__(self, name: str = "", description: str = "", inputSchema: dict = None):
                self.name = name
                self.description = description
                self.inputSchema = inputSchema or {}

        mcp_mod = types.ModuleType("mcp")
        server_mod = types.ModuleType("mcp.server")
        stdio_mod = types.ModuleType("mcp.server.stdio")
        types_mod = types.ModuleType("mcp.types")

        server_mod.Server = _MockServer
        stdio_mod.stdio_server = lambda: None
        types_mod.TextContent = _TextContent
        types_mod.Tool = _Tool

        mcp_mod.server = server_mod
        server_mod.stdio = stdio_mod
        mcp_mod.types = types_mod

        sys.modules["mcp"] = mcp_mod
        sys.modules["mcp.server"] = server_mod
        sys.modules["mcp.server.stdio"] = stdio_mod
        sys.modules["mcp.types"] = types_mod
    yield


# ===== Layer 1: Agent 入口层 =====


class TestAgentToolsLayer:
    """测试 agent/tools.py 的 5 个工具函数"""

    def test_tool_registry_has_5_tools(self):
        from agent.tools import TOOL_REGISTRY
        assert len(TOOL_REGISTRY) == 5
        expected = {"query_batch_params", "query_defect_history",
                    "run_metallurgy_model", "search_handbook", "search_cases"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_query_batch_params_seed_case(self):
        from agent.tools import query_batch_params
        r = query_batch_params("B20260701-A")
        assert r["_source"] == "seed_case"
        assert "temperature" in r and "holding_time" in r and "cooling_rate" in r

    def test_query_batch_params_mock(self):
        from agent.tools import query_batch_params
        r = query_batch_params("B20260628-A")
        assert r["_source"] == "mock_data"
        assert r["temperature"] == 830

    def test_query_batch_params_unknown_fallback(self):
        from agent.tools import query_batch_params
        r = query_batch_params("UNKNOWN-TEST-001")
        assert r["_source"] == "mock_default"

    def test_query_defect_history_filtered(self):
        from agent.tools import query_defect_history
        r = query_defect_history(defect_type="hardness_low", days_back=30, limit=10)
        assert r["total"] > 0
        assert all(rec["defect_type"] == "hardness_low" for rec in r["records"])

    def test_jmak_model_returns_hardness(self):
        from agent.tools import run_metallurgy_model
        r = run_metallurgy_model("jmak", temperature=850, holding_time=120)
        assert r["model"] == "JMAK"
        assert 40 <= r["outputs"]["predicted_hardness_HRc"] <= 65

    def test_cooling_rate_model(self):
        from agent.tools import run_metallurgy_model
        r = run_metallurgy_model("cooling_rate", cooling_rate=10.0)
        assert r["outputs"]["estimated_hardness_HRc"] > 45

    def test_hall_petch_model(self):
        from agent.tools import run_metallurgy_model
        r = run_metallurgy_model("hall_petch", grain_size=20.0)
        assert r["outputs"]["yield_strength_MPa"] > 200

    def test_jmak_missing_params_returns_error(self):
        from agent.tools import run_metallurgy_model
        r = run_metallurgy_model("jmak")
        assert "error" in r

    def test_unknown_model_returns_error(self):
        from agent.tools import run_metallurgy_model
        r = run_metallurgy_model("nonexistent_model", temperature=850)
        assert "error" in r

    def test_search_handbook_returns_results(self):
        from agent.tools import search_handbook
        r = search_handbook("45钢 调质 保温时间", top_k=3)
        assert "total" in r and "results" in r

    def test_search_cases_returns_results(self):
        from agent.tools import search_cases
        r = search_cases("硬度偏低", top_k=3)
        assert "total" in r and "results" in r

    def test_call_tool_dispatch(self):
        from agent.tools import call_tool
        r = call_tool("query_batch_params", batch_id="B20260701-A")
        assert "temperature" in r

    def test_call_tool_unknown_returns_error(self):
        from agent.tools import call_tool
        r = call_tool("nonexistent_tool", foo="bar")
        assert "error" in r


# ===== Layer 2: MCP 实现层 =====


class TestMcpMesServer:
    """测试 MES MCP Server"""

    @pytest.fixture(autouse=True)
    def _isolate_mes_env(self, monkeypatch):
        """隔离 MES 环境变量，避免 .env 的 MES_API_BASE 导致走真实接口分支"""
        monkeypatch.delenv("MES_API_BASE", raising=False)
        monkeypatch.delenv("MES_API_KEY", raising=False)

    def test_list_tools_count(self):
        from mcp_servers.mes_server import server
        tools = asyncio.run(server._list_tools())
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"query_batch_params", "query_defect_history", "submit_adjustment"}

    def test_query_batch_params_required_fields(self):
        from mcp_servers.mes_server import query_batch_params
        r = asyncio.run(query_batch_params("B20260701-A"))
        for k in ["batch_id", "temperature", "holding_time", "cooling_rate"]:
            assert k in r, f"缺少必填字段 {k}"

    def test_query_defect_history(self):
        from mcp_servers.mes_server import query_defect_history
        r = asyncio.run(query_defect_history(defect_type="hardness_low", days_back=30, limit=10))
        assert "total" in r and "records" in r

    def test_submit_adjustment_status_pending(self):
        from mcp_servers.mes_server import submit_adjustment
        r = asyncio.run(submit_adjustment(
            batch_id="B20260701-A",
            adjustments={"cooling_rate": "+2.0"},
            reason="测试",
        ))
        assert r["status"] == "pending_review"
        assert r["proposal_id"].startswith("PA-")

    def test_call_tool_returns_textcontent(self):
        from mcp_servers.mes_server import server
        handler = server._call_tool
        result = asyncio.run(handler("query_batch_params", {"batch_id": "B20260701-A"}))
        assert len(result) == 1
        assert hasattr(result[0], "text")

    def test_call_tool_unknown_tool(self):
        from mcp_servers.mes_server import server
        result = asyncio.run(server._call_tool("nonexistent", {}))
        import json
        parsed = json.loads(result[0].text)
        assert "error" in parsed

    def test_call_tool_missing_required_param(self):
        from mcp_servers.mes_server import server
        result = asyncio.run(server._call_tool("query_batch_params", {}))
        import json
        parsed = json.loads(result[0].text)
        assert "error" in parsed


class TestMcpMetallurgyServer:
    """测试 Metallurgy MCP Server"""

    def test_list_tools_count(self):
        from mcp_servers.metallurgy_server import server
        tools = asyncio.run(server._list_tools())
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"run_metallurgy_model", "validate_hypothesis"}

    def test_jmak_model(self):
        from mcp_servers.metallurgy_server import run_metallurgy_model
        r = run_metallurgy_model("jmak", temperature=850, holding_time=120,
                                 cooling_rate=None, grain_size=None)
        assert r["model"] == "JMAK"
        # MCP 层与 agent/tools.py 输出 key 已统一为 predicted_hardness_HRc
        assert "predicted_hardness_HRc" in r["outputs"]

    def test_cooling_rate_model(self):
        from mcp_servers.metallurgy_server import run_metallurgy_model
        r = run_metallurgy_model("cooling_rate", temperature=None, holding_time=None,
                                 cooling_rate=8.0, grain_size=None)
        assert "estimated_hardness_HRc" in r["outputs"]

    def test_hall_petch_model(self):
        from mcp_servers.metallurgy_server import run_metallurgy_model
        r = run_metallurgy_model("hall_petch", temperature=None, holding_time=None,
                                 cooling_rate=None, grain_size=15.0)
        assert "yield_strength_MPa" in r["outputs"]

    def test_jmak_temp_factor_clamped_below_700(self):
        """温度低于 700℃ 时 temp_factor 应被钳制为 0.1"""
        from mcp_servers.metallurgy_server import run_metallurgy_model
        r = run_metallurgy_model("jmak", temperature=500, holding_time=120,
                                 cooling_rate=None, grain_size=None)
        assert r["parameters"]["temp_factor"] == 0.1

    def test_validate_hypothesis_supported(self):
        from mcp_servers.metallurgy_server import validate_hypothesis
        r = validate_hypothesis(
            hypothesis="保温时间不足导致硬度偏低",
            actual_params={"temperature": 840, "holding_time": 50, "cooling_rate": 6.0},
            measured_value=52.0,
            standard_value=58.0,
        )
        assert r["verdict"] == "supported"
        assert r["confidence"] > 0.5
        assert len(r["evidence"]) >= 2

    def test_validate_hypothesis_inconclusive(self):
        from mcp_servers.metallurgy_server import validate_hypothesis
        r = validate_hypothesis(
            hypothesis="无问题",
            actual_params={"temperature": 850, "holding_time": 120, "cooling_rate": 8.0},
            measured_value=58.0,
            standard_value=58.0,
        )
        assert r["verdict"] == "inconclusive"

    def test_unknown_model_returns_error(self):
        from mcp_servers.metallurgy_server import run_metallurgy_model
        r = run_metallurgy_model("unknown", temperature=850, holding_time=120,
                                 cooling_rate=None, grain_size=None)
        assert "error" in r


class TestMcpKnowledgeServer:
    """测试 Knowledge MCP Server 工具清单"""

    def test_list_tools_count(self):
        from mcp_servers.knowledge_server import server
        tools = asyncio.run(server._list_tools())
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"search_handbook", "search_cases",
                         "write_case", "update_case_confidence"}

    def test_search_handbook_input_schema(self):
        from mcp_servers.knowledge_server import server
        tools = asyncio.run(server._list_tools())
        for t in tools:
            if t.name == "search_handbook":
                assert "query" in t.inputSchema.get("required", [])
                assert "query" in t.inputSchema["properties"]
                return
        pytest.fail("未找到 search_handbook 工具")

    def test_write_case_required_fields(self):
        from mcp_servers.knowledge_server import server
        tools = asyncio.run(server._list_tools())
        for t in tools:
            if t.name == "write_case":
                required = set(t.inputSchema.get("required", []))
                assert {"case_id", "defect_type", "root_cause", "solution"}.issubset(required)
                return
        pytest.fail("未找到 write_case 工具")


# ===== Layer 3: 端到端调用链 =====


class TestEndToEndFlow:
    """模拟 Agent 处理一个 case 的完整工具调用链"""

    def test_full_flow_sc001(self):
        from agent.tools import (
            query_batch_params,
            query_defect_history,
            run_metallurgy_model,
            search_cases,
            search_handbook,
        )

        # Step 1: 查批次
        batch = query_batch_params("B20260701-A")
        assert batch["batch_id"] == "B20260701-A"

        # Step 2: 查历史缺陷
        defects = query_defect_history(defect_type="hardness_low", days_back=90)
        assert defects["total"] > 0

        # Step 3: JMAK 预测
        jmak = run_metallurgy_model(
            "jmak",
            temperature=batch["temperature"],
            holding_time=batch["holding_time"],
        )
        assert "predicted_hardness_HRc" in jmak["outputs"]

        # Step 4: cooling_rate 模型
        cool = run_metallurgy_model("cooling_rate", cooling_rate=batch["cooling_rate"])
        assert "estimated_hardness_HRc" in cool["outputs"]

        # Step 5: 检索手册
        handbook = search_handbook("45钢 调质 保温时间", top_k=2)
        assert handbook["total"] >= 0  # 可能为 0，但调用不报错

        # Step 6: 检索案例
        cases = search_cases("硬度偏低", top_k=2)
        assert cases["total"] >= 0

        # Step 7: 提交建议
        from mcp_servers.mes_server import submit_adjustment
        proposal = asyncio.run(submit_adjustment(
            batch_id=batch["batch_id"],
            adjustments={"holding_time": "+40min"},
            reason="测试建议",
        ))
        assert proposal["status"] == "pending_review"


# ===== 工具清单 schema 校验 =====


class TestToolSchema:
    """校验 MCP 工具的 inputSchema 结构"""

    def test_all_mcp_tools_have_input_schema(self):
        """所有 MCP 工具必须有 inputSchema 且 type=object"""
        from mcp_servers.mes_server import server as mes_server
        from mcp_servers.metallurgy_server import server as meta_server
        from mcp_servers.knowledge_server import server as knowledge_server

        all_tools = []
        all_tools.extend(asyncio.run(mes_server._list_tools()))
        all_tools.extend(asyncio.run(meta_server._list_tools()))
        all_tools.extend(asyncio.run(knowledge_server._list_tools()))

        assert len(all_tools) == 9
        for tool in all_tools:
            assert tool.inputSchema["type"] == "object", f"{tool.name} inputSchema.type 非 object"
            assert "properties" in tool.inputSchema, f"{tool.name} 缺 properties"

    def test_query_batch_params_schema(self):
        from mcp_servers.mes_server import server
        tools = asyncio.run(server._list_tools())
        for t in tools:
            if t.name == "query_batch_params":
                assert "batch_id" in t.inputSchema.get("required", [])
                assert t.inputSchema["properties"]["batch_id"]["type"] == "string"
                return
        pytest.fail()

    def test_run_metallurgy_model_required_only_model_type(self):
        from mcp_servers.metallurgy_server import server
        tools = asyncio.run(server._list_tools())
        for t in tools:
            if t.name == "run_metallurgy_model":
                assert t.inputSchema.get("required", []) == ["model_type"]
                assert t.inputSchema["properties"]["model_type"]["enum"] == ["jmak", "hall_petch", "cooling_rate"]
                return
        pytest.fail()


# ===== 联调日志文件 =====


class TestIntegrationLogFile:
    """验证联调日志文件存在且内容完整"""

    def test_log_file_exists(self):
        log_path = PROJECT_ROOT / "data" / "tool_integration_log.md"
        if not log_path.exists():
            pytest.skip("联调日志未生成（需先运行 scripts/tool_integration_test.py）")
        content = log_path.read_text(encoding="utf-8")
        assert "M1-17 工具联调" in content
        assert "Layer 1" in content
        assert "Layer 2" in content
        assert "Layer 3" in content
        assert "结论" in content
