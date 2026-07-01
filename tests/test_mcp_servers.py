"""
MCP Server 单测（M1-13/14/16 配套）

测试 mcp_servers/ 下的三个 MCP Server 的工具函数：
- mes_server.py: query_batch_params / query_defect_history / submit_adjustment
- metallurgy_server.py: jmak_model / hall_petch_model / cooling_rate_model / run_metallurgy_model / validate_hypothesis
- knowledge_server.py: 通过 MemoryService 调用（Chroma 不可用时部分测试会 skip）

注意：MCP 的 @server.call_tool 装饰器包装的入口难以直接单测，
      但底层函数（query_batch_params 等）可直接调用。

环境兼容：当前环境未安装 `mcp` 库，测试通过 sys.modules 注入 mock
        （只 mock 装饰器，不影响工具函数本身逻辑）。
"""
import asyncio
import json
import os
import sys
import types
from contextlib import asynccontextmanager

import pytest

# ===== 注入 mcp mock（如果真实 mcp 已安装则跳过）=====
if "mcp" not in sys.modules:
    # 构造 mcp.server.Server 装饰器 mock
    class _MockServer:
        def __init__(self, name: str = "mock"):
            self.name = name
            self._tools = []
            self._handlers = {}

        def list_tools(self, func=None):
            # 兼容 @server.list_tools() 和 @server.list_tools 两种用法
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

    # 构造 mcp.types.TextContent / Tool
    class _TextContent:
        def __init__(self, type: str = "text", text: str = ""):
            self.type = type
            self.text = text

    class _Tool:
        def __init__(self, name: str = "", description: str = "", inputSchema: dict = None):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema or {}

    # 注入到 sys.modules
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    stdio_mod = types.ModuleType("mcp.server.stdio")
    types_mod = types.ModuleType("mcp.types")

    server_mod.Server = _MockServer

    @asynccontextmanager
    async def _mock_stdio_server():
        yield (None, None)
    stdio_mod.stdio_server = _mock_stdio_server

    types_mod.TextContent = _TextContent
    types_mod.Tool = _Tool

    mcp_mod.server = server_mod
    server_mod.stdio = stdio_mod
    mcp_mod.types = types_mod

    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.stdio"] = stdio_mod
    sys.modules["mcp.types"] = types_mod

# 把项目根加进 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "mcp_servers"))

from mcp_servers.mes_server import (
    query_batch_params as mes_query_batch,
    query_defect_history as mes_query_defects,
    submit_adjustment,
)
from mcp_servers.metallurgy_server import (
    jmak_model,
    hall_petch_model,
    cooling_rate_model,
    run_metallurgy_model,
    validate_hypothesis,
)


# ===== MES Server =====
class TestMesQueryBatchParams:
    """测试 MES 批次参数查询（async）"""

    @pytest.fixture(autouse=True)
    def _isolate_mes_env(self, monkeypatch):
        """隔离 MES 环境变量：测试时强制走 mock 数据分支，避免 .env 中
        MES_API_BASE 指向不存在的服务导致连接失败"""
        monkeypatch.delenv("MES_API_BASE", raising=False)
        monkeypatch.delenv("MES_API_KEY", raising=False)

    def test_returns_batch_info(self):
        """应返回批次基本信息"""
        result = asyncio.run(mes_query_batch("B20260701-A"))
        assert result["batch_id"] == "B20260701-A"
        assert "temperature" in result
        assert "holding_time" in result
        assert "cooling_rate" in result
        assert result["process_type"] == "heat_treatment"

    def test_returns_mock_note(self):
        """未接入真实 MES 时应标注 _note"""
        result = asyncio.run(mes_query_batch("B-TEST-001"))
        assert "_note" in result
        assert "模拟" in result["_note"] or "mock" in result["_note"].lower()

    def test_raw_material_batch_format(self):
        """raw_material_batch 应包含批次后 4 位"""
        result = asyncio.run(mes_query_batch("B20260701-ABCD"))
        # mes_server.py: f"RM-{batch_id[-4:]}" → "ABCD"
        assert result["raw_material_batch"].endswith(result["batch_id"][-4:])

    def test_timestamps_present(self):
        """应包含 start_time 和 end_time"""
        result = asyncio.run(mes_query_batch("B-X"))
        assert "start_time" in result
        assert "end_time" in result


class TestMesQueryDefectHistory:
    """测试 MES 历史缺陷查询（async）"""

    def test_returns_records(self):
        result = asyncio.run(mes_query_defects(defect_type=None, days_back=30, limit=50))
        assert result["total"] >= 1
        assert len(result["records"]) >= 1
        assert result["_note"] == "模拟数据"

    def test_filter_metadata(self):
        """返回结果应包含 filter 字段"""
        result = asyncio.run(mes_query_defects(defect_type="hardness_low", days_back=7, limit=10))
        assert result["filter"]["defect_type"] == "hardness_low"
        assert result["filter"]["days_back"] == 7
        assert result["filter"]["limit"] == 10

    def test_record_has_required_fields(self):
        """每条记录应包含必要字段"""
        result = asyncio.run(mes_query_defects(defect_type=None, days_back=30, limit=5))
        for r in result["records"]:
            assert "record_id" in r
            assert "batch_id" in r
            assert "defect_type" in r
            assert "measured_value" in r
            assert "root_cause" in r
            assert "created_at" in r


class TestMesSubmitAdjustment:
    """测试 MES 提交调整建议（async）"""

    def test_returns_proposal_id(self):
        """应返回 proposal_id"""
        result = asyncio.run(submit_adjustment(
            batch_id="B20260701-A",
            adjustments={"holding_time": "+30"},
            reason="保温时间不足",
        ))
        assert "proposal_id" in result
        assert result["proposal_id"].startswith("PA-")

    def test_status_pending_review(self):
        """提交后状态应为 pending_review"""
        result = asyncio.run(submit_adjustment(
            batch_id="B-X",
            adjustments={"temperature": "+10"},
            reason="温度偏低",
        ))
        assert result["status"] == "pending_review"

    def test_preserves_adjustments(self):
        """adjustments 应原样保留"""
        adj = {"holding_time": "+50", "temperature": "+5"}
        result = asyncio.run(submit_adjustment(
            batch_id="B-Y",
            adjustments=adj,
            reason="",
        ))
        assert result["adjustments"] == adj

    def test_timestamp_present(self):
        """应包含 submitted_at"""
        result = asyncio.run(submit_adjustment(
            batch_id="B-Z",
            adjustments={},
            reason="test",
        ))
        assert "submitted_at" in result
        assert "T" in result["submitted_at"]  # ISO 格式


# ===== Metallurgy Server =====
class TestMetallurgyJmakModel:
    """测试 JMAK 模型"""

    def test_returns_transformation_fraction(self):
        r = jmak_model(850, 120)
        assert r["model"] == "JMAK"
        assert "transformation_fraction" in r["outputs"]
        assert 0 <= r["outputs"]["transformation_fraction"] <= 1

    def test_returns_predicted_hardness(self):
        r = jmak_model(850, 120)
        assert "predicted_hardness_HRc" in r["outputs"]
        assert r["outputs"]["predicted_hardness_HRc"] > 0

    def test_low_temp_lower_fraction(self):
        """温度低 → 相变分数低"""
        high = jmak_model(850, 120)
        low = jmak_model(750, 120)
        assert low["outputs"]["transformation_fraction"] < high["outputs"]["transformation_fraction"]

    def test_longer_time_higher_fraction(self):
        """保温时间越长 → 相变分数越高"""
        short = jmak_model(850, 30)
        long_ = jmak_model(850, 240)
        assert long_["outputs"]["transformation_fraction"] > short["outputs"]["transformation_fraction"]

    def test_temp_factor_clamped(self):
        """温度低于 700℃ 时 temp_factor 应被钳制到 0.1"""
        r = jmak_model(500, 120)  # (500-700)/200 < 0 → clamped to 0.1
        assert r["parameters"]["temp_factor"] == 0.1

    def test_inputs_echoed(self):
        r = jmak_model(845, 90)
        assert r["inputs"]["temperature"] == 845
        assert r["inputs"]["holding_time"] == 90

    def test_parameters_included(self):
        r = jmak_model(845, 90)
        assert "k" in r["parameters"]
        assert "n" in r["parameters"]


class TestMetallurgyHallPetchModel:
    """测试 Hall-Petch 模型"""

    def test_returns_yield_strength(self):
        r = hall_petch_model(20.0)
        assert r["model"] == "Hall-Petch"
        assert r["outputs"]["yield_strength_MPa"] > 0

    def test_returns_estimated_hardness(self):
        r = hall_petch_model(20.0)
        assert r["outputs"]["estimated_hardness_HV"] > 0

    def test_smaller_grain_higher_strength(self):
        """晶粒越细 → 强度越高"""
        coarse = hall_petch_model(50.0)
        fine = hall_petch_model(5.0)
        assert fine["outputs"]["yield_strength_MPa"] > coarse["outputs"]["yield_strength_MPa"]

    def test_inputs_echoed(self):
        r = hall_petch_model(15.0)
        assert r["inputs"]["grain_size_um"] == 15.0


class TestMetallurgyCoolingRateModel:
    """测试冷却速率模型"""

    def test_returns_hardness(self):
        r = cooling_rate_model(5.0)
        assert r["model"] == "cooling_rate"
        assert "estimated_hardness_HRc" in r["outputs"]

    def test_faster_cooling_higher_hardness(self):
        """冷却速率越快 → 硬度越高"""
        slow = cooling_rate_model(2.0)
        fast = cooling_rate_model(10.0)
        assert fast["outputs"]["estimated_hardness_HRc"] > slow["outputs"]["estimated_hardness_HRc"]

    def test_inputs_echoed(self):
        r = cooling_rate_model(7.5)
        assert r["inputs"]["cooling_rate"] == 7.5


class TestMetallurgyRunModel:
    """测试 run_metallurgy_model 分发函数

    Note: 该函数签名要求所有参数显式传入（无默认值），
          缺失的参数传 None。
    """

    def test_jmak_dispatch(self):
        r = run_metallurgy_model("jmak", temperature=850, holding_time=120, cooling_rate=None, grain_size=None)
        assert r["model"] == "JMAK"

    def test_hall_petch_dispatch(self):
        r = run_metallurgy_model("hall_petch", temperature=None, holding_time=None, cooling_rate=None, grain_size=20.0)
        assert r["model"] == "Hall-Petch"

    def test_cooling_rate_dispatch(self):
        r = run_metallurgy_model("cooling_rate", temperature=None, holding_time=None, cooling_rate=5.0, grain_size=None)
        assert r["model"] == "cooling_rate"

    def test_jmak_missing_temp(self):
        r = run_metallurgy_model("jmak", temperature=None, holding_time=120, cooling_rate=None, grain_size=None)
        assert "error" in r

    def test_jmak_missing_holding_time(self):
        r = run_metallurgy_model("jmak", temperature=850, holding_time=None, cooling_rate=None, grain_size=None)
        assert "error" in r

    def test_hall_petch_missing_grain_size(self):
        r = run_metallurgy_model("hall_petch", temperature=None, holding_time=None, cooling_rate=None, grain_size=None)
        assert "error" in r

    def test_cooling_rate_missing_param(self):
        r = run_metallurgy_model("cooling_rate", temperature=None, holding_time=None, cooling_rate=None, grain_size=None)
        assert "error" in r

    def test_unknown_model_type(self):
        r = run_metallurgy_model("nonexistent", temperature=None, holding_time=None, cooling_rate=None, grain_size=None)
        assert "error" in r
        assert "nonexistent" in r["error"]


class TestMetallurgyValidateHypothesis:
    """测试假设验证"""

    def test_inconclusive_when_no_deviation(self):
        """无偏差时 verdict 应为 inconclusive"""
        r = validate_hypothesis(
            hypothesis="保温时间不足",
            actual_params={"holding_time": 120},
            measured_value=58.0,
            standard_value=58.0,
        )
        assert r["verdict"] == "inconclusive"
        assert r["confidence"] == 0.5

    def test_supported_when_holding_time_low(self):
        """保温时间 < 60 → supported"""
        r = validate_hypothesis(
            hypothesis="保温时间不足",
            actual_params={"holding_time": 50},
            measured_value=52.0,
            standard_value=58.0,
        )
        assert r["verdict"] == "supported"
        assert r["confidence"] >= 0.7
        assert any("保温时间" in e for e in r["evidence"])

    def test_supported_when_cooling_rate_low(self):
        """冷却速率 < 1.0 → supported"""
        r = validate_hypothesis(
            hypothesis="冷却速率过低",
            actual_params={"cooling_rate": 0.8},
            measured_value=53.0,
            standard_value=58.0,
        )
        assert r["verdict"] == "supported"
        assert r["confidence"] >= 0.75

    def test_evidence_contains_measured_value(self):
        """evidence 应包含实测值信息"""
        r = validate_hypothesis(
            hypothesis="保温不足",
            actual_params={"holding_time": 40},
            measured_value=51.5,
            standard_value=58.0,
        )
        evidence_text = " ".join(r["evidence"])
        assert "51.5" in evidence_text
        assert "58.0" in evidence_text

    def test_preserves_hypothesis(self):
        """hypothesis 字段应原样保留"""
        r = validate_hypothesis(
            hypothesis="温度偏低导致硬度不足",
            actual_params={},
            measured_value=None,
            standard_value=None,
        )
        assert r["hypothesis"] == "温度偏低导致硬度不足"

    def test_no_measured_value_inconclusive(self):
        """无 measured_value 时应 inconclusive"""
        r = validate_hypothesis(
            hypothesis="保温不足",
            actual_params={"holding_time": 30},
            measured_value=None,
            standard_value=None,
        )
        assert r["verdict"] == "inconclusive"


# ===== Knowledge Server（依赖 Chroma，部分会 skip）=====
class TestKnowledgeServer:
    """测试 Knowledge MCP Server 的工具函数

    Note: knowledge_server.py 的工具函数都通过 MemoryService 调用 Chroma，
    Chroma 不可用时返回空列表或抛异常。这里只做最小验证。
    """

    def test_memory_service_can_init(self):
        """MemoryService 应可初始化（即使 Chroma 不可用）"""
        from mcp_servers.knowledge_server import get_memory
        m = get_memory()
        assert m is not None
        assert m.collection_name == "metacraft_cases"

    def test_search_handbook_returns_dict(self):
        """search_handbook 应返回 dict（即使空结果）"""
        from mcp_servers.knowledge_server import get_memory
        m = get_memory()
        # 直接调底层方法
        try:
            results = m.search_semantic(query="保温时间", top_k=3)
            assert isinstance(results, list)
        except Exception as e:
            # Chroma 不可用时应有友好降级
            pytest.skip(f"Chroma 不可用：{e}")


# ===== MCP 工具列表 =====
class TestMcpToolLists:
    """测试 list_tools() 返回的工具定义"""

    def test_mes_lists_3_tools(self):
        from mcp_servers.mes_server import list_tools
        tools = asyncio.run(list_tools())
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"query_batch_params", "query_defect_history", "submit_adjustment"}

    def test_metallurgy_lists_2_tools(self):
        from mcp_servers.metallurgy_server import list_tools
        tools = asyncio.run(list_tools())
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"run_metallurgy_model", "validate_hypothesis"}

    def test_knowledge_lists_4_tools(self):
        from mcp_servers.knowledge_server import list_tools
        tools = asyncio.run(list_tools())
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"search_handbook", "search_cases", "write_case", "update_case_confidence"}

    def test_tools_have_input_schema(self):
        """每个工具都应有 inputSchema"""
        from mcp_servers.mes_server import list_tools as mes_list
        from mcp_servers.metallurgy_server import list_tools as met_list
        from mcp_servers.knowledge_server import list_tools as kb_list

        for tools in [asyncio.run(mes_list()), asyncio.run(met_list()), asyncio.run(kb_list())]:
            for t in tools:
                assert t.inputSchema is not None
                assert t.inputSchema.get("type") == "object"

    def test_required_fields_declared(self):
        """必填字段应在 required 中声明"""
        from mcp_servers.mes_server import list_tools as mes_list
        tools = asyncio.run(mes_list())
        # submit_adjustment 必填 batch_id + adjustments
        sa = next(t for t in tools if t.name == "submit_adjustment")
        assert "batch_id" in sa.inputSchema["required"]
        assert "adjustments" in sa.inputSchema["required"]
        # query_batch_params 必填 batch_id
        qb = next(t for t in tools if t.name == "query_batch_params")
        assert qb.inputSchema["required"] == ["batch_id"]
