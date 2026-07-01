"""
Agent 工具函数单元测试

测试内容：
- query_batch_params: 批次参数查询（mock/seed/default 三级）
- query_defect_history: 历史缺陷查询
- run_metallurgy_model: 机理模型（JMAK/冷却速率/Hall-Petch）
- search_handbook: 手册检索
- search_cases: 案例检索
- call_tool: 统一调用入口
"""
import pytest

from agent.tools import (
    TOOL_REGISTRY,
    _cooling_rate_model,
    _hall_petch_model,
    _jmak_model,
    call_tool,
    query_batch_params,
    query_defect_history,
    run_metallurgy_model,
    search_cases,
    search_handbook,
)


class TestQueryBatchParams:
    """测试批次参数查询"""

    def test_known_mock_batch(self):
        """内置 mock 批次应返回正确参数"""
        result = query_batch_params("B20260628-A")
        assert result["batch_id"] == "B20260628-A"
        assert result["_source"] == "mock_data"
        assert "temperature" in result
        assert "holding_time" in result
        assert "cooling_rate" in result

    def test_unknown_batch_returns_default(self):
        """未知批次应返回默认模拟数据"""
        result = query_batch_params("B-UNKNOWN-9999")
        assert result["batch_id"] == "B-UNKNOWN-9999"
        assert result["_source"] == "mock_default"
        assert "temperature" in result

    def test_seed_case_batch(self):
        """种子案例批次应从 seed_cases 加载"""
        # B20260701-A 是 seed_cases.json 中的第一条用例
        result = query_batch_params("B20260701-A")
        assert result.get("_source") == "seed_case"
        assert "temperature" in result
        assert "holding_time" in result


class TestQueryDefectHistory:
    """测试历史缺陷查询"""

    def test_returns_all_defects(self):
        """无过滤应返回全部 mock 缺陷"""
        result = query_defect_history()
        assert result["total"] >= 3
        assert len(result["records"]) >= 3
        assert result["_source"] == "mock_data"

    def test_filter_by_defect_type(self):
        """按缺陷类型过滤"""
        result = query_defect_history(defect_type="hardness_low")
        for record in result["records"]:
            assert record["defect_type"] == "hardness_low"

    def test_limit(self):
        """limit 应限制返回条数"""
        result = query_defect_history(limit=1)
        assert len(result["records"]) <= 1


class TestRunMetallurgyModel:
    """测试机理模型"""

    def test_jmak_model(self):
        """JMAK 模型应返回相变分数和预测硬度"""
        result = run_metallurgy_model("jmak", temperature=850, holding_time=120)
        assert result["model"] == "JMAK"
        assert "transformation_fraction" in result["outputs"]
        assert "predicted_hardness_HRc" in result["outputs"]
        assert 0 <= result["outputs"]["transformation_fraction"] <= 1

    def test_jmak_low_temp_lower_fraction(self):
        """温度偏低应导致相变分数下降"""
        high_temp = run_metallurgy_model("jmak", temperature=850, holding_time=120)
        low_temp = run_metallurgy_model("jmak", temperature=800, holding_time=120)
        assert low_temp["outputs"]["transformation_fraction"] < high_temp["outputs"]["transformation_fraction"]

    def test_jmak_missing_params(self):
        """缺少参数应返回错误"""
        result = run_metallurgy_model("jmak", temperature=850)
        assert "error" in result

    def test_cooling_rate_model(self):
        """冷却速率模型应返回估计硬度"""
        result = run_metallurgy_model("cooling_rate", cooling_rate=8.0)
        assert result["model"] == "cooling_rate"
        assert "estimated_hardness_HRc" in result["outputs"]

    def test_cooling_rate_missing_param(self):
        """缺少 cooling_rate 应返回错误"""
        result = run_metallurgy_model("cooling_rate")
        assert "error" in result

    def test_hall_petch_model(self):
        """Hall-Petch 模型应返回屈服强度"""
        result = run_metallurgy_model("hall_petch", grain_size=20.0)
        assert result["model"] == "Hall-Petch"
        assert "yield_strength_MPa" in result["outputs"]

    def test_unknown_model_type(self):
        """未知模型类型应返回错误"""
        result = run_metallurgy_model("nonexistent")
        assert "error" in result


class TestSearchHandbook:
    """测试手册检索（支持索引 / mock 双模式）"""

    def test_returns_results(self):
        """应返回检索结果"""
        result = search_handbook("保温时间")
        assert result["total"] > 0
        assert len(result["results"]) > 0
        assert result["_source"] in ("mock_data", "handbook_index")

    def test_relevance_sorting(self):
        """结果应按相关度排序"""
        result = search_handbook("冷却速率 保温时间")
        scores = [r["relevance_score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self):
        """top_k 应限制返回条数"""
        result = search_handbook("硬度", top_k=1)
        assert len(result["results"]) <= 1

    def test_index_mode_returns_relevant(self):
        """索引模式下应返回高相关度结果"""
        from agent.tools import _load_handbook_index
        index = _load_handbook_index()
        if not index:
            pytest.skip("手册索引未生成，跳过索引模式测试")
        result = search_handbook("硬度偏低 保温时间 冷却速率", top_k=3)
        assert result["_source"] == "handbook_index"
        assert result["total"] > 0
        # top1 应包含至少一个查询关键词
        top1_content = result["results"][0]["content"]
        assert any(kw in top1_content for kw in ["硬度", "保温", "冷却"])

    def test_index_mode_source_label(self):
        """索引模式下来源标签应包含文件名"""
        from agent.tools import _load_handbook_index
        index = _load_handbook_index()
        if not index:
            pytest.skip("手册索引未生成")
        result = search_handbook("淬火温度", top_k=1)
        if result["_source"] == "handbook_index":
            source = result["results"][0]["source"]
            assert ".md" in source or ".pdf" in source or ".docx" in source


class TestSearchCases:
    """测试案例检索"""

    def test_returns_results(self):
        """应返回案例结果"""
        result = search_cases("硬度偏低")
        assert result["total"] > 0
        assert len(result["results"]) > 0

    def test_no_match_returns_fallback(self):
        """无匹配时应返回全部作为兜底"""
        result = search_cases("完全不存在的关键词 xyz123")
        assert len(result["results"]) > 0  # 兜底返回全部

    def test_top_k_limit(self):
        """top_k 应限制返回条数"""
        result = search_cases("硬度", top_k=1)
        assert len(result["results"]) <= 1


class TestCallTool:
    """测试统一调用入口"""

    def test_call_known_tool(self):
        """调用已知工具应成功"""
        result = call_tool("query_batch_params", batch_id="B20260628-A")
        assert result["batch_id"] == "B20260628-A"

    def test_call_unknown_tool(self):
        """调用未知工具应返回错误"""
        result = call_tool("nonexistent_tool")
        assert "error" in result

    def test_tool_registry_completeness(self):
        """工具注册表应包含所有工具"""
        expected_tools = {
            "query_batch_params",
            "query_defect_history",
            "run_metallurgy_model",
            "search_handbook",
            "search_cases",
        }
        assert expected_tools.issubset(set(TOOL_REGISTRY.keys()))


class TestInternalModels:
    """测试内部模型函数"""

    def test_jmak_model_directly(self):
        """直接测试 JMAK 模型函数"""
        result = _jmak_model(850, 120)
        assert result["model"] == "JMAK"
        assert result["inputs"]["temperature"] == 850
        assert result["inputs"]["holding_time"] == 120

    def test_cooling_rate_model_directly(self):
        """直接测试冷却速率模型函数"""
        result = _cooling_rate_model(5.0)
        assert result["model"] == "cooling_rate"
        assert result["inputs"]["cooling_rate"] == 5.0

    def test_hall_petch_model_directly(self):
        """直接测试 Hall-Petch 模型函数"""
        result = _hall_petch_model(15.0)
        assert result["model"] == "Hall-Petch"
        assert result["outputs"]["yield_strength_MPa"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
