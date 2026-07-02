"""
M4-13 故障演练测试

验证：
1. 故障注入/恢复函数正确工作
2. 降级判定逻辑正确
3. 完整演练能生成报告
"""
import json
import os
import sys
import sqlite3
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_chaos_test import (
    ENDPOINTS,
    SCENARIOS,
    _classify_degradation,
    inject_llm_failure,
    restore_llm_failure,
    inject_chroma_failure,
    restore_chroma_failure,
    inject_sqlite_failure,
    restore_sqlite_failure,
    run_chaos_test,
    generate_markdown_report,
)


# ===== 1. 脚本可导入性 =====

def test_chaos_script_importable():
    """脚本可正常导入"""
    import scripts.run_chaos_test as mod
    assert hasattr(mod, "run_chaos_test")
    assert hasattr(mod, "generate_markdown_report")
    assert len(SCENARIOS) == 4
    assert len(ENDPOINTS) == 5


# ===== 2. 故障注入/恢复 =====

def test_inject_restore_llm_failure():
    """LLM 故障注入后 get_llm 抛异常，恢复后正常"""
    import importlib
    planner = importlib.import_module("agent.nodes.planner")
    original = planner.get_llm

    state = inject_llm_failure()
    try:
        # 注入后调用应抛异常
        with pytest.raises(ConnectionError, match="LLM 服务不可用"):
            planner.get_llm("planner")
        # 确认多个模块都被注入
        decision = importlib.import_module("agent.nodes.decision_agent")
        with pytest.raises(ConnectionError):
            decision.get_llm("decision")
    finally:
        restore_llm_failure(state)

    # 恢复后应能正常调用（不抛 ConnectionError）
    assert planner.get_llm is original


def test_inject_restore_chroma_failure():
    """Chroma 故障注入后 collection=None，恢复后还原"""
    from api.routes import memory
    original_collection = memory._collection

    state = inject_chroma_failure(memory)
    try:
        assert memory._collection is None
        assert memory._chroma_client is None
        # search_semantic 应降级返回空列表
        result = memory.search_semantic("test query", top_k=3)
        assert result == []
    finally:
        restore_chroma_failure(memory, state)

    assert memory._collection is original_collection


def test_inject_restore_sqlite_failure():
    """SQLite 故障注入后 execute 抛异常，恢复后正常"""
    from api.routes import memory
    original_db = memory.db

    state = inject_sqlite_failure(memory)
    try:
        assert memory.db is not original_db
        # execute 应抛异常
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            memory.db.execute("SELECT 1")
    finally:
        restore_sqlite_failure(memory, state)

    assert memory.db is original_db
    # 恢复后应能正常查询
    memory.db.execute("SELECT 1").fetchone()


# ===== 3. 降级判定逻辑 =====

def test_classify_degradation_not_applicable():
    """故障未命中依赖 → normal/not_applicable"""
    # auth 不依赖任何故障资源，llm 故障不影响
    assert _classify_degradation("auth", ["llm"], 200, "") == "normal"
    assert _classify_degradation("auth", ["llm"], -1, "") == "not_applicable"


def test_classify_degraded_ok():
    """故障命中依赖但返回 2xx → degraded_ok"""
    # cases 依赖 sqlite+chroma，sqlite 故障命中
    assert _classify_degradation("cases", ["sqlite"], 200, "") == "degraded_ok"
    # analyze 依赖 llm，llm 故障命中
    assert _classify_degradation("analyze", ["llm"], 200, "") == "degraded_ok"


def test_classify_failed():
    """故障命中依赖且返回 5xx → failed"""
    assert _classify_degradation("cases", ["sqlite"], 500, "") == "failed"
    assert _classify_degradation("dashboard", ["chroma"], 500, "") == "failed"
    assert _classify_degradation("feedback", ["sqlite"], -1, "") == "failed"


def test_classify_combined():
    """组合故障判定"""
    # analyze 依赖 llm+sqlite+chroma，combined(llm+chroma) 命中
    assert _classify_degradation("analyze", ["llm", "chroma"], 200, "") == "degraded_ok"
    assert _classify_degradation("analyze", ["llm", "chroma"], 500, "") == "failed"
    # auth 不依赖，combined 不影响
    assert _classify_degradation("auth", ["llm", "chroma"], 200, "") == "normal"


# ===== 4. 小规模演练 =====

def test_run_chaos_test_small():
    """小规模演练（仅 chroma_failure 场景，仅 auth+cases 端点）"""
    # 临时限制端点
    original_endpoints = ENDPOINTS.copy()
    try:
        ENDPOINTS.clear()
        ENDPOINTS["auth"] = original_endpoints["auth"]
        ENDPOINTS["cases"] = original_endpoints["cases"]
        report = run_chaos_test(scenario_keys=["chroma_failure"], repeats=1)

        assert "baseline" in report
        assert "results" in report
        assert "chroma_failure" in report["results"]
        assert "auth" in report["results"]["chroma_failure"]["endpoints"]
        assert "cases" in report["results"]["chroma_failure"]["endpoints"]

        # auth 不依赖 chroma，应正常
        auth_data = report["results"]["chroma_failure"]["endpoints"]["auth"]
        assert auth_data["degradation"] in ("normal", "degraded_ok")
        # cases 依赖 chroma，应降级（Chroma 已有降级返回空）
        cases_data = report["results"]["chroma_failure"]["endpoints"]["cases"]
        assert cases_data["degradation"] in ("degraded_ok", "failed")
    finally:
        ENDPOINTS.clear()
        ENDPOINTS.update(original_endpoints)


# ===== 5. 报告生成 =====

def test_generate_markdown_report():
    """Markdown 报告生成"""
    fake_report = {
        "test_time": "2026-07-03T00:00:00",
        "config": {
            "scenarios": ["llm_failure"],
            "repeats": 1,
            "endpoints": ["auth", "cases"],
        },
        "mes_note": "MES mock 模式",
        "baseline": {
            "auth": {
                "description": "权限查询",
                "status_codes": [200],
                "avg_latency_ms": 10.0,
                "body_snippet": "{}",
                "error": None,
                "degradation": "baseline",
            },
        },
        "results": {
            "llm_failure": {
                "description": "LLM 故障",
                "faults": ["llm"],
                "endpoints": {
                    "cases": {
                        "description": "案例查询",
                        "status_codes": [200],
                        "avg_latency_ms": 50.0,
                        "body_snippet": "[]",
                        "error": None,
                        "degradation": "normal",
                    },
                    "analyze": {
                        "description": "归因分析",
                        "status_codes": [200],
                        "avg_latency_ms": 3000.0,
                        "body_snippet": "{}",
                        "error": None,
                        "degradation": "degraded_ok",
                    },
                },
            }
        },
    }
    md = generate_markdown_report(fake_report)
    assert "# M4-13 故障演练报告" in md
    assert "基线" in md
    assert "llm_failure" in md
    assert "降级汇总" in md
    assert "验收标准" in md


# ===== 6. 完整演练 + 报告生成 =====

def test_full_chaos_test_and_generate_report():
    """完整故障演练（4 场景 x 5 端点）+ 生成 JSON/Markdown 报告

    这是 M4-13 的核心验收测试，生成 data/chaos_test_report.{json,md}
    """
    report = run_chaos_test(repeats=1)

    # 验证报告结构
    assert "baseline" in report
    assert "results" in report
    assert len(report["results"]) == 4  # 4 个场景
    for sc in SCENARIOS:
        assert sc in report["results"]
        assert len(report["results"][sc]["endpoints"]) == 5  # 5 个端点（含 skipped）

    # 验证基线（analyze 基线 skipped，其余 4 个 baseline）
    assert len(report["baseline"]) == 5
    assert report["baseline"]["analyze"]["degradation"] == "skipped"

    # 生成 Markdown 报告
    md = generate_markdown_report(report)
    assert "故障演练报告" in md
    assert "弱点清单" in md

    # 保存报告
    json_path = PROJECT_ROOT / "data" / "chaos_test_report.json"
    md_path = PROJECT_ROOT / "data" / "chaos_test_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    assert json_path.exists()
    assert md_path.exists()

    # 清理演练反馈数据
    try:
        from api.routes import memory
        memory.db.execute("DELETE FROM feedback WHERE comment = '故障演练反馈'")
        memory.db.commit()
    except Exception:
        pass

    # 打印关键结论供调试
    import sys
    hit_total = 0
    hit_ok = 0
    hit_failed = []
    for sc_name, sc_data in report["results"].items():
        faults = set(sc_data["faults"])
        for ep, data in sc_data["endpoints"].items():
            if data["degradation"] == "skipped":
                continue
            depends = set(ENDPOINTS[ep]["depends_on"])
            if depends & faults:
                hit_total += 1
                if data["degradation"] == "degraded_ok":
                    hit_ok += 1
                elif data["degradation"] == "failed":
                    hit_failed.append(f"{sc_name}/{ep}")
    sys.stderr.write(
        f"\n[M4-13] 降级率: {hit_ok}/{hit_total} = {hit_ok/max(hit_total,1):.1%}, "
        f"弱点: {hit_failed}\n"
    )
