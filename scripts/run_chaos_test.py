"""
M4-13 故障演练：模拟 LLM / 向量库 / SQLite 故障，验证系统降级行为

故障注入方式：monkeypatch（运行时替换，场景结束自动恢复）
演练维度：状态码、降级是否生效、响应时间、错误信息

故障场景：
  - llm_failure:      LLM 服务不可用（get_llm 抛异常，验证各 Agent 规则降级）
  - chroma_failure:   向量库不可用（collection=None，验证记忆层降级返回空）
  - sqlite_failure:   SQLite 不可用（execute 抛异常，验证是否有容错）
  - combined_failure: LLM + Chroma 同时故障（验证多故障叠加降级）

MES 说明：当前 MES 为 mock 模式（无真实接入），不构成单点故障，报告标注 N/A。

测试端点：
  - GET  /api/v1/auth/permissions     权限查询（对照，不依赖外部资源）
  - GET  /api/v1/cases                案例查询（读 SQLite + Chroma）
  - GET  /api/v1/dashboard/overview   跨产线看板（读 SQLite + Chroma 全量扫描）
  - POST /api/v1/feedback             反馈写入（写 SQLite）
  - POST /api/v1/analyze              归因分析（调 LLM + 全链路）

用法：
    python scripts/run_chaos_test.py
    python scripts/run_chaos_test.py --scenarios llm_failure,chroma_failure

输出：
    data/chaos_test_report.json  结构化演练数据
    data/chaos_test_report.md    人类可读报告
"""
import argparse
import json
import logging
import sqlite3
import sys
import time
import unittest.mock
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 抑制所有日志输出，避免 sandbox 缓冲区溢出导致进程被 kill
logging.disable(logging.CRITICAL)
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except Exception:
    pass


# ===== 端点配置 =====

ENDPOINTS = {
    "auth": {
        "method": "GET",
        "path": "/api/v1/auth/permissions",
        "params": {"user_id": "admin"},
        "json": None,
        "description": "权限查询（轻量，对照）",
        "depends_on": [],  # 不依赖外部故障资源
    },
    "cases": {
        "method": "GET",
        "path": "/api/v1/cases",
        "params": {"user_id": "admin", "limit": 10},
        "json": None,
        "description": "案例查询（读 SQLite + Chroma）",
        "depends_on": ["sqlite", "chroma"],
    },
    "dashboard": {
        "method": "GET",
        "path": "/api/v1/dashboard/overview",
        "params": {"user_id": "admin", "days": 30},
        "json": None,
        "description": "跨产线看板（读 SQLite + Chroma 全量扫描）",
        "depends_on": ["sqlite", "chroma"],
    },
    "feedback": {
        "method": "POST",
        "path": "/api/v1/feedback",
        "params": None,
        "json": {
            "proposal_id": "chaos_test_proposal",
            "user_id": "admin",
            "action": "adopted",
            "score": 0.8,
            "comment": "故障演练反馈",
            "line_id": "heat_treatment",
        },
        "description": "反馈写入（写 SQLite）",
        "depends_on": ["sqlite"],
    },
    "analyze": {
        "method": "POST",
        "path": "/api/v1/analyze",
        "params": None,
        "json": {
            "query": "请分析此批次的缺陷原因",
            "batch_id": "B20250101-001",
            "defect_type": "硬度偏低",
            "measured_value": 55.0,
            "standard_value": 58.0,
            "line_id": "heat_treatment",
            "user_id": "admin",
        },
        "description": "归因分析（调 LLM + 全链路）",
        "depends_on": ["llm", "sqlite", "chroma"],
        "requires_llm": True,  # 需要真实 LLM，仅 llm 故障注入时测试（走降级，不调真实 LLM）
    },
}


def _should_test(ep_name, ep_cfg, faults, is_baseline):
    """判断端点在当前场景是否可测（避免基线/非 llm 故障时调用真实 LLM）"""
    if ep_cfg.get("requires_llm"):
        # analyze 仅在 llm 故障注入时测试（此时走降级路径，不调真实 LLM）
        return (not is_baseline) and ("llm" in faults)
    return True


# ===== 故障注入器 =====

# 需要注入 LLM 故障的节点模块（均为 from agent.utils import get_llm）
_LLM_MODULES = [
    "agent.nodes.planner",
    "agent.nodes.decision_agent",
    "agent.nodes.executor",
    "agent.nodes.interaction_agent",
    "agent.nodes.mechanism_agent",
    "agent.nodes.reflector",
    "agent.nodes.review_agent",
]


def _failing_get_llm(*args, **kwargs):
    """故障版 get_llm：直接抛异常，模拟 LLM 服务不可用"""
    raise ConnectionError("LLM 服务不可用（故障注入）")


def inject_llm_failure():
    """注入 LLM 故障：替换所有节点模块的 get_llm"""
    import importlib
    originals = {}
    for mod_name in _LLM_MODULES:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "get_llm"):
            originals[mod_name] = mod.get_llm
            mod.get_llm = _failing_get_llm
    return originals


def restore_llm_failure(originals):
    """恢复 LLM 故障注入"""
    import importlib
    for mod_name, original in originals.items():
        mod = importlib.import_module(mod_name)
        mod.get_llm = original


def inject_chroma_failure(memory_service):
    """注入 Chroma 故障：置空 collection，所有语义方法降级返回空"""
    original_collection = memory_service._collection
    original_client = memory_service._chroma_client
    memory_service._collection = None
    memory_service._chroma_client = None
    return (original_collection, original_client)


def restore_chroma_failure(memory_service, state):
    """恢复 Chroma 故障注入"""
    memory_service._collection = state[0]
    memory_service._chroma_client = state[1]


def inject_sqlite_failure(memory_service):
    """注入 SQLite 故障：mock db.execute 抛异常"""
    original_db = memory_service.db
    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked（故障注入）")
    mock_db.commit.side_effect = sqlite3.OperationalError("database is locked（故障注入）")
    memory_service.db = mock_db
    return original_db


def restore_sqlite_failure(memory_service, original_db):
    """恢复 SQLite 故障注入"""
    memory_service.db = original_db


# ===== 故障场景定义 =====

SCENARIOS = {
    "llm_failure": {
        "description": "LLM 服务不可用",
        "faults": ["llm"],
        "inject": lambda mem: inject_llm_failure(),
        "restore": lambda mem, state: restore_llm_failure(state),
    },
    "chroma_failure": {
        "description": "向量库（Chroma）不可用",
        "faults": ["chroma"],
        "inject": lambda mem: inject_chroma_failure(mem),
        "restore": lambda mem, state: restore_chroma_failure(mem, state),
    },
    "sqlite_failure": {
        "description": "SQLite 数据库不可用",
        "faults": ["sqlite"],
        "inject": lambda mem: inject_sqlite_failure(mem),
        "restore": lambda mem, state: restore_sqlite_failure(mem, state),
    },
    "combined_failure": {
        "description": "LLM + Chroma 同时故障",
        "faults": ["llm", "chroma"],
        "inject": lambda mem: (inject_llm_failure(), inject_chroma_failure(mem)),
        "restore": lambda mem, state: (
            restore_llm_failure(state[0]),
            restore_chroma_failure(mem, state[1]),
        ),
    },
}


# ===== 演练执行 =====

def _send_request(client, ep_cfg):
    """发送单个请求，返回 (status_code, elapsed_ms, body_snippet, error)"""
    start = time.perf_counter()
    try:
        if ep_cfg["method"] == "GET":
            resp = client.get(ep_cfg["path"], params=ep_cfg["params"])
        else:
            resp = client.post(ep_cfg["path"], json=ep_cfg["json"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        snippet = json.dumps(body, ensure_ascii=False)[:300]
        return resp.status_code, elapsed_ms, snippet, None
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return -1, elapsed_ms, "", str(e)


def _classify_degradation(ep_name, faults, status_code, body_snippet):
    """判断降级是否生效

    判定规则：
    - 故障未影响该端点依赖 → 'not_applicable'（对照）
    - 2xx 且故障命中依赖 → 'degraded_ok'（降级生效）
    - 5xx 且故障命中依赖 → 'failed'（无降级，直接报错）
    - 2xx 且无故障命中 → 'normal'
    """
    depends = ENDPOINTS[ep_name]["depends_on"]
    fault_hit = any(dep in faults for dep in depends)

    if not fault_hit:
        return "not_applicable" if status_code == -1 else "normal"
    if 200 <= status_code < 300:
        return "degraded_ok"
    if status_code >= 500 or status_code == -1:
        return "failed"
    return "degraded_ok"  # 4xx 视为降级（如权限问题，非故障导致）


def run_chaos_test(scenario_keys=None, repeats=1):
    """执行故障演练矩阵

    Args:
        scenario_keys: 指定场景列表，None 则全跑
        repeats: 每个端点重复次数（取平均响应时间）

    Returns:
        dict: 演练报告
    """
    from fastapi.testclient import TestClient
    import api.routes

    app = api.routes.app
    memory = api.routes.memory

    # SQLite 改为多线程安全（TestClient 内部线程池）
    try:
        memory.db.close()
        memory.db = sqlite3.connect(str(memory.db_path), check_same_thread=False)
    except Exception:
        pass

    if scenario_keys is None:
        scenario_keys = list(SCENARIOS.keys())

    client = TestClient(app)
    report = {
        "test_time": datetime.now().isoformat(),
        "config": {
            "scenarios": scenario_keys,
            "repeats": repeats,
            "endpoints": list(ENDPOINTS.keys()),
        },
        "baseline": {},  # 无故障基线
        "results": {},   # 各场景结果
        "mes_note": "MES 当前为 mock 模式（无真实接入），不构成单点故障，未单独演练",
    }

    # 1. 基线测试（无故障）
    baseline = {}
    for ep_name, ep_cfg in ENDPOINTS.items():
        if not _should_test(ep_name, ep_cfg, [], is_baseline=True):
            baseline[ep_name] = {
                "description": ep_cfg["description"],
                "status_codes": [],
                "avg_latency_ms": 0,
                "body_snippet": "",
                "error": None,
                "degradation": "skipped",
            }
            continue
        latencies = []
        statuses = []
        last_snippet = ""
        last_error = None
        for _ in range(repeats):
            sc, ms, snippet, err = _send_request(client, ep_cfg)
            latencies.append(ms)
            statuses.append(sc)
            last_snippet = snippet
            last_error = err
        baseline[ep_name] = {
            "description": ep_cfg["description"],
            "status_codes": statuses,
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
            "body_snippet": last_snippet,
            "error": last_error,
            "degradation": "baseline",
        }
    report["baseline"] = baseline

    # 2. 各故障场景测试
    for sc_name in scenario_keys:
        sc_cfg = SCENARIOS[sc_name]
        faults = sc_cfg["faults"]

        # 注入故障
        fault_state = sc_cfg["inject"](memory)

        scenario_result = {
            "description": sc_cfg["description"],
            "faults": faults,
            "endpoints": {},
        }
        try:
            for ep_name, ep_cfg in ENDPOINTS.items():
                if not _should_test(ep_name, ep_cfg, faults, is_baseline=False):
                    scenario_result["endpoints"][ep_name] = {
                        "description": ep_cfg["description"],
                        "status_codes": [],
                        "avg_latency_ms": 0,
                        "body_snippet": "",
                        "error": None,
                        "degradation": "skipped",
                    }
                    continue
                latencies = []
                statuses = []
                last_snippet = ""
                last_error = None
                for _ in range(repeats):
                    sc, ms, snippet, err = _send_request(client, ep_cfg)
                    latencies.append(ms)
                    statuses.append(sc)
                    last_snippet = snippet
                    last_error = err

                degradation = _classify_degradation(
                    ep_name, faults, statuses[-1] if statuses else -1, last_snippet
                )
                scenario_result["endpoints"][ep_name] = {
                    "description": ep_cfg["description"],
                    "status_codes": statuses,
                    "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
                    "body_snippet": last_snippet,
                    "error": last_error,
                    "degradation": degradation,
                }
        finally:
            # 恢复故障
            sc_cfg["restore"](memory, fault_state)

        report["results"][sc_name] = scenario_result

    return report


# ===== 报告生成 =====

def generate_markdown_report(report: dict) -> str:
    """生成 Markdown 演练报告"""
    lines = []
    lines.append("# M4-13 故障演练报告\n")
    lines.append(f"**测试时间**：{report['test_time']}")
    lines.append(f"**演练场景**：{', '.join(report['config']['scenarios'])}")
    lines.append(f"**测试端点**：{', '.join(report['config']['endpoints'])}")
    lines.append(f"**每端点重复**：{report['config']['repeats']} 次\n")

    # MES 说明
    lines.append(f"**MES 说明**：{report['mes_note']}\n")

    # 1. 基线
    lines.append("## 1. 基线（无故障）\n")
    lines.append("| 端点 | 描述 | 状态码 | 平均耗时(ms) | 降级 |")
    lines.append("|------|------|--------|-------------|------|")
    for ep, data in report["baseline"].items():
        sc = data["status_codes"][-1] if data["status_codes"] else "N/A"
        lines.append(
            f"| {ep} | {data['description']} | {sc} | {data['avg_latency_ms']} | {data['degradation']} |"
        )

    # 2. 各场景
    for sc_name, sc_data in report["results"].items():
        lines.append(f"\n## 2. 场景：{sc_name}（{sc_data['description']}）\n")
        lines.append(f"**注入故障**：{', '.join(sc_data['faults'])}\n")
        lines.append("| 端点 | 描述 | 状态码 | 平均耗时(ms) | 降级判定 | 错误信息 |")
        lines.append("|------|------|--------|-------------|---------|---------|")
        for ep, data in sc_data["endpoints"].items():
            sc = data["status_codes"][-1] if data["status_codes"] else "N/A"
            err = (data["error"] or "")[:60]
            lines.append(
                f"| {ep} | {data['description']} | {sc} | {data['avg_latency_ms']} | {data['degradation']} | {err} |"
            )

    # 3. 降级汇总
    lines.append("\n## 3. 降级汇总\n")
    lines.append("| 场景 | 端点 | 依赖故障 | 降级判定 |")
    lines.append("|------|------|---------|---------|")
    for sc_name, sc_data in report["results"].items():
        faults = set(sc_data["faults"])
        for ep, data in sc_data["endpoints"].items():
            depends = set(ENDPOINTS[ep]["depends_on"])
            hit = "是" if depends & faults else "否"
            lines.append(f"| {sc_name} | {ep} | {hit} | {data['degradation']} |")

    # 4. 验收
    lines.append("\n## 4. 验收标准对照\n")
    lines.append("| 验收项 | 标准 | 结果 |")
    lines.append("|--------|------|------|")

    # 统计：故障命中依赖的端点中，降级生效的比例
    hit_total = 0
    hit_degraded_ok = 0
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
                    hit_degraded_ok += 1
                elif data["degradation"] == "failed":
                    hit_failed.append(f"{sc_name}/{ep}")

    rate = hit_degraded_ok / max(hit_total, 1)
    verdict = "✅ 达标" if rate >= 0.9 else "❌ 未达标"
    lines.append(
        f"| 故障命中端点降级率 | ≥ 90% | {verdict}（{hit_degraded_ok}/{hit_total} = {rate:.1%}） |"
    )

    # 5. 弱点清单
    lines.append("\n## 5. 弱点清单（故障命中但未降级）\n")
    if hit_failed:
        lines.append("| 场景 | 端点 | 说明 |")
        lines.append("|------|------|------|")
        for item in hit_failed:
            sc, ep = item.split("/")
            lines.append(f"| {sc} | {ep} | 故障命中依赖但返回 5xx，无降级 |")
        lines.append("\n> 以上弱点需在 M4-14 降级策略中补齐（加 try/except + 降级返回）。")
    else:
        lines.append("无弱点，所有故障命中端点均成功降级。")

    # 6. 结论
    lines.append("\n## 6. 结论\n")
    lines.append(f"- 故障命中端点降级率：{rate:.1%}（{hit_degraded_ok}/{hit_total}）")
    lines.append(f"- 未降级弱点数：{len(hit_failed)}")
    if hit_failed:
        lines.append(f"- 弱点：{', '.join(hit_failed)}")
        lines.append("- 建议：M4-14 对弱点端点补齐 try/except + 降级返回 + 全局异常处理。")
    else:
        lines.append("- 所有故障命中端点均降级生效，系统具备基本的故障容错能力。")

    return "\n".join(lines) + "\n"


# ===== 主入口 =====

def main():
    parser = argparse.ArgumentParser(description="M4-13 故障演练")
    parser.add_argument("--scenarios", default=None, help="场景列表（逗号分隔），默认全跑")
    parser.add_argument("--repeats", type=int, default=1, help="每端点重复次数")
    parser.add_argument("--output", default="data/chaos_test_report", help="输出文件前缀")
    args = parser.parse_args()

    scenario_keys = args.scenarios.split(",") if args.scenarios else None

    print("[INFO] 开始故障演练...", file=sys.stderr)
    report = run_chaos_test(scenario_keys=scenario_keys, repeats=args.repeats)

    # 保存 JSON
    json_path = PROJECT_ROOT / f"{args.output}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON 报告: {json_path}", file=sys.stderr)

    # 保存 Markdown
    md_path = PROJECT_ROOT / f"{args.output}.md"
    md = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] Markdown 报告: {md_path}", file=sys.stderr)

    # 清理演练产生的反馈数据
    try:
        import api.routes
        api.routes.memory.db.execute(
            "DELETE FROM feedback WHERE comment = '故障演练反馈'"
        )
        api.routes.memory.db.commit()
        print("[OK] 已清理演练反馈数据", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] 清理反馈数据失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
