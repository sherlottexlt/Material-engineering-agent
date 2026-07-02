"""M4-15: SLA 保障验证脚本

验证 99.5% 可用性目标：
1. 模拟正常流量（health, lines, cases, dashboard, sla/status）
2. 注入 SQLite 故障，触发降级（cases/dashboard/feedback 返回 200+degraded）
3. 查询 /api/v1/sla/status 和 /api/v1/sla/report
4. 验证 availability >= 99.5%

降级机制保证：SQLite 故障时端点返回 200+degraded（非 5xx），availability 不受影响。

用法：
    python scripts/run_sla_report.py
"""
import json
import logging
import sqlite3
import sys
import time
import types
import unittest.mock
from datetime import datetime
from pathlib import Path

# 必须在 import 任何项目模块前抑制日志
logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===== 注入 mock langchain_openai（避免 sandbox import 超时）=====

class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.max_retries = kwargs.get("max_retries", 3)
    def invoke(self, *a, **kw):
        return types.SimpleNamespace(content="mock")
    async def ainvoke(self, *a, **kw):
        return types.SimpleNamespace(content="mock")

_fake = types.ModuleType("langchain_openai")
_fake.ChatOpenAI = _FakeChatOpenAI
sys.modules["langchain_openai"] = _fake


from fastapi.testclient import TestClient
import api.routes
from agent.sla import sla_monitor


def run_sla_verification():
    """运行 SLA 验证

    流程：
    1. 重置 SLA 监控
    2. 发送正常流量（health, lines, cases, dashboard, sla/status）
    3. 注入 SQLite 故障，发送降级流量（cases, dashboard, feedback）
    4. 恢复 SQLite，发送正常流量
    5. 查询 SLA 状态
    """
    client = TestClient(api.routes.app)
    memory = api.routes.memory

    # SQLite 多线程安全
    try:
        memory.db.close()
        memory.db = sqlite3.connect(str(memory.db_path), check_same_thread=False)
    except Exception:
        pass

    # 重置 SLA 监控（清空历史记录）
    sla_monitor.reset()

    results = {
        "normal_requests": 0,
        "degraded_requests": 0,
        "error_requests": 0,
        "details": [],
    }

    # ===== 阶段 1: 正常流量（80 次）=====
    normal_endpoints = [
        ("GET", "/health", None),
        ("GET", "/api/v1/cases", {"user_id": "admin"}),
        ("GET", "/api/v1/dashboard/overview", {"user_id": "admin", "days": 30}),
        ("GET", "/api/v1/sla/status", {"window_minutes": 60}),
    ]
    for i in range(20):  # 4 端点 x 20 次 = 80 次
        for method, path, params in normal_endpoints:
            resp = client.get(path, params=params or {})
            results["normal_requests"] += 1
            results["details"].append({
                "phase": "normal",
                "method": method,
                "path": path,
                "status": resp.status_code,
                "degraded": resp.headers.get("X-Degraded", "false") == "true",
            })

    # ===== 阶段 2: 注入 SQLite 故障，降级流量（30 次）=====
    original_db = memory.db
    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked")
    mock_db.commit.side_effect = sqlite3.OperationalError("database is locked")
    memory.db = mock_db
    if hasattr(memory, "_stats_cache"):
        memory._stats_cache.clear()

    degraded_endpoints = [
        ("GET", "/api/v1/cases", {"user_id": "admin"}),
        ("GET", "/api/v1/dashboard/overview", {"user_id": "admin", "days": 30}),
        ("POST", "/api/v1/feedback", {
            "proposal_id": "sla_test",
            "user_id": "admin",
            "action": "adopted",
            "score": 0.8,
            "comment": "sla验证",
            "line_id": "heat_treatment",
        }),
    ]
    for i in range(10):  # 3 端点 x 10 次 = 30 次
        for method, path, payload in degraded_endpoints:
            if method == "GET":
                resp = client.get(path, params=payload)
            else:
                resp = client.post(path, json=payload)
            degraded = resp.headers.get("X-Degraded", "false") == "true"
            results["degraded_requests"] += 1
            results["details"].append({
                "phase": "degraded",
                "method": method,
                "path": path,
                "status": resp.status_code,
                "degraded": degraded,
            })

    # 恢复 SQLite
    memory.db = original_db
    # 清理 feedback 队列中的测试数据
    api.routes._feedback_queue[:] = [
        x for x in api.routes._feedback_queue if x.get("comment") != "sla验证"
    ]

    # ===== 阶段 3: 恢复后正常流量（20 次）=====
    for i in range(20):
        resp = client.get("/health")
        results["normal_requests"] += 1
        results["details"].append({
            "phase": "recovered",
            "method": "GET",
            "path": "/health",
            "status": resp.status_code,
            "degraded": False,
        })

    # ===== 查询 SLA 状态 =====
    sla_status = client.get("/api/v1/sla/status", params={"window_minutes": 60}).json()
    sla_report = client.get("/api/v1/sla/report", params={
        "window_minutes": 60, "user_id": "admin"
    }).json()

    return results, sla_status, sla_report


def main():
    results, sla_status, sla_report = run_sla_verification()

    # 构建报告
    report = {
        "test_time": datetime.now().isoformat(),
        "sla_target": sla_monitor.SLA_TARGET,
        "summary": {
            "normal_requests": results["normal_requests"],
            "degraded_requests": results["degraded_requests"],
            "error_requests": results["error_requests"],
            "total_requests": sla_status["total_requests"],
        },
        "sla_status": sla_status,
        "sla_by_endpoint": sla_report["by_endpoint"],
        "verification": {
            "availability": sla_status["availability"],
            "availability_pct": f"{sla_status['availability']*100:.2f}%",
            "sla_target_pct": f"{sla_monitor.SLA_TARGET*100:.1f}%",
            "sla_met": sla_status["sla_met"],
            "degraded_count": sla_status["degraded_count"],
            "p95_latency_ms": sla_status["p95_latency_ms"],
            "p99_latency_ms": sla_status["p99_latency_ms"],
        },
        "acceptance": "✅ 达标" if sla_status["sla_met"] else "❌ 未达标",
    }

    # 写 JSON 报告
    json_path = PROJECT_ROOT / "data" / "sla_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 写 Markdown 报告
    md_path = PROJECT_ROOT / "data" / "sla_report.md"
    md = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    # 简短输出
    print(f"Done: availability={sla_status['availability']*100:.2f}%, "
          f"sla_met={sla_status['sla_met']}, "
          f"total={sla_status['total_requests']}, "
          f"degraded={sla_status['degraded_count']}")
    return report


def generate_markdown_report(report: dict) -> str:
    """生成 Markdown 报告"""
    lines = []
    lines.append("# M4-15 SLA 保障验证报告\n\n")
    lines.append(f"**测试时间**：{report['test_time']}\n")
    lines.append(f"**SLA 目标**：可用性 ≥ {report['sla_target']*100:.1f}%\n\n")

    lines.append("## 1. 流量摘要\n\n")
    s = report["summary"]
    lines.append(f"- 正常请求：{s['normal_requests']}\n")
    lines.append(f"- 降级请求：{s['degraded_requests']}\n")
    lines.append(f"- 错误请求：{s['error_requests']}\n")
    lines.append(f"- 总请求数：{s['total_requests']}\n\n")

    lines.append("## 2. SLA 指标\n\n")
    v = report["verification"]
    lines.append("| 指标 | 值 | 说明 |\n")
    lines.append("|------|----|------|\n")
    lines.append(f"| 可用性 | {v['availability_pct']} | 目标 {v['sla_target_pct']} |\n")
    lines.append(f"| P95 延迟 | {v['p95_latency_ms']:.2f} ms | |\n")
    lines.append(f"| P99 延迟 | {v['p99_latency_ms']:.2f} ms | |\n")
    lines.append(f"| 降级次数 | {v['degraded_count']} | 软降级（200+degraded）|\n")
    lines.append(f"| SLA 达标 | {report['acceptance']} | |\n\n")

    lines.append("## 3. 按端点细分\n\n")
    lines.append("| 端点 | 总请求 | 可用性 | P95(ms) | 降级次数 |\n")
    lines.append("|------|--------|--------|---------|----------|\n")
    for path, stats in report["sla_by_endpoint"].items():
        lines.append(
            f"| {path} | {stats['total_requests']} | "
            f"{stats['availability']*100:.2f}% | "
            f"{stats['p95_latency_ms']:.2f} | "
            f"{stats['degraded_count']} |\n"
        )

    lines.append("\n## 4. 验收标准对照\n\n")
    lines.append("| 验收项 | 标准 | 结果 |\n")
    lines.append("|--------|------|------|\n")
    lines.append(
        f"| M4-15 可用性 ≥ 99.5% | {report['sla_target']*100:.1f}% | "
        f"{report['acceptance']}（{v['availability_pct']}）|\n"
    )

    lines.append("\n## 5. 降级机制说明\n\n")
    lines.append("SQLite 故障时，cases/dashboard/feedback 端点通过 try/except 捕获异常，")
    lines.append("返回 200+degraded（软降级），不计入 5xx 错误，因此 availability 不受影响。\n")
    lines.append("全局异常处理器 + HTTP middleware 兜底未处理异常，返回 503（硬降级），")
    lines.append("会计入 5xx，影响 availability。\n\n")

    lines.append("## 6. 结论\n\n")
    if report["verification"]["sla_met"]:
        lines.append("✅ **SLA 达标**：可用性满足 99.5% 目标，降级机制有效保障服务可用性。\n")
    else:
        lines.append("❌ **SLA 未达标**：可用性低于 99.5%，需排查 5xx 错误来源。\n")

    return "".join(lines)


if __name__ == "__main__":
    main()
