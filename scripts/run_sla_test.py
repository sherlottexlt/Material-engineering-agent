"""M4-15 SLA 保障测试运行器

用 sys.modules mock langchain_openai 绕过 sandbox import 超时问题，
运行 tests/test_sla.py 的全部测试，结果写到 data/sla_test_report.{json,md}。

用法：
    python scripts/run_sla_test.py
"""
import json
import logging
import sys
import time
import traceback
import types
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

_fake_module = types.ModuleType("langchain_openai")
_fake_module.ChatOpenAI = _FakeChatOpenAI
sys.modules["langchain_openai"] = _fake_module


def run_all_tests():
    """运行 tests/test_sla.py 的全部测试"""
    from tests.test_sla import (
        test_sla_monitor_record_and_stats,
        test_sla_availability_calc,
        test_sla_p95_p99,
        test_sla_degraded_count,
        test_sla_empty_stats,
        test_sla_window_filter,
        test_sla_stats_by_endpoint,
        test_sla_reset,
        test_sla_status_endpoint,
        test_sla_report_endpoint_admin,
        test_sla_report_permission_denied,
        test_sla_middleware_records_requests,
        test_sla_middleware_records_degraded,
        test_sla_middleware_records_503,
        test_sla_99_5_target_met,
    )

    tests = [
        ("test_sla_monitor_record_and_stats", test_sla_monitor_record_and_stats,
         "SLAMonitor 记录请求并查询统计"),
        ("test_sla_availability_calc", test_sla_availability_calc,
         "可用性计算：5xx 算不可用"),
        ("test_sla_p95_p99", test_sla_p95_p99,
         "P95/P99 百分位计算"),
        ("test_sla_degraded_count", test_sla_degraded_count,
         "降级计数"),
        ("test_sla_empty_stats", test_sla_empty_stats,
         "空记录返回默认值"),
        ("test_sla_window_filter", test_sla_window_filter,
         "时间窗口过滤"),
        ("test_sla_stats_by_endpoint", test_sla_stats_by_endpoint,
         "按端点细分统计"),
        ("test_sla_reset", test_sla_reset,
         "reset 清空记录"),
        ("test_sla_status_endpoint", test_sla_status_endpoint,
         "/api/v1/sla/status 端点"),
        ("test_sla_report_endpoint_admin", test_sla_report_endpoint_admin,
         "admin 可访问 /api/v1/sla/report"),
        ("test_sla_report_permission_denied", test_sla_report_permission_denied,
         "非 admin 无权访问 /api/v1/sla/report"),
        ("test_sla_middleware_records_requests", test_sla_middleware_records_requests,
         "SLA 中间件自动记录请求"),
        ("test_sla_middleware_records_degraded", test_sla_middleware_records_degraded,
         "中间件记录降级请求（X-Degraded header）"),
        ("test_sla_middleware_records_503", test_sla_middleware_records_503,
         "中间件记录 503 硬降级"),
        ("test_sla_99_5_target_met", test_sla_99_5_target_met,
         "99.5% 可用性达标验证"),
    ]

    results = []
    for name, func, desc in tests:
        t0 = time.time()
        try:
            func()
            dt = time.time() - t0
            results.append({
                "name": name,
                "description": desc,
                "status": "PASS",
                "time_ms": round(dt * 1000, 1),
                "error": "",
            })
        except Exception as e:
            dt = time.time() - t0
            results.append({
                "name": name,
                "description": desc,
                "status": "FAIL",
                "time_ms": round(dt * 1000, 1),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })

    passed = sum(1 for r in results if r["status"] == "PASS")
    report = {
        "test_time": datetime.now().isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4),
        "results": results,
    }
    return report


def generate_markdown_report(report: dict) -> str:
    lines = []
    lines.append("# M4-15 SLA 保障测试报告\n\n")
    lines.append(f"**测试时间**：{report['test_time']}\n")
    lines.append(f"**测试总数**：{report['total']}\n")
    lines.append(f"**通过**：{report['passed']}\n")
    lines.append(f"**失败**：{report['failed']}\n")
    lines.append(f"**通过率**：{report['pass_rate']*100:.1f}%\n\n")

    lines.append("## 测试明细\n\n")
    lines.append("| 测试 | 描述 | 状态 | 耗时(ms) | 错误 |\n")
    lines.append("|------|------|------|---------|------|\n")
    for r in report["results"]:
        err = r.get("error", "")
        lines.append(
            f"| {r['name']} | {r['description']} | {r['status']} | "
            f"{r['time_ms']} | {err} |\n"
        )

    if report["failed"] > 0:
        lines.append("\n## 失败详情\n\n")
        for r in report["results"]:
            if r["status"] == "FAIL":
                lines.append(f"### {r['name']}\n\n")
                lines.append(f"**错误**：{r.get('error', '')}\n\n")
                if "traceback" in r:
                    lines.append("```\n" + r["traceback"] + "\n```\n\n")

    lines.append("\n## 验收标准对照\n\n")
    lines.append("| 验收项 | 标准 | 结果 |\n")
    lines.append("|--------|------|------|\n")
    overall = "✅ 达标" if report["failed"] == 0 else "❌ 未达标"
    lines.append(f"| M4-15 SLA 测试通过率 | 100% | {overall}（{report['passed']}/{report['total']}）|\n")

    return "".join(lines)


def main():
    try:
        report = run_all_tests()
    except Exception as e:
        # 捕获 import 或运行时错误，写到日志
        err_log = PROJECT_ROOT / "data" / "_sla_test_error.log"
        with open(err_log, "w", encoding="utf-8") as f:
            f.write(f"run_all_tests FAILED: {e}\n")
            f.write(traceback.format_exc())
        print(f"FAILED: {e}")
        return None

    json_path = PROJECT_ROOT / "data" / "sla_test_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = PROJECT_ROOT / "data" / "sla_test_report.md"
    md = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Done: {report['passed']}/{report['total']} passed")
    return report


if __name__ == "__main__":
    main()
