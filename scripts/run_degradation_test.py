"""M4-14 降级策略测试运行器

用 sys.modules mock langchain_openai 绕过 sandbox import 超时问题，
运行 tests/test_degradation.py 的 8 个测试，结果写到 data/degradation_test_report.{json,md}。

用法：
    python scripts/run_degradation_test.py
"""
import logging
import sys
import os
import time
import types
import json
import traceback
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
    """模拟 ChatOpenAI，仅用于测试 import 与配置读取"""
    def __init__(self, **kwargs):
        self.max_retries = kwargs.get("max_retries", 3)
        self.model = kwargs.get("model", "")
        self.temperature = kwargs.get("temperature", 0)
        self.request_timeout = kwargs.get("request_timeout", 120)
        self.openai_api_key = kwargs.get("openai_api_key", "")
        self.openai_api_base = kwargs.get("openai_api_base", "")
    def invoke(self, *a, **kw):
        return types.SimpleNamespace(content="mock")
    async def ainvoke(self, *a, **kw):
        return types.SimpleNamespace(content="mock")

_fake_module = types.ModuleType("langchain_openai")
_fake_module.ChatOpenAI = _FakeChatOpenAI
sys.modules["langchain_openai"] = _fake_module


# ===== 运行测试 =====

def run_all_tests():
    """运行 tests/test_degradation.py 的全部测试"""
    from tests.test_degradation import (
        test_cases_sqlite_degradation,
        test_dashboard_sqlite_degradation,
        test_feedback_sqlite_degradation,
        test_global_exception_handler,
        test_sqlite_init_fallback,
        test_wal_mode_enabled,
        test_get_line_stats_cache,
        test_llm_max_retries,
    )

    tests = [
        ("test_cases_sqlite_degradation", test_cases_sqlite_degradation,
         "SQLite 故障时 cases 端点返回 200+空结果+degraded"),
        ("test_dashboard_sqlite_degradation", test_dashboard_sqlite_degradation,
         "SQLite 故障时 dashboard 端点返回 200+零值统计+degraded"),
        ("test_feedback_sqlite_degradation", test_feedback_sqlite_degradation,
         "SQLite 故障时 feedback 端点返回 200+degraded+暂存队列"),
        ("test_global_exception_handler", test_global_exception_handler,
         "未处理异常返回 503+degraded"),
        ("test_sqlite_init_fallback", test_sqlite_init_fallback,
         "SQLite 初始化失败降级为内存数据库"),
        ("test_wal_mode_enabled", test_wal_mode_enabled,
         "SQLite 启用 WAL 模式"),
        ("test_get_line_stats_cache", test_get_line_stats_cache,
         "get_line_stats 60 秒内返回缓存"),
        ("test_llm_max_retries", test_llm_max_retries,
         "get_llm 返回的客户端 max_retries=3"),
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
    """生成 Markdown 报告"""
    lines = []
    lines.append("# M4-14 降级策略测试报告\n")
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
    lines.append(f"| M4-14 降级测试通过率 | 100% | {overall}（{report['passed']}/{report['total']}）|\n")

    return "".join(lines)


def main():
    report = run_all_tests()

    # 写 JSON 报告
    json_path = PROJECT_ROOT / "data" / "degradation_test_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 写 Markdown 报告
    md_path = PROJECT_ROOT / "data" / "degradation_test_report.md"
    md = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    # 简短输出（避免 sandbox 缓冲区溢出）
    print(f"Done: {report['passed']}/{report['total']} passed")
    return report


if __name__ == "__main__":
    main()
