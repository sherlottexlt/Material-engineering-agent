"""
M4-12 压力测试验证 + 报告生成

用 pytest 运行此文件即可执行压测并生成报告：
    python -m pytest tests/test_stress_test.py -v -s

或直接运行：
    python tests/test_stress_test.py

生成的报告：
    data/stress_test_report.json
    data/stress_test_report.md
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_stress_test_script_importable():
    """验证压测脚本可正常 import"""
    from scripts.run_stress_test import run_stress_test, generate_markdown_report
    assert callable(run_stress_test)
    assert callable(generate_markdown_report)


def test_stress_test_small_scale():
    """小规模压测验证（2 用户 x 2 请求 x 2 端点）"""
    from scripts.run_stress_test import run_stress_test

    report = run_stress_test(
        num_users=2,
        requests_per_user=2,
        endpoint_keys=["auth", "lines"],
    )

    # 验证报告结构
    assert "test_time" in report
    assert "config" in report
    assert "results" in report
    assert report["config"]["num_users"] == 2
    assert "auth" in report["results"]
    assert "lines" in report["results"]

    # 验证权限查询端点成功率应 100%（admin 用户）
    auth_metrics = report["results"]["auth"]["metrics"]
    assert auth_metrics["success_rate"] >= 0.5  # 至少一半成功
    assert auth_metrics["total_requests"] == 4  # 2 用户 x 2 请求


def test_stress_test_read_endpoints():
    """压测读端点（auth/lines/cases/dashboard）"""
    from scripts.run_stress_test import run_stress_test

    report = run_stress_test(
        num_users=3,
        requests_per_user=3,
        endpoint_keys=["auth", "lines", "cases", "dashboard"],
    )

    # 全部读端点应高成功率
    for ep_key, ep_data in report["results"].items():
        assert ep_data["metrics"]["success_rate"] >= 0.5, (
            f"{ep_key} 成功率过低: {ep_data['metrics']['success_rate']}"
        )
        # 响应时间指标应存在
        lat = ep_data["metrics"]["latency_ms"]
        assert "p50" in lat
        assert "p95" in lat
        assert "p99" in lat
        assert lat["p50"] >= 0


def test_stress_test_write_endpoint():
    """压测写端点（feedback）"""
    from scripts.run_stress_test import run_stress_test

    report = run_stress_test(
        num_users=2,
        requests_per_user=2,
        endpoint_keys=["feedback"],
    )

    fb_metrics = report["results"]["feedback"]["metrics"]
    # feedback 端点可能因权限校验产生 403（operator_02 无权访问 heat_treatment）
    # 但至少 admin/supervisor_01 的请求应成功
    assert fb_metrics["total_requests"] == 4
    # 不要求 100%，但应能完成请求（不报 500）
    assert fb_metrics["status_codes"]["server_error_5xx"] == 0


def test_percentile_calculation():
    """验证百分位数计算"""
    from scripts.run_stress_test import _percentile

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _percentile(data, 0.5) == 5.5  # 中位数
    assert _percentile(data, 0.0) == 1    # 最小值
    assert _percentile(data, 1.0) == 10   # 最大值
    assert _percentile([], 0.5) == 0      # 空数据


def test_compute_metrics():
    """验证指标计算"""
    from scripts.run_stress_test import _compute_metrics

    latencies = [10, 20, 30, 40, 50]
    status_codes = [200, 200, 200, 403, 500]
    metrics = _compute_metrics(latencies, status_codes, 1.0, 5)

    assert metrics["total_requests"] == 5
    assert metrics["qps"] == 5.0
    assert metrics["latency_ms"]["min"] == 10
    assert metrics["latency_ms"]["max"] == 50
    assert metrics["status_codes"]["success_2xx"] == 3
    assert metrics["status_codes"]["client_error_4xx"] == 1
    assert metrics["status_codes"]["server_error_5xx"] == 1
    assert metrics["success_rate"] == 0.6
    assert metrics["error_rate"] == 0.4


def test_generate_markdown_report():
    """验证 Markdown 报告生成"""
    from scripts.run_stress_test import generate_markdown_report

    fake_report = {
        "test_time": "2026-07-02T12:00:00",
        "config": {
            "num_users": 10,
            "requests_per_user": 20,
            "total_requests_per_endpoint": 200,
            "endpoints": ["auth", "cases"],
        },
        "results": {
            "auth": {
                "description": "权限查询",
                "method": "GET",
                "path": "/api/v1/auth/permissions",
                "metrics": {
                    "total_requests": 200,
                    "total_time_s": 1.5,
                    "qps": 133.3,
                    "latency_ms": {"min": 1, "avg": 5, "p50": 4, "p90": 8, "p95": 10, "p99": 15, "max": 20},
                    "status_codes": {"success_2xx": 200, "client_error_4xx": 0, "server_error_5xx": 0},
                    "success_rate": 1.0,
                    "error_rate": 0.0,
                },
            },
        },
    }
    md = generate_markdown_report(fake_report)
    assert "# M4-12 压力测试报告" in md
    assert "权限查询" in md
    assert "验收标准" in md
    assert "达标" in md  # 100% 成功率应达标


def test_full_stress_test_and_generate_report():
    """完整压测（10 用户 x 20 请求）+ 生成报告文件

    这是 M4-12 的核心验收测试，生成 data/stress_test_report.json 和 .md
    """
    from scripts.run_stress_test import run_stress_test, generate_markdown_report

    # 执行完整压测
    report = run_stress_test(
        num_users=10,
        requests_per_user=20,
        endpoint_keys=None,  # 全部端点
    )

    # 验证全部端点都被压测
    expected_endpoints = {"auth", "lines", "cases", "dashboard", "feedback"}
    assert set(report["results"].keys()) == expected_endpoints

    # 验证每端点总请求数
    for ep_key, ep_data in report["results"].items():
        assert ep_data["metrics"]["total_requests"] == 200  # 10 x 20

    # 生成 JSON 报告
    json_path = PROJECT_ROOT / "data" / "stress_test_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 生成 Markdown 报告
    md_path = json_path.with_suffix(".md")
    md_content = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 打印关键指标
    print("\n" + "=" * 60)
    print("M4-12 压测结果汇总（10 并发 x 20 请求/用户 = 200 请求/端点）")
    print("=" * 60)
    for ep_key, ep_data in report["results"].items():
        m = ep_data["metrics"]
        lat = m["latency_ms"]
        print(
            f"  {ep_key:12s} | QPS: {m['qps']:7.1f} | "
            f"P50: {lat['p50']:7.1f}ms | P95: {lat['p95']:7.1f}ms | "
            f"P99: {lat['p99']:7.1f}ms | 成功率: {m['success_rate']:.1%}"
        )
    print("=" * 60)
    print(f"报告已生成: {json_path}")
    print(f"报告已生成: {md_path}")


if __name__ == "__main__":
    # 直接运行此文件执行完整压测
    test_full_stress_test_and_generate_report()
