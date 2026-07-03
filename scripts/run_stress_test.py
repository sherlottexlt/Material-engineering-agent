"""
M4-12 压力测试：10 并发用户压测

模拟 10 个并发用户同时访问 API，收集响应时间、吞吐量、成功率等指标。

压测端点（不压测 /api/v1/analyze，避免调用 LLM 消耗配额）：
  - GET  /api/v1/auth/permissions   权限查询（轻量）
  - GET  /api/v1/lines               产线列表（轻量）
  - GET  /api/v1/cases               案例查询（中等，读 SQLite）
  - GET  /api/v1/dashboard/overview  跨产线看板（中等，聚合统计）
  - POST /api/v1/feedback            反馈写入（中等，写 SQLite）

指标：
  - 响应时间：P50 / P90 / P95 / P99 / Max / Avg（ms）
  - 吞吐量：QPS（requests/sec）
  - 成功率：2xx 占比
  - 错误分类：4xx / 5xx 数量

用法：
    python scripts/run_stress_test.py
    python scripts/run_stress_test.py --users 20 --requests 50
    python scripts/run_stress_test.py --endpoints cases,feedback
    python scripts/run_stress_test.py --output data/stress_test_report.json

输出：
    data/stress_test_report.json  结构化压测数据
    data/stress_test_report.md    人类可读报告
"""
import argparse
import json
import logging
import sqlite3
import statistics
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# M4-16: 抑制日志输出，避免 sandbox 缓冲区溢出
logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# M4-16: mock langchain_openai 避免 sandbox import 超时（压测不调用 LLM 端点）
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


# ===== 压测场景定义 =====

# 10 个模拟用户（覆盖不同角色，验证权限隔离下的并发）
SIMULATED_USERS = [
    {"user_id": "admin", "role": "admin"},
    {"user_id": "supervisor_01", "role": "supervisor"},
    {"user_id": "operator_01", "role": "operator"},
    {"user_id": "operator_02", "role": "operator"},
    {"user_id": "admin", "role": "admin"},          # 重复用户模拟并发
    {"user_id": "supervisor_01", "role": "supervisor"},
    {"user_id": "operator_01", "role": "operator"},
    {"user_id": "operator_02", "role": "operator"},
    {"user_id": "admin", "role": "admin"},
    {"user_id": "supervisor_01", "role": "supervisor"},
]

# 压测端点配置
ENDPOINTS = {
    "auth": {
        "method": "GET",
        "path": "/api/v1/auth/permissions",
        "params": lambda uid: {"user_id": uid},
        "json": None,
        "description": "权限查询（轻量）",
    },
    "lines": {
        "method": "GET",
        "path": "/api/v1/lines",
        "params": lambda uid: {"user_id": uid},
        "json": None,
        "description": "产线列表（轻量）",
    },
    "cases": {
        "method": "GET",
        "path": "/api/v1/cases",
        "params": lambda uid: {"user_id": uid, "limit": 20},
        "json": None,
        "description": "案例查询（中等，读 SQLite）",
    },
    "dashboard": {
        "method": "GET",
        "path": "/api/v1/dashboard/overview",
        "params": lambda uid: {"user_id": uid, "days": 30},
        "json": None,
        "description": "跨产线看板（中等，聚合统计）",
    },
    "feedback": {
        "method": "POST",
        "path": "/api/v1/feedback",
        "params": None,
        "json": lambda uid, idx: {
            "proposal_id": f"P_STRESS_{uid}_{idx}",
            "user_id": uid,
            "action": "adopted",
            "score": 0.85,
            "comment": "压测反馈",
            "line_id": "heat_treatment" if uid != "operator_02" else "welding",
        },
        "description": "反馈写入（中等，写 SQLite）",
    },
}


def _percentile(sorted_data: list[float], p: float) -> float:
    """计算百分位数（线性插值法）"""
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _compute_metrics(latencies: list[float], status_codes: list[int],
                     total_time: float, total_requests: int) -> dict:
    """计算压测指标"""
    sorted_lat = sorted(latencies)
    success_count = sum(1 for sc in status_codes if 200 <= sc < 300)
    client_errors = sum(1 for sc in status_codes if 400 <= sc < 500)
    server_errors = sum(1 for sc in status_codes if sc >= 500)

    return {
        "total_requests": total_requests,
        "total_time_s": round(total_time, 3),
        "qps": round(total_requests / total_time, 2) if total_time > 0 else 0,
        "latency_ms": {
            "min": round(sorted_lat[0], 2) if sorted_lat else 0,
            "avg": round(statistics.mean(sorted_lat), 2) if sorted_lat else 0,
            "p50": round(_percentile(sorted_lat, 0.50), 2),
            "p90": round(_percentile(sorted_lat, 0.90), 2),
            "p95": round(_percentile(sorted_lat, 0.95), 2),
            "p99": round(_percentile(sorted_lat, 0.99), 2),
            "max": round(sorted_lat[-1], 2) if sorted_lat else 0,
        },
        "status_codes": {
            "success_2xx": success_count,
            "client_error_4xx": client_errors,
            "server_error_5xx": server_errors,
        },
        "success_rate": round(success_count / total_requests, 4) if total_requests > 0 else 0,
        "error_rate": round((client_errors + server_errors) / total_requests, 4) if total_requests > 0 else 0,
    }


def _send_request(client, endpoint_cfg: dict, user_id: str, idx: int) -> tuple[float, int]:
    """发送单个请求，返回 (latency_ms, status_code)"""
    method = endpoint_cfg["method"]
    path = endpoint_cfg["path"]

    start = time.perf_counter()
    try:
        if method == "GET":
            params = endpoint_cfg["params"](user_id) if endpoint_cfg["params"] else None
            resp = client.get(path, params=params)
        else:  # POST
            json_body = endpoint_cfg["json"](user_id, idx) if endpoint_cfg["json"] else None
            resp = client.post(path, json=json_body)
        status = resp.status_code
    except Exception as e:
        # 连接异常等，视为 500
        status = 500
    elapsed = (time.perf_counter() - start) * 1000  # ms
    return elapsed, status


def run_stress_test(
    num_users: int = 10,
    requests_per_user: int = 20,
    endpoint_keys: list[str] | None = None,
) -> dict:
    """执行压测

    Args:
        num_users: 并发用户数
        requests_per_user: 每用户请求数
        endpoint_keys: 压测的端点列表，None 表示全部

    Returns:
        压测结果 dict
    """
    # 延迟导入，确保替换 SQLite 连接在 import 之后
    from fastapi.testclient import TestClient
    import api.routes

    # M4-12/M4-16: 替换 SQLite 连接为多线程安全 + WAL + busy_timeout 优化版本
    # （MemoryService 默认已含这些优化，此处替换确保压测环境一致性）
    api.routes.memory.db.close()
    api.routes.memory.db = sqlite3.connect(
        str(api.routes.memory.db_path), timeout=5.0, check_same_thread=False
    )
    api.routes.memory.db.execute("PRAGMA journal_mode=WAL")
    api.routes.memory.db.execute("PRAGMA busy_timeout=5000")

    app = api.routes.app
    target_endpoints = endpoint_keys or list(ENDPOINTS.keys())

    print(f"\n{'='*60}")
    print(f"M4-12 压力测试")
    print(f"  并发用户数: {num_users}")
    print(f"  每用户请求数: {requests_per_user}")
    print(f"  总请求数: {num_users * requests_per_user * len(target_endpoints)}")
    print(f"  压测端点: {', '.join(target_endpoints)}")
    print(f"{'='*60}\n")

    all_results: dict[str, dict] = {}

    for ep_key in target_endpoints:
        ep_cfg = ENDPOINTS[ep_key]

        latencies: list[float] = []
        status_codes: list[int] = []
        total_requests = num_users * requests_per_user

        # 用 ThreadPoolExecutor 模拟并发用户
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            start_time = time.perf_counter()

            for i in range(num_users):
                user = SIMULATED_USERS[i % len(SIMULATED_USERS)]
                uid = user["user_id"]
                # 每个线程创建独立 TestClient（共享 app + memory）
                client = TestClient(app)
                for j in range(requests_per_user):
                    futures.append(
                        executor.submit(_send_request, client, ep_cfg, uid, j)
                    )

            for future in as_completed(futures):
                elapsed, status = future.result()
                latencies.append(elapsed)
                status_codes.append(status)

            total_time = time.perf_counter() - start_time

        metrics = _compute_metrics(latencies, status_codes, total_time, total_requests)
        all_results[ep_key] = {
            "description": ep_cfg["description"],
            "method": ep_cfg["method"],
            "path": ep_cfg["path"],
            "metrics": metrics,
        }

        # 减少 stdout 输出，避免 sandbox 缓冲区溢出
        # print(f"  QPS: {metrics['qps']}, P50: {metrics['latency_ms']['p50']}ms, ...")

    # 清理压测写入的反馈数据
    try:
        api.routes.memory.db.execute(
            "DELETE FROM feedback WHERE feedback_id LIKE 'fb_%' AND comment = '压测反馈'"
        )
        api.routes.memory.db.commit()
    except Exception:
        pass

    return {
        "test_time": datetime.now().isoformat(),
        "config": {
            "num_users": num_users,
            "requests_per_user": requests_per_user,
            "total_requests_per_endpoint": num_users * requests_per_user,
            "endpoints": target_endpoints,
        },
        "results": all_results,
    }


def generate_markdown_report(report_data: dict) -> str:
    """生成 Markdown 压测报告"""
    cfg = report_data["config"]
    lines = [
        "# M4-12 压力测试报告",
        "",
        f"**测试时间**：{report_data['test_time']}",
        f"**并发用户数**：{cfg['num_users']}",
        f"**每用户请求数**：{cfg['requests_per_user']}",
        f"**每端点总请求数**：{cfg['total_requests_per_endpoint']}",
        f"**压测端点**：{', '.join(cfg['endpoints'])}",
        "",
        "## 1. 汇总指标",
        "",
        "| 端点 | 描述 | 方法 | QPS | P50(ms) | P90(ms) | P95(ms) | P99(ms) | Max(ms) | 成功率 | 错误数 |",
        "|------|------|------|-----|---------|---------|---------|---------|---------|--------|--------|",
    ]

    for ep_key, ep_data in report_data["results"].items():
        m = ep_data["metrics"]
        lat = m["latency_ms"]
        sc = m["status_codes"]
        error_count = sc["client_error_4xx"] + sc["server_error_5xx"]
        lines.append(
            f"| {ep_key} | {ep_data['description']} | {ep_data['method']} | "
            f"{m['qps']} | {lat['p50']} | {lat['p90']} | {lat['p95']} | {lat['p99']} | "
            f"{lat['max']} | {m['success_rate']:.1%} | {error_count} |"
        )

    lines.extend([
        "",
        "## 2. 响应时间分布",
        "",
        "| 端点 | Min(ms) | Avg(ms) | P50(ms) | P90(ms) | P95(ms) | P99(ms) | Max(ms) |",
        "|------|---------|---------|---------|---------|---------|---------|---------|",
    ])

    for ep_key, ep_data in report_data["results"].items():
        lat = ep_data["metrics"]["latency_ms"]
        lines.append(
            f"| {ep_key} | {lat['min']} | {lat['avg']} | {lat['p50']} | "
            f"{lat['p90']} | {lat['p95']} | {lat['p99']} | {lat['max']} |"
        )

    lines.extend([
        "",
        "## 3. 状态码分布",
        "",
        "| 端点 | 2xx 成功 | 4xx 客户端错误 | 5xx 服务端错误 | 成功率 |",
        "|------|----------|----------------|----------------|--------|",
    ])

    for ep_key, ep_data in report_data["results"].items():
        sc = ep_data["metrics"]["status_codes"]
        lines.append(
            f"| {ep_key} | {sc['success_2xx']} | {sc['client_error_4xx']} | "
            f"{sc['server_error_5xx']} | {ep_data['metrics']['success_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## 4. 验收标准对照",
        "",
        "| 验收项 | 标准 | 结果 |",
        "|--------|------|------|",
        f"| 单产线 10 并发可用 | 10 并发下成功率 ≥ 95% | {_check_acceptance(report_data)} |",
        "",
        "## 5. 结论",
        "",
        _generate_conclusion(report_data),
        "",
    ])

    return "\n".join(lines)


def _check_acceptance(report_data: dict) -> str:
    """检查验收标准：10 并发下成功率 ≥ 95%"""
    for ep_data in report_data["results"].values():
        if ep_data["metrics"]["success_rate"] < 0.95:
            return "❌ 未达标（部分端点成功率 < 95%）"
    return "✅ 达标（全部端点成功率 ≥ 95%）"


def _generate_conclusion(report_data: dict) -> str:
    """生成结论文字"""
    all_p95 = [
        ep_data["metrics"]["latency_ms"]["p95"]
        for ep_data in report_data["results"].values()
    ]
    all_success = [
        ep_data["metrics"]["success_rate"]
        for ep_data in report_data["results"].values()
    ]
    avg_p95 = sum(all_p95) / len(all_p95) if all_p95 else 0
    min_success = min(all_success) if all_success else 0

    conclusion = f"在 {report_data['config']['num_users']} 并发用户压测下：\n"
    conclusion += f"- 平均 P95 响应时间：{avg_p95:.1f}ms\n"
    conclusion += f"- 最低成功率：{min_success:.1%}\n"
    if min_success >= 0.95 and avg_p95 < 1000:
        conclusion += "- 系统在 10 并发下表现稳定，满足验收标准。\n"
    elif min_success >= 0.95:
        conclusion += "- 成功率达标，但响应时间较高，建议优化慢查询。\n"
    else:
        conclusion += "- 部分端点成功率不达标，需排查并发问题。\n"
    return conclusion


def main():
    parser = argparse.ArgumentParser(description="M4-12 压力测试")
    parser.add_argument("--users", type=int, default=10, help="并发用户数（默认 10）")
    parser.add_argument("--requests", type=int, default=20, help="每用户请求数（默认 20）")
    parser.add_argument(
        "--endpoints", type=str, default=None,
        help="压测端点（逗号分隔，如 cases,feedback）；默认全部"
    )
    parser.add_argument(
        "--output", type=str, default="data/stress_test_report.json",
        help="JSON 报告输出路径"
    )
    args = parser.parse_args()

    endpoint_keys = args.endpoints.split(",") if args.endpoints else None

    # 执行压测
    report_data = run_stress_test(
        num_users=args.users,
        requests_per_user=args.requests,
        endpoint_keys=endpoint_keys,
    )

    # 写 JSON 报告
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n[报告] JSON 报告已生成: {output_path}")

    # 写 Markdown 报告
    md_path = output_path.with_suffix(".md")
    md_content = generate_markdown_report(report_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[报告] Markdown 报告已生成: {md_path}")

    # 打印结论
    print(f"\n{'='*60}")
    print("压测结论：")
    print(_generate_conclusion(report_data))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
