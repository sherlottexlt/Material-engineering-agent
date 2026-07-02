"""
M4-15 SLA 保障测试

验证：
1. SLAMonitor 记录与统计（可用性、P95/P99、降级计数、时间窗口）
2. SLA 端点（/api/v1/sla/status、/api/v1/sla/report）
3. SLA 监控中间件自动记录请求
4. 降级响应 X-Degraded header 传播
5. 99.5% 可用性达标验证
"""
import sys
import time
import types
import unittest.mock
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===== 1. SLAMonitor 单元测试 =====

def test_sla_monitor_record_and_stats():
    """记录请求并查询统计"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    monitor.record("GET", "/health", 200, 10.0)
    monitor.record("GET", "/health", 200, 20.0)
    monitor.record("GET", "/health", 200, 30.0)

    stats = monitor.get_stats(window_minutes=60)
    assert stats["total_requests"] == 3
    assert stats["availability"] == 1.0  # 全部 2xx
    assert stats["error_rate"] == 0.0
    assert stats["degraded_count"] == 0
    assert stats["sla_met"] is True
    assert stats["sla_target"] == 0.995


def test_sla_availability_calc():
    """可用性计算：2xx+3xx+4xx 算可用，5xx 算不可用"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    # 99 个 200 + 1 个 500 = 99% 可用性（未达标）
    for _ in range(99):
        monitor.record("GET", "/ok", 200, 10.0)
    monitor.record("GET", "/fail", 500, 50.0)

    stats = monitor.get_stats(window_minutes=60)
    assert stats["total_requests"] == 100
    assert stats["availability"] == 0.99
    assert stats["error_rate"] == 0.01
    assert stats["sla_met"] is False  # 99% < 99.5%


def test_sla_p95_p99():
    """P95/P99 百分位计算"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    # 100 个请求，延迟 1-100ms
    for i in range(1, 101):
        monitor.record("GET", "/test", 200, float(i))

    stats = monitor.get_stats(window_minutes=60)
    # P95 ≈ 95.05，P99 ≈ 99.01（线性插值）
    assert 94 <= stats["p95_latency_ms"] <= 96
    assert 98 <= stats["p99_latency_ms"] <= 100
    assert stats["avg_latency_ms"] == 50.5


def test_sla_degraded_count():
    """降级计数：X-Degraded 标记的请求"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    monitor.record("GET", "/ok", 200, 10.0, degraded=False)
    monitor.record("GET", "/degraded", 200, 15.0, degraded=True)
    monitor.record("GET", "/fail", 503, 20.0, degraded=True)

    stats = monitor.get_stats(window_minutes=60)
    assert stats["degraded_count"] == 2
    # 503 算不可用
    assert stats["availability"] == pytest.approx(2 / 3, abs=0.01)


def test_sla_empty_stats():
    """空记录返回默认值（可用性 1.0，达标）"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    stats = monitor.get_stats(window_minutes=60)
    assert stats["total_requests"] == 0
    assert stats["availability"] == 1.0
    assert stats["sla_met"] is True
    assert stats["p95_latency_ms"] == 0.0


def test_sla_window_filter():
    """时间窗口过滤：旧记录不统计"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    # 手动插入一条 2 小时前的记录
    with monitor._lock:
        monitor._records.append({
            "timestamp": time.time() - 7200,  # 2 小时前
            "method": "GET", "path": "/old",
            "status_code": 200, "duration_ms": 10.0,
            "degraded": False, "user_id": None, "line_id": None,
        })
    monitor.record("GET", "/new", 200, 20.0)

    # 60 分钟窗口：只统计 1 条
    stats = monitor.get_stats(window_minutes=60)
    assert stats["total_requests"] == 1

    # 180 分钟窗口：统计 2 条
    stats = monitor.get_stats(window_minutes=180)
    assert stats["total_requests"] == 2


def test_sla_stats_by_endpoint():
    """按端点细分统计"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    monitor.record("GET", "/health", 200, 10.0)
    monitor.record("GET", "/health", 200, 20.0)
    monitor.record("GET", "/cases", 200, 50.0)
    monitor.record("GET", "/cases", 500, 100.0)

    by_ep = monitor.get_stats_by_endpoint(window_minutes=60)
    assert "/health" in by_ep
    assert "/cases" in by_ep
    assert by_ep["/health"]["total_requests"] == 2
    assert by_ep["/health"]["availability"] == 1.0
    assert by_ep["/cases"]["total_requests"] == 2
    assert by_ep["/cases"]["availability"] == 0.5


def test_sla_reset():
    """reset 清空记录"""
    from agent.sla import SLAMonitor

    monitor = SLAMonitor()
    monitor.record("GET", "/test", 200, 10.0)
    assert monitor.get_stats(window_minutes=60)["total_requests"] == 1
    monitor.reset()
    assert monitor.get_stats(window_minutes=60)["total_requests"] == 0


# ===== 2. SLA 端点测试 =====

def _get_client():
    """获取 TestClient（复用 api.routes 单例）"""
    from fastapi.testclient import TestClient
    import api.routes
    return TestClient(api.routes.app), api.routes


def test_sla_status_endpoint():
    """/api/v1/sla/status 返回 SLA 状态"""
    client, _ = _get_client()
    resp = client.get("/api/v1/sla/status", params={"window_minutes": 60})
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "availability" in data
    assert "p95_latency_ms" in data
    assert "p99_latency_ms" in data
    assert "degraded_count" in data
    assert data["sla_target"] == 0.995
    assert "sla_met" in data


def test_sla_report_endpoint_admin():
    """admin 用户可访问 /api/v1/sla/report"""
    client, _ = _get_client()
    resp = client.get("/api/v1/sla/report", params={
        "window_minutes": 60, "user_id": "admin"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "overall" in data
    assert "by_endpoint" in data
    assert data["sla_target"] == 0.995


def test_sla_report_permission_denied():
    """非 admin 用户无权访问 /api/v1/sla/report"""
    client, _ = _get_client()
    resp = client.get("/api/v1/sla/report", params={
        "window_minutes": 60, "user_id": "operator_01"
    })
    assert resp.status_code == 403


# ===== 3. SLA 监控中间件测试 =====

def test_sla_middleware_records_requests():
    """SLA 中间件自动记录请求"""
    from agent.sla import sla_monitor
    client, _ = _get_client()

    sla_monitor.reset()
    # 发送几个请求
    client.get("/health")
    client.get("/api/v1/sla/status", params={"window_minutes": 60})

    stats = sla_monitor.get_stats(window_minutes=60)
    assert stats["total_requests"] >= 2
    # health 和 sla/status 都是 2xx
    assert stats["availability"] == 1.0


def test_sla_middleware_records_degraded():
    """中间件记录降级请求（X-Degraded header）"""
    import sqlite3
    from agent.sla import sla_monitor
    client, routes = _get_client()
    memory = routes.memory

    sla_monitor.reset()
    original_db = memory.db
    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked")
    mock_db.commit.side_effect = sqlite3.OperationalError("database is locked")
    memory.db = mock_db
    if hasattr(memory, "_stats_cache"):
        memory._stats_cache.clear()

    try:
        # cases 端点降级返回 200+X-Degraded
        resp = client.get("/api/v1/cases", params={"user_id": "admin"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Degraded") == "true"
    finally:
        memory.db = original_db

    stats = sla_monitor.get_stats(window_minutes=60)
    assert stats["degraded_count"] >= 1


def test_sla_middleware_records_503():
    """中间件记录 503 硬降级"""
    from agent.sla import sla_monitor
    client, routes = _get_client()

    sla_monitor.reset()
    # mock memory 抛异常触发全局异常处理
    original_stats = routes.memory.get_memory_stats
    routes.memory.get_memory_stats = unittest.mock.MagicMock(
        side_effect=RuntimeError("forced 503")
    )
    try:
        resp = client.get("/api/v1/memory/stats", params={"user_id": "admin"})
        assert resp.status_code == 503
    finally:
        routes.memory.get_memory_stats = original_stats

    stats = sla_monitor.get_stats(window_minutes=60)
    assert stats["degraded_count"] >= 1
    # 503 算不可用
    assert stats["availability"] < 1.0


# ===== 4. 99.5% 可用性达标验证 =====

def test_sla_99_5_target_met():
    """99.5% 可用性达标验证（正常流量 + 降级流量）"""
    import sqlite3
    from agent.sla import sla_monitor
    client, routes = _get_client()
    memory = routes.memory

    sla_monitor.reset()

    # 阶段 1: 100 次正常请求
    for _ in range(100):
        client.get("/health")

    # 阶段 2: 50 次降级请求（200+degraded，不算 5xx）
    original_db = memory.db
    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked")
    mock_db.commit.side_effect = sqlite3.OperationalError("database is locked")
    memory.db = mock_db
    if hasattr(memory, "_stats_cache"):
        memory._stats_cache.clear()

    for _ in range(50):
        client.get("/api/v1/cases", params={"user_id": "admin"})

    memory.db = original_db

    # 清理可能的 feedback 队列
    routes._feedback_queue[:] = []

    # 验证：所有请求 < 500，availability = 100%
    stats = sla_monitor.get_stats(window_minutes=60)
    assert stats["total_requests"] >= 150
    assert stats["availability"] == 1.0
    assert stats["sla_met"] is True
    assert stats["degraded_count"] >= 50
