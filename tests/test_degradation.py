"""
M4-14 降级策略测试

验证：
1. API 端点故障降级（cases/dashboard/feedback 返回 200+degraded 而非 500）
2. 全局异常处理器（未处理异常返回 503+degraded）
3. SQLite 初始化容错（降级为内存数据库）
4. WAL 模式启用
5. get_line_stats TTL 缓存
6. LLM max_retries
"""
import sqlite3
import sys
import time
import unittest.mock
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===== 1. cases 端点 SQLite 故障降级 =====

def test_cases_sqlite_degradation():
    """SQLite 故障时 cases 端点返回 200+空结果+degraded"""
    from fastapi.testclient import TestClient
    import api.routes

    client = TestClient(api.routes.app)
    memory = api.routes.memory
    original_db = memory.db

    # 注入 SQLite 故障
    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked")
    memory.db = mock_db

    try:
        resp = client.get("/api/v1/cases", params={"user_id": "admin", "limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["cases"] == []
        assert data.get("degraded") is True
    finally:
        memory.db = original_db


# ===== 2. dashboard 端点 SQLite 故障降级 =====

def test_dashboard_sqlite_degradation():
    """SQLite 故障时 dashboard 端点返回 200+零值统计+degraded"""
    from fastapi.testclient import TestClient
    import api.routes

    client = TestClient(api.routes.app)
    memory = api.routes.memory
    # 清除缓存避免干扰
    if hasattr(memory, "_stats_cache"):
        memory._stats_cache.clear()
    original_db = memory.db

    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked")
    memory.db = mock_db

    try:
        resp = client.get("/api/v1/dashboard/overview", params={"user_id": "admin", "days": 30})
        assert resp.status_code == 200
        data = resp.json()
        assert "lines" in data
        # 每个产线都应降级
        for line in data["lines"]:
            assert line.get("degraded") is True
            assert line["episodic_count"] == 0
    finally:
        memory.db = original_db


# ===== 3. feedback 端点 SQLite 故障降级 =====

def test_feedback_sqlite_degradation():
    """SQLite 故障时 feedback 端点返回 200+degraded+暂存队列"""
    from fastapi.testclient import TestClient
    import api.routes

    client = TestClient(api.routes.app)
    memory = api.routes.memory
    original_db = memory.db
    original_queue_len = len(api.routes._feedback_queue)

    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked")
    mock_db.commit.side_effect = sqlite3.OperationalError("database is locked")
    memory.db = mock_db

    try:
        resp = client.post("/api/v1/feedback", json={
            "proposal_id": "test_deg_proposal",
            "user_id": "admin",
            "action": "adopted",
            "score": 0.8,
            "comment": "降级测试",
            "line_id": "heat_treatment",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data.get("degraded") is True
        # 队列应有新增
        assert len(api.routes._feedback_queue) > original_queue_len
    finally:
        memory.db = original_db
        # 清理测试数据
        api.routes._feedback_queue[:] = [
            x for x in api.routes._feedback_queue
            if x.get("comment") != "降级测试"
        ]


# ===== 4. 全局异常处理器 =====

def test_global_exception_handler():
    """未处理异常返回 503+degraded（memory/stats 端点无 try/except）"""
    from fastapi.testclient import TestClient
    import api.routes

    client = TestClient(api.routes.app)
    memory = api.routes.memory
    original_db = memory.db

    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = RuntimeError("unexpected error")
    memory.db = mock_db

    try:
        resp = client.get("/api/v1/memory/stats", params={"user_id": "admin"})
        # 全局异常处理器兜底，返回 503
        assert resp.status_code == 503
        data = resp.json()
        assert data.get("degraded") is True
        assert "trace_id" in data
    finally:
        memory.db = original_db


# ===== 5. SQLite 初始化容错 =====

def test_sqlite_init_fallback():
    """SQLite 初始化失败时降级为内存数据库（不崩溃）"""
    from agent.memory.memory_service import MemoryService

    # mock 前保存真实的 sqlite3.connect，避免 side_effect 内递归
    real_connect = sqlite3.connect
    with unittest.mock.patch("sqlite3.connect") as mock_connect:
        call_count = [0]

        def side_effect(path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise sqlite3.OperationalError("disk full")
            # 第二次（:memory:）用真实函数返回连接
            return real_connect(":memory:")

        mock_connect.side_effect = side_effect
        svc = MemoryService(db_path="data/_test_fallback.db")

    # 应降级为内存数据库，能正常查询
    assert svc.db is not None
    svc.db.execute("SELECT 1").fetchone()


# ===== 6. WAL 模式 =====

def test_wal_mode_enabled():
    """SQLite 启用 WAL 模式"""
    from api.routes import memory
    cur = memory.db.execute("PRAGMA journal_mode")
    mode = cur.fetchone()[0]
    # WAL 模式下返回 "wal"（某些环境可能是 "wal" 小写）
    assert mode.lower() == "wal"


# ===== 7. get_line_stats TTL 缓存 =====

def test_get_line_stats_cache():
    """get_line_stats 60 秒内返回缓存"""
    from api.routes import memory
    # 清除缓存
    if hasattr(memory, "_stats_cache"):
        memory._stats_cache.clear()

    # 第一次调用（实际查询）
    stats1 = memory.get_line_stats("heat_treatment", days=30)
    assert "line_id" in stats1

    # 第二次调用（应从缓存返回）
    stats2 = memory.get_line_stats("heat_treatment", days=30)
    assert stats2["line_id"] == stats1["line_id"]
    assert stats2["episodic_count"] == stats1["episodic_count"]

    # 验证缓存确实存在
    assert hasattr(memory, "_stats_cache")
    assert ("heat_treatment", 30) in memory._stats_cache


# ===== 8. LLM max_retries =====

def test_llm_max_retries():
    """get_llm 返回的客户端配置了 max_retries"""
    from agent.utils import get_llm
    llm = get_llm("planner")
    # ChatOpenAI 的 max_retries 属性
    assert llm.max_retries == 3
