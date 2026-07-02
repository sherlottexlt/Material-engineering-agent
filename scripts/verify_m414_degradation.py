"""M4-14 验证：直接测试 sqlite_failure 场景的 3 个弱点端点"""
import logging
import sys
import types
import sqlite3
import unittest.mock
import json
from pathlib import Path

logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

client = TestClient(api.routes.app)
memory = api.routes.memory

# SQLite 多线程安全
try:
    memory.db.close()
    memory.db = sqlite3.connect(str(memory.db_path), check_same_thread=False)
except Exception:
    pass

results = []

def test_endpoint(name, method, path, **kwargs):
    """测试单个端点在 SQLite 故障下的降级"""
    original_db = memory.db
    mock_db = unittest.mock.MagicMock()
    mock_db.execute.side_effect = sqlite3.OperationalError("database is locked")
    mock_db.commit.side_effect = sqlite3.OperationalError("database is locked")
    memory.db = mock_db
    # 清除缓存
    if hasattr(memory, "_stats_cache"):
        memory._stats_cache.clear()
    try:
        if method == "GET":
            resp = client.get(path, params=kwargs.get("params", {"user_id": "admin"}))
        elif method == "POST":
            resp = client.post(path, json=kwargs.get("json"))
        status = resp.status_code
        body = resp.json()
        degraded = body.get("degraded", False) if isinstance(body, dict) else False
        # 降级判定：2xx + degraded = degraded_ok；5xx = failed
        if 200 <= status < 300 and degraded:
            classification = "degraded_ok"
        elif 200 <= status < 300:
            classification = "normal"
        else:
            classification = "failed"
        results.append({
            "endpoint": name,
            "status_code": status,
            "degraded": degraded,
            "classification": classification,
        })
    except Exception as e:
        results.append({
            "endpoint": name,
            "status_code": -1,
            "degraded": False,
            "classification": "failed",
            "error": str(e)[:100],
        })
    finally:
        memory.db = original_db

# 测试 3 个弱点端点
test_endpoint("cases", "GET", "/api/v1/cases")
test_endpoint("dashboard", "GET", "/api/v1/dashboard/overview")
test_endpoint("feedback", "POST", "/api/v1/feedback", json={
    "proposal_id": "test_verify",
    "user_id": "admin",
    "action": "adopted",
    "score": 0.8,
    "comment": "验证",
    "line_id": "heat_treatment",
})

# 清理 feedback 队列
api.routes._feedback_queue[:] = [
    x for x in api.routes._feedback_queue if x.get("comment") != "验证"
]

# 计算降级率
total = len(results)
degraded_ok = sum(1 for r in results if r["classification"] == "degraded_ok")
rate = degraded_ok / total if total > 0 else 0

# 写报告
report = {
    "scenario": "sqlite_failure",
    "m4_14_fix": "cases/dashboard/feedback 端点加 try/except + 降级返回",
    "total_endpoints": total,
    "degraded_ok": degraded_ok,
    "failed": sum(1 for r in results if r["classification"] == "failed"),
    "degradation_rate": round(rate, 4),
    "results": results,
    "acceptance": "✅ 达标" if rate >= 0.9 else "❌ 未达标",
}

json_path = PROJECT_ROOT / "data" / "m414_verification.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# 写简短文本
txt_path = PROJECT_ROOT / "data" / "m414_verification.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("M4-14 降级验证（sqlite_failure 场景）\n\n")
    for r in results:
        f.write(f"{r['endpoint']}: status={r['status_code']}, degraded={r['degraded']}, "
                f"classification={r['classification']}\n")
    f.write(f"\n降级率: {degraded_ok}/{total} = {rate*100:.1f}%\n")
    f.write(f"验收: {report['acceptance']}\n")

print(f"Done: {degraded_ok}/{total} = {rate*100:.1f}%")
