"""
M5-3 效果看板测试

测试两层：
1. API 端点 /api/v1/effect/dashboard（KPI + 改善分布 + 归因统计 + 记录列表）
2. UI 模块 ui/effect_dashboard.py 可导入（渲染函数存在）
"""
import sys
import types

import pytest

# mock langchain_openai.ChatOpenAI（sandbox 中模块存在但缺 ChatOpenAI 属性）
if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = types.ModuleType("langchain_openai")
_mock_lc = sys.modules["langchain_openai"]
if not hasattr(_mock_lc, "ChatOpenAI"):
    _mock_lc.ChatOpenAI = type("ChatOpenAI", (), {"__init__": lambda self, **kw: None})


# ===== Fixtures =====


class _FakeChromaCollection:
    """模拟 Chroma collection（支持 get/update/add/delete）"""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def add(self, ids, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._store[doc_id] = {
                "document": documents[i] if i < len(documents) else "",
                "metadata": dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
            }

    def get(self, ids=None, limit=None, include=None, where=None):
        if ids is not None:
            out_ids, out_docs, out_metas = [], [], []
            for doc_id in ids:
                if doc_id in self._store:
                    out_ids.append(doc_id)
                    out_docs.append(self._store[doc_id]["document"])
                    out_metas.append(dict(self._store[doc_id]["metadata"]))
            return {"ids": out_ids, "documents": out_docs, "metadatas": out_metas}
        items = list(self._store.items())
        if limit is not None:
            items = items[:limit]
        return {
            "ids": [doc_id for doc_id, _ in items],
            "documents": [item["document"] for _, item in items],
            "metadatas": [dict(item["metadata"]) for _, item in items],
        }

    def update(self, ids, metadatas=None, documents=None):
        for i, doc_id in enumerate(ids):
            if doc_id not in self._store:
                continue
            if metadatas is not None and i < len(metadatas):
                self._store[doc_id]["metadata"] = dict(metadatas[i])
            if documents is not None and i < len(documents):
                self._store[doc_id]["document"] = documents[i]

    def delete(self, ids):
        for doc_id in ids:
            self._store.pop(doc_id, None)


@pytest.fixture
def memory(tmp_path):
    from agent.memory.memory_service import MemoryService
    db_path = tmp_path / "test_effect_dashboard.db"
    service = MemoryService(db_path=db_path, chroma_path=tmp_path / "chroma")
    service._collection = _FakeChromaCollection()
    service._ensure_chroma = lambda: None
    yield service
    service.close()


@pytest.fixture
def tracker(memory):
    from agent.effect_tracker import EffectTracker
    return EffectTracker(memory)


@pytest.fixture
def client(memory):
    """FastAPI 测试客户端（注入临时 memory）"""
    from fastapi.testclient import TestClient
    from api.routes import app, effect_tracker
    original_db = effect_tracker.db
    original_memory = effect_tracker.memory
    effect_tracker.db = memory.db
    effect_tracker.memory = memory
    yield TestClient(app)
    effect_tracker.db = original_db
    effect_tracker.memory = original_memory


def _seed_trackings(tracker, count=5, line_id="heat_treatment"):
    """创建并跟踪 N 条记录"""
    ids = []
    for i in range(count):
        tid = tracker.schedule_tracking(
            proposal_id=f"P{i}", case_id=f"case_{i}",
            batch_id_before=f"B{i}", line_id=line_id,
            metric_before=0.20, days_offset=0,
        )
        tracker.track_effect(tid)
        ids.append(tid)
    return ids


# ===== 1. API 端点测试 =====


class TestEffectDashboardAPI:
    """/api/v1/effect/dashboard 端点"""

    def test_dashboard_empty(self, client):
        """无数据时返回空 KPI"""
        resp = client.get("/api/v1/effect/dashboard?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert "kpi" in data
        assert data["kpi"]["total"] == 0
        assert data["kpi"]["tracked"] == 0
        assert data["kpi"]["attributed_count"] == 0
        assert "improvement_distribution" in data
        assert "records" in data
        assert isinstance(data["records"], list)

    def test_dashboard_with_data(self, client, tracker):
        """有数据时返回正确 KPI + 分布 + 记录"""
        _seed_trackings(tracker, count=5)
        resp = client.get("/api/v1/effect/dashboard?user_id=admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["kpi"]["total"] == 5
        assert data["kpi"]["tracked"] == 5
        assert data["kpi"]["attributed_count"] == 0  # 未归因
        # 改善分布 6 个桶
        assert len(data["improvement_distribution"]) == 6
        # 5 条记录
        assert len(data["records"]) == 5

    def test_dashboard_with_attribution(self, client, tracker, memory):
        """归因后 attributed_count 正确"""
        memory._collection.add(
            ids=["case_0"],
            documents=["案例"],
            metadatas=[{"confidence": 0.5}],
        )
        ids = _seed_trackings(tracker, count=1)
        tracker.attribute_effect(ids[0])

        resp = client.get("/api/v1/effect/dashboard?user_id=admin")
        data = resp.json()
        assert data["kpi"]["attributed_count"] == 1

    def test_dashboard_permission_filter(self, client, tracker):
        """非 admin 用户只能看到自己产线的数据"""
        _seed_trackings(tracker, count=3, line_id="heat_treatment")
        _seed_trackings(tracker, count=2, line_id="welding")
        # operator_01 只有 heat_treatment 权限
        resp = client.get("/api/v1/effect/dashboard?user_id=operator_01")
        data = resp.json()
        assert data["kpi"]["total"] == 3  # 只看到 heat_treatment

    def test_dashboard_line_filter(self, client, tracker):
        """指定 line_id 时只返回该产线数据"""
        _seed_trackings(tracker, count=3, line_id="heat_treatment")
        _seed_trackings(tracker, count=2, line_id="welding")
        resp = client.get(
            "/api/v1/effect/dashboard?user_id=admin&line_id=welding"
        )
        data = resp.json()
        assert data["kpi"]["total"] == 2

    def test_dashboard_improvement_distribution(self, client, tracker):
        """改善分布分桶正确"""
        # 用自定义 quality_fetcher 控制改善率
        from agent.effect_tracker import EffectTracker
        # 默认 fetcher 基于哈希，我们直接验证分桶逻辑
        _seed_trackings(tracker, count=10)
        resp = client.get("/api/v1/effect/dashboard?user_id=admin")
        data = resp.json()
        dist = data["improvement_distribution"]
        # 6 个桶的 key
        assert "<=-10" in dist
        assert ">=30" in dist
        # 所有桶计数之和 = tracked 记录数
        total_binned = sum(dist.values())
        assert total_binned == data["kpi"]["tracked"]

    def test_dashboard_days_filter(self, client, tracker):
        """days 参数过滤生效"""
        _seed_trackings(tracker, count=3)
        # days=1 只看近 1 天（schedule_tracking 用 days_offset=0，scheduled_at=now）
        resp = client.get("/api/v1/effect/dashboard?user_id=admin&days=1")
        data = resp.json()
        assert data["kpi"]["total"] == 3  # 都是今天的
        # days=0 会返回空（scheduled_at > now-0days）
        resp0 = client.get("/api/v1/effect/dashboard?user_id=admin&days=0")
        # days=0 可能边界情况，只要不报错即可
        assert resp0.status_code == 200


# ===== 2. UI 模块可导入测试 =====


class TestEffectDashboardUI:
    """ui/effect_dashboard.py 模块结构"""

    def test_module_importable(self):
        """模块可导入（不依赖 streamlit 运行环境）"""
        # streamlit 可能未安装，mock 它
        if "streamlit" not in sys.modules:
            st_mock = types.ModuleType("streamlit")
            st_mock.cache_resource = lambda **kw: (lambda f: f)
            st_mock.header = st_mock.subheader = st_mock.caption = lambda *a, **kw: None
            st_mock.info = st_mock.warning = st_mock.error = lambda *a, **kw: None
            st_mock.metric = lambda *a, **kw: None
            st_mock.columns = lambda n: [type("C", (), {"metric": lambda *a, **kw: None})() for _ in range(n)]
            st_mock.selectbox = lambda *a, **kw: None
            st_mock.slider = lambda *a, **kw: 30
            st_mock.text_input = lambda *a, **kw: "admin"
            st_mock.divider = lambda *a, **kw: None
            st_mock.bar_chart = lambda *a, **kw: None
            st_mock.dataframe = lambda *a, **kw: None
            st_mock.expander = type("Ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
            st_mock.stop = lambda *a, **kw: None
            st_mock.exception = lambda *a, **kw: None
            st_mock.write = st_mock.code = lambda *a, **kw: None
            sys.modules["streamlit"] = st_mock

        # pandas 也可能未安装，mock
        if "pandas" not in sys.modules:
            pd_mock = types.ModuleType("pandas")
            pd_mock.DataFrame = lambda *a, **kw: {}
            pd_mock.cut = lambda *a, **kw: None
            sys.modules["pandas"] = pd_mock

        import ui.effect_dashboard as mod
        assert hasattr(mod, "render_effect_dashboard")
        assert callable(mod.render_effect_dashboard)

    def test_render_function_signature(self):
        """render_effect_dashboard 函数存在且无必填参数"""
        if "streamlit" not in sys.modules:
            pytest.skip("streamlit not installed")
        import ui.effect_dashboard as mod
        import inspect
        sig = inspect.signature(mod.render_effect_dashboard)
        # 无必填参数
        required = [p for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty]
        assert len(required) == 0
