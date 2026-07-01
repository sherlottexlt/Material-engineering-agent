"""
MetaCraft Agent 工具函数（M0 阶段）

M0 阶段使用直接函数调用代替 MCP，返回模拟数据。
M1 阶段切换为真正的 MCP Server 调用。

工具清单：
- query_batch_params: 查询批次工艺参数
- query_defect_history: 查询历史缺陷记录
- run_metallurgy_model: 调用机理模型
- search_handbook: 检索工艺手册
- search_cases: 检索历史案例
"""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from agent.memory.memory_service import MemoryService


# ===== 种子案例加载（评估用）=====
_SEED_CASES_PATH = Path(__file__).parent.parent / "data" / "seed_cases" / "seed_cases.json"
_SEED_BATCH_PARAMS: dict[str, dict] = {}


def _load_seed_cases():
    """惰性加载种子案例的批次参数到内存索引

    评估时 query_batch_params 优先查这里，确保 50 条用例各有不同参数。
    """
    global _SEED_BATCH_PARAMS
    if _SEED_BATCH_PARAMS:
        return
    if not _SEED_CASES_PATH.exists():
        return
    try:
        with open(_SEED_CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data.get("cases", []):
            params = case.get("batch_params") or {}
            bid = params.get("batch_id") or case.get("batch_id")
            if bid:
                _SEED_BATCH_PARAMS[bid] = params
        logger.info(f"[Tool] 已加载 {len(_SEED_BATCH_PARAMS)} 条种子案例批次参数")
    except Exception as e:
        logger.warning(f"[Tool] 加载种子案例失败: {e}")


# ===== 模拟数据库 =====

_MOCK_BATCHES = {
    "B20260628-A": {
        "batch_id": "B20260628-A",
        "process_type": "heat_treatment",
        "temperature": 830,
        "holding_time": 90,
        "cooling_rate": 3.0,
        "pressure": None,
        "raw_material_batch": "RM-2026-0042",
        "start_time": "2026-06-28T08:00:00",
        "end_time": "2026-06-28T09:30:00",
    },
    "B20260620-A": {
        "batch_id": "B20260620-A",
        "process_type": "heat_treatment",
        "temperature": 850,
        "holding_time": 60,
        "cooling_rate": 8.0,
        "raw_material_batch": "RM-2026-0038",
        "start_time": "2026-06-20T08:00:00",
        "end_time": "2026-06-20T09:00:00",
    },
}

_MOCK_DEFECTS = [
    {
        "record_id": "DR-20260620-001",
        "batch_id": "B20260620-A",
        "defect_type": "hardness_low",
        "measured_value": 52.3,
        "standard_value": 58.0,
        "root_cause": "保温时间不足",
        "solution": "保温时间 +15 分钟",
        "created_at": "2026-06-20T14:00:00",
    },
    {
        "record_id": "DR-20260615-003",
        "batch_id": "B20260615-C",
        "defect_type": "hardness_low",
        "measured_value": 54.1,
        "standard_value": 58.0,
        "root_cause": "冷却速率过低",
        "solution": "提高冷却速率至 10 ℃/s",
        "created_at": "2026-06-15T16:00:00",
    },
    {
        "record_id": "DR-20260610-002",
        "batch_id": "B20260610-B",
        "defect_type": "hardness_low",
        "measured_value": 53.0,
        "standard_value": 58.0,
        "root_cause": "温度偏低",
        "solution": "温度 +20℃",
        "created_at": "2026-06-10T11:00:00",
    },
]

_MOCK_HANDBOOK = [
    {
        "source": "热处理工艺手册 第3版 - 4.2节",
        "content": "45钢调质处理：淬火温度 840±10℃，保温时间按 1.5min/mm 计算（有效厚度），冷却介质为水或油，冷却速率需 ≥ 5℃/s 以保证马氏体转变充分。",
    },
    {
        "source": "热处理工艺手册 第3版 - 4.3节",
        "content": "回火硬度与保温时间关系：保温时间不足会导致碳化物析出不充分，硬度偏低。最小保温时间应 ≥ 60min（有效厚度 ≤ 50mm）。",
    },
    {
        "source": "金属热处理工艺学 - 第6章",
        "content": "JMAK 方程描述奥氏体转变：f = 1 - exp(-k * t^n)，其中 k 受温度影响显著。温度偏低 20℃ 可使转变分数下降 15-25%。",
    },
]

# ===== 手册索引（由 ingest_handbooks.py 生成）=====
_HANDBOOK_INDEX_PATH = Path(__file__).parent.parent / "data" / "handbook_index.json"
_handbook_index_cache: dict | None = None
_handbook_index_loaded = False


def _load_handbook_index():
    """惰性加载手册索引（仅加载一次）"""
    global _handbook_index_cache, _handbook_index_loaded
    if _handbook_index_loaded:
        return _handbook_index_cache
    _handbook_index_loaded = True
    if not _HANDBOOK_INDEX_PATH.exists():
        return None
    try:
        with open(_HANDBOOK_INDEX_PATH, "r", encoding="utf-8") as f:
            _handbook_index_cache = json.load(f)
        logger.info(
            f"[Tool] 手册索引已加载: {_handbook_index_cache.get('total_chunks', 0)} chunks, "
            f"来源 {_handbook_index_cache.get('source_files', [])}"
        )
    except Exception as e:
        logger.warning(f"[Tool] 加载手册索引失败: {e}")
        _handbook_index_cache = None
    return _handbook_index_cache


def _tokenize_query(query: str) -> list[str]:
    """将查询切分为关键词（支持中英文混合）

    策略：
    - 按空格/标点切分得到"词"
    - 中文连续片段再切分为 2-gram（提升中文检索效果）
    - 去重、过滤空串
    """
    import re
    # 按非字母数字汉字切分
    tokens = re.split(r"[^\w\u4e00-\u9fff]+", query)
    result = []
    seen = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok or len(tok) < 1:
            continue
        # 英文/数字整体保留
        if re.match(r"^[a-zA-Z0-9_]+$", tok):
            if tok not in seen:
                result.append(tok)
                seen.add(tok)
        else:
            # 中文：整体 + 2-gram
            if tok not in seen:
                result.append(tok)
                seen.add(tok)
            if len(tok) >= 2:
                for i in range(len(tok) - 1):
                    gram = tok[i:i + 2]
                    if gram not in seen:
                        result.append(gram)
                        seen.add(gram)
    return result


def _bm25_like_score(query_tokens: list[str], doc: str) -> float:
    """BM25-like 评分（简化版，无需外部库）

    评分 = Σ (token 在 doc 中出现次数 / (doc_len * avgdl)) * idf
    简化为：子串匹配 + 出现次数加权
    """
    if not doc:
        return 0.0
    doc_len = len(doc)
    score = 0.0
    for tok in query_tokens:
        if not tok:
            continue
        # 子串匹配计数
        count = doc.count(tok)
        if count > 0:
            # 短 token 权重低（避免单字噪声），长 token 权重高
            tok_weight = min(len(tok) / 2.0, 2.0)
            # 出现次数有递减收益
            freq_score = (count / (count + 1.5))
            score += tok_weight * freq_score
    # 按文档长度归一化（短文档命中更相关）
    norm = 1.0 / (1.0 + doc_len / 1000.0)
    return score * norm


# ===== 工具实现 =====

def query_batch_params(batch_id: str) -> dict:
    """查询批次工艺参数

    查询优先级：
    1. 种子案例库（评估用，确保 50 条用例参数各异）
    2. _MOCK_BATCHES（M0 内置示例数据）
    3. 默认数据（未知批次兜底）

    Args:
        batch_id: 批次编号

    Returns:
        批次工艺参数字典
    """
    logger.info(f"[Tool] query_batch_params: {batch_id}")

    # 1. 优先查种子案例库
    _load_seed_cases()
    if batch_id in _SEED_BATCH_PARAMS:
        result = _SEED_BATCH_PARAMS[batch_id].copy()
        result["_source"] = "seed_case"
        return result

    # 2. 查内置 mock 数据
    if batch_id in _MOCK_BATCHES:
        result = _MOCK_BATCHES[batch_id].copy()
        result["_source"] = "mock_data"
        return result

    # 3. 未知批次：生成一个默认的偏低参数（模拟问题批次）
    logger.warning(f"[Tool] 批次 {batch_id} 不在模拟库中，使用默认数据")
    return {
        "batch_id": batch_id,
        "process_type": "heat_treatment",
        "temperature": 830,
        "holding_time": 90,
        "cooling_rate": 3.0,
        "raw_material_batch": f"RM-{batch_id[-4:]}",
        "start_time": "2026-06-28T08:00:00",
        "end_time": "2026-06-28T09:30:00",
        "_source": "mock_default",
        "_note": f"批次 {batch_id} 使用默认模拟数据",
    }


def query_defect_history(
    defect_type: Optional[str] = None,
    days_back: int = 30,
    limit: int = 50,
) -> dict:
    """查询历史缺陷记录

    Args:
        defect_type: 缺陷类型过滤
        days_back: 查询天数
        limit: 返回条数上限

    Returns:
        缺陷记录字典
    """
    logger.info(f"[Tool] query_defect_history: type={defect_type}, days={days_back}")

    records = _MOCK_DEFECTS.copy()
    if defect_type:
        records = [r for r in records if r["defect_type"] == defect_type]

    return {
        "total": len(records),
        "records": records[:limit],
        "_source": "mock_data",
    }


def run_metallurgy_model(
    model_type: str,
    temperature: Optional[float] = None,
    holding_time: Optional[float] = None,
    cooling_rate: Optional[float] = None,
    grain_size: Optional[float] = None,
) -> dict:
    """调用材料机理模型

    Args:
        model_type: 模型类型 (jmak / hall_petch / cooling_rate)
        temperature: 温度 (℃)
        holding_time: 保温时间 (分钟)
        cooling_rate: 冷却速率 (℃/s)
        grain_size: 晶粒尺寸 (μm)

    Returns:
        模型预测结果
    """
    logger.info(f"[Tool] run_metallurgy_model: type={model_type}")

    if model_type == "jmak":
        if temperature is None or holding_time is None:
            return {"error": "JMAK 模型需要 temperature 和 holding_time"}
        return _jmak_model(temperature, holding_time)

    elif model_type == "cooling_rate":
        if cooling_rate is None:
            return {"error": "冷却速率模型需要 cooling_rate"}
        return _cooling_rate_model(cooling_rate)

    elif model_type == "hall_petch":
        if grain_size is None:
            return {"error": "Hall-Petch 模型需要 grain_size"}
        return _hall_petch_model(grain_size)

    return {"error": f"未知模型类型: {model_type}"}


def _jmak_model(temperature: float, holding_time: float) -> dict:
    """JMAK 方程：预测相变分数与硬度"""
    k = 0.01
    n = 1.5
    temp_factor = max(0.1, (temperature - 700) / 200)

    fraction = 1 - math.exp(-k * (holding_time ** n) * temp_factor)
    predicted_hardness = 40 + 20 * fraction

    return {
        "model": "JMAK",
        "inputs": {"temperature": temperature, "holding_time": holding_time},
        "outputs": {
            "transformation_fraction": round(fraction, 4),
            "predicted_hardness_HRc": round(predicted_hardness, 2),
        },
        "parameters": {"k": k, "n": n, "temp_factor": round(temp_factor, 4)},
        "note": "简化模型，参数需根据实际材料标定",
    }


def _cooling_rate_model(cooling_rate: float) -> dict:
    """冷却速率与硬度关系"""
    base_hardness = 45
    hardness_increase = cooling_rate * 0.8

    return {
        "model": "cooling_rate",
        "inputs": {"cooling_rate": cooling_rate},
        "outputs": {
            "estimated_hardness_HRc": round(base_hardness + hardness_increase, 2),
        },
        "note": "简化线性模型",
    }


def _hall_petch_model(grain_size: float) -> dict:
    """Hall-Petch 关系"""
    sigma_0 = 200
    k = 0.5
    strength = sigma_0 + k * (grain_size ** -0.5)

    return {
        "model": "Hall-Petch",
        "inputs": {"grain_size_um": grain_size},
        "outputs": {
            "yield_strength_MPa": round(strength, 2),
        },
    }


def search_handbook(query: str, top_k: int = 3) -> dict:
    """检索工艺手册

    优先使用 ingest_handbooks.py 生成的关键词索引（BM25-like 评分），
    无索引时回退到内置 mock 数据。

    Args:
        query: 检索关键词
        top_k: 返回条数

    Returns:
        匹配的手册片段
    """
    logger.info(f"[Tool] search_handbook: {query}")

    # 1. 优先查手册索引
    index = _load_handbook_index()
    if index and index.get("chunks"):
        tokens = _tokenize_query(query)
        scored = []
        for chunk in index["chunks"]:
            content = chunk.get("content", "")
            score = _bm25_like_score(tokens, content)
            if score > 0:
                # 构造来源标签：文件名 + section（如果有）
                source_parts = [chunk.get("source", "")]
                if chunk.get("section"):
                    source_parts.append(chunk["section"])
                if chunk.get("page"):
                    source_parts.append(f"p{chunk['page']}")
                source_label = " - ".join(str(p) for p in source_parts if p)
                scored.append({
                    "source": source_label,
                    "content": content,
                    "relevance_score": round(score, 4),
                    "chunk_id": chunk.get("chunk_id", ""),
                })

        if scored:
            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            return {
                "total": len(scored[:top_k]),
                "results": scored[:top_k],
                "_source": "handbook_index",
            }
        # 索引存在但无命中，继续走 fallback

    # 2. 回退：mock 数据关键词匹配
    hits = []
    for item in _MOCK_HANDBOOK:
        score = sum(1 for kw in query.split() if kw in item["content"])
        if score > 0 or len(hits) < top_k:
            hits.append({**item, "relevance_score": score})

    hits.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {
        "total": len(hits[:top_k]),
        "results": hits[:top_k],
        "_source": "mock_data",
    }


# ===== 记忆服务（延迟初始化，与 memory_writer 共享数据路径）=====
_memory_service: MemoryService | None = None


def _get_memory_service() -> MemoryService:
    """惰性获取 MemoryService 单例（与 memory_writer 共用同一数据路径）"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


def _set_memory_service(service: MemoryService | None) -> None:
    """注入 MemoryService 实例（测试用，避免污染真实库）"""
    global _memory_service
    _memory_service = service


def search_cases(query: str, top_k: int = 3) -> dict:
    """检索历史案例（优先走长期记忆语义检索，降级到 mock 关键词匹配）

    Args:
        query: 检索关键词 / 自然语言描述
        top_k: 返回条数

    Returns:
        匹配的历史案例，_source 标识来源：semantic_memory / mock_data
    """
    logger.info(f"[Tool] search_cases: {query}")

    # 1. 优先走长期记忆（Chroma 语义检索）
    try:
        memory = _get_memory_service()
        results = memory.search_semantic(query, top_k=top_k)
        if results:
            formatted = []
            for r in results:
                doc = r.get("document") or ""
                meta = r.get("metadata") or {}
                # document 格式: "defect_type\nroot_cause\nsolution"
                parts = doc.split("\n", 2)
                defect_type = parts[0] if len(parts) > 0 else meta.get("defect_type", "unknown")
                root_cause = parts[1] if len(parts) > 1 else ""
                solution = parts[2] if len(parts) > 2 else ""
                distance = r.get("distance")
                formatted.append({
                    "record_id": r.get("id"),
                    "defect_type": defect_type,
                    "root_cause": root_cause,
                    "solution": solution,
                    "confidence": meta.get("confidence"),
                    "created_at": meta.get("created_at"),
                    "source": meta.get("source", "auto"),
                    # 距离越小相关度越高，转换为 0-1 的相关度评分
                    "relevance_score": round(1.0 - distance, 3) if distance is not None else None,
                })
            logger.info(f"[Tool] search_cases 语义检索命中 {len(formatted)} 条")
            return {
                "total": len(formatted),
                "results": formatted,
                "_source": "semantic_memory",
            }
    except Exception as e:
        logger.warning(f"[Tool] 语义检索异常，降级到 mock: {e}")

    # 2. 降级：mock 关键词匹配（保持向后兼容，Chroma 不可用时走这里）
    hits = []
    for defect in _MOCK_DEFECTS:
        text = f"{defect['defect_type']} {defect['root_cause']} {defect['solution']}"
        score = sum(1 for kw in query.split() if kw in text)
        if score > 0:
            hits.append({**defect, "relevance_score": score})

    hits.sort(key=lambda x: x["relevance_score"], reverse=True)

    # 如果没匹配到，返回全部
    if not hits:
        hits = _MOCK_DEFECTS[:top_k]

    return {
        "total": len(hits[:top_k]),
        "results": hits[:top_k],
        "_source": "mock_data",
    }


# ===== 工具注册表（便于动态调用）=====

TOOL_REGISTRY = {
    "query_batch_params": query_batch_params,
    "query_defect_history": query_defect_history,
    "run_metallurgy_model": run_metallurgy_model,
    "search_handbook": search_handbook,
    "search_cases": search_cases,
}


def call_tool(name: str, **kwargs) -> dict:
    """统一工具调用入口

    Args:
        name: 工具名
        **kwargs: 工具参数

    Returns:
        工具返回结果
    """
    if name not in TOOL_REGISTRY:
        return {"error": f"未知工具: {name}, 可用: {list(TOOL_REGISTRY.keys())}"}

    try:
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        logger.error(f"[Tool] 工具 {name} 调用失败: {e}")
        return {"error": str(e)}
