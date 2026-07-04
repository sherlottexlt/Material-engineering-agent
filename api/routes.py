"""
MetaCraft Agent REST API
对应 TDD 第 5 节

端点：
- POST /api/v1/analyze     提交归因请求（同步执行）
- GET  /api/v1/result/{id} 查询归因结果
- POST /api/v1/feedback    提交用户反馈
- GET  /api/v1/cases       查询案例库
- WS   /api/v1/stream      实时推送执行过程
"""
import json
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from agent.memory.memory_service import MemoryService
from agent.orchestrator import build_orchestrator
from agent.sla import sla_monitor
from agent.utils import (
    can_access_line,
    get_user_permissions,
    get_user_lines,
    list_available_lines,
    setup_tracing,
)
from models.entities import (
    BatchParams,
    CaseRecord,
    DefectType,
    ProcessType,
    UserFeedback,
)
from models.state import AgentState

# 初始化 trace
setup_tracing()

app = FastAPI(
    title="MetaCraft Agent API",
    description="材料加工产线智能工艺优化 Agent",
    version="0.1.0",
)

# CORS（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# M4-15: 统一降级响应 helper（设置 X-Degraded header 供 SLA 中间件统计）
def _degraded_response(content: dict, status_code: int = 200) -> JSONResponse:
    """构造降级响应，标记 X-Degraded header

    Args:
        content: 响应体（应包含 degraded: True 字段）
        status_code: HTTP 状态码（200 用于软降级，503 用于硬降级）

    Returns:
        JSONResponse，带 X-Degraded: true header
    """
    resp = JSONResponse(status_code=status_code, content=content)
    resp.headers["X-Degraded"] = "true"
    return resp


# M4-14: 全局异常处理器（兜底降级，未处理异常返回 503 而非 500）
# 双重保障：exception_handler + middleware，确保 TestClient 与生产环境都能返回 503
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理异常，返回 503 降级响应（HTTPException 由 FastAPI 自身处理）"""
    trace_id = request.headers.get("X-Trace-Id", uuid.uuid4().hex[:8])
    logger.error(
        f"未处理异常降级: path={request.url.path}, trace_id={trace_id}, "
        f"error={exc}"
    )
    return _degraded_response(
        status_code=503,
        content={
            "detail": "服务暂时不可用（降级模式）",
            "trace_id": trace_id,
            "path": request.url.path,
            "error": str(exc)[:200],
            "degraded": True,
        },
    )


# M4-14: HTTP middleware 兜底（exception_handler 在某些 starlette 版本会被
# ServerErrorMiddleware 拦截，TestClient raise_server_exceptions=True 时直接 raise；
# middleware 在 ExceptionMiddleware 外层，能可靠捕获并返回 503）
@app.middleware("http")
async def global_exception_middleware(request: Request, call_next):
    """HTTP middleware 兜底降级，捕获 exception_handler 未拦截的未处理异常"""
    try:
        return await call_next(request)
    except Exception as exc:
        trace_id = request.headers.get("X-Trace-Id", uuid.uuid4().hex[:8])
        logger.error(
            f"middleware 降级: path={request.url.path}, trace_id={trace_id}, "
            f"error={exc}"
        )
        return _degraded_response(
            status_code=503,
            content={
                "detail": "服务暂时不可用（降级模式）",
                "trace_id": trace_id,
                "path": request.url.path,
                "error": str(exc)[:200],
                "degraded": True,
            },
        )


# M4-15: SLA 监控中间件（最外层，最后注册最先执行，记录每个请求的指标）
@app.middleware("http")
async def sla_monitoring_middleware(request: Request, call_next):
    """M4-15: SLA 监控中间件，记录每个请求的响应时间、状态码、降级标记"""
    start = time.time()
    status_code = 500
    degraded = False
    try:
        response = await call_next(request)
        status_code = response.status_code
        # 通过响应头 X-Degraded 判断软降级（200+degraded）
        if response.headers.get("X-Degraded") == "true":
            degraded = True
        # 503 硬降级
        if status_code == 503:
            degraded = True
        return response
    finally:
        duration_ms = (time.time() - start) * 1000
        sla_monitor.record(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
            degraded=degraded,
        )

# 全局 MemoryService
memory = MemoryService()

# M5-1: 全局 EffectTracker（复用 memory 的 db 连接）
from agent.effect_tracker import EffectTracker
effect_tracker = EffectTracker(memory)

# 存储已完成的归因结果（M0 用内存存储，M1 切换为 checkpointer）
_results_store: dict[str, dict] = {}

# M4-14: 反馈降级队列（SQLite 故障时暂存，服务恢复后可重试持久化）
_feedback_queue: list[dict] = []


# ===== M4-10: 多租户权限校验 =====


def _check_line_access(user_id: str, line_id: str, require_write: bool = False) -> None:
    """校验用户对产线的访问权限（M4-10）

    Args:
        user_id: 用户ID
        line_id: 产线ID
        require_write: 是否需要写权限（POST/PUT/DELETE 端点设 True）

    Raises:
        HTTPException(403): 无权限时抛出
    """
    if not can_access_line(user_id, line_id):
        raise HTTPException(
            status_code=403,
            detail=f"用户 {user_id} 无权访问产线 {line_id}",
        )
    if require_write:
        perms = get_user_permissions(user_id)
        if not perms["can_write"]:
            raise HTTPException(
                status_code=403,
                detail=f"用户 {user_id} 无写权限（角色: {perms['role']}）",
            )


def _check_delete_access(user_id: str, line_id: str) -> None:
    """校验删除权限（M4-10）"""
    _check_line_access(user_id, line_id, require_write=True)
    perms = get_user_permissions(user_id)
    if not perms["can_delete"]:
        raise HTTPException(
            status_code=403,
            detail=f"用户 {user_id} 无删除权限（角色: {perms['role']}）",
        )


# ===== 请求/响应模型 =====

class AnalyzeRequest(BaseModel):
    """归因请求"""
    query: str = "请分析此批次的缺陷原因"
    batch_id: str
    defect_type: Optional[str] = None
    measured_value: Optional[float] = None
    standard_value: Optional[float] = None
    line_id: str = "heat_treatment"  # M4-9: 产线ID，默认热处理
    user_id: str = "operator_01"     # M4-10: 提交用户


class AnalyzeResponse(BaseModel):
    """归因响应"""
    trace_id: str
    status: str
    message: str
    final_answer: Optional[str] = None
    proposals: Optional[list] = None


class FeedbackRequest(BaseModel):
    """反馈请求"""
    proposal_id: str
    user_id: str
    action: str  # adopted / rejected / partial
    score: float  # 0-1
    comment: Optional[str] = None
    line_id: str = "heat_treatment"  # M4-9: 产线ID


# ===== 端点 =====

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """提交归因请求（M0 同步执行，返回完整结果）

    流程：planner → data → mechanism → knowledge → decision → review → interaction → memory_writer
    """
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    logger.info(f"归因请求: trace_id={trace_id}, batch={req.batch_id}, query={req.query}, line={req.line_id}")

    # M4-10: 权限校验
    _check_line_access(req.user_id, req.line_id)

    # 构造初始状态
    initial_state: AgentState = {
        "user_query": req.query,
        "batch_id": req.batch_id,
        "defect_record": None,
        "plan": [],
        "current_step": 0,
        "observations": [],
        "data_result": None,
        "mechanism_result": None,
        "knowledge_result": None,
        "decision_result": None,
        "review_result": None,
        "proposal": None,
        "final_answer": None,
        "retry_count": 0,
        "needs_replan": False,
        "max_replan": 3,
        "trace_id": trace_id,
        "session_id": trace_id,
        "line_id": req.line_id,  # M4-9: 产线ID，贯穿全链路
    }

    try:
        orchestrator = build_orchestrator()
        config = {"configurable": {"thread_id": trace_id}}

        # 同步执行 Agent
        final_state = await orchestrator.ainvoke(initial_state, config)

        # 存储结果
        _results_store[trace_id] = final_state

        # 提取关键字段
        decision = final_state.get("decision_result") or {}
        proposals = decision.get("proposals", [])

        logger.info(
            f"归因完成: trace_id={trace_id}, "
            f"proposals={len(proposals)}, "
            f"source={decision.get('source', 'unknown')}"
        )

        return AnalyzeResponse(
            trace_id=trace_id,
            status="completed",
            message="归因完成",
            final_answer=final_state.get("final_answer", ""),
            proposals=proposals,
        )
    except Exception as e:
        logger.error(f"归因失败: trace_id={trace_id}, error={e}")
        raise HTTPException(status_code=500, detail=f"归因失败: {e}")


@app.get("/api/v1/result/{trace_id}")
async def get_result(trace_id: str):
    """查询归因结果"""
    if trace_id not in _results_store:
        raise HTTPException(status_code=404, detail=f"trace_id={trace_id} 不存在")

    state = _results_store[trace_id]
    decision = state.get("decision_result") or {}
    review = state.get("review_result") or {}

    return {
        "trace_id": trace_id,
        "status": "completed",
        "final_answer": state.get("final_answer", ""),
        "proposals": decision.get("proposals", []),
        "review": review,
        "observations_count": len(state.get("observations", [])),
        "retry_count": state.get("retry_count", 0),
    }


@app.post("/api/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    """提交用户反馈（持久化到 SQLite）"""
    feedback_id = f"fb_{uuid.uuid4().hex[:8]}"

    # M4-10: 权限校验（反馈需要写权限）
    _check_line_access(req.user_id, req.line_id, require_write=True)

    # 持久化到 SQLite（M4-14: 故障降级，暂存内存队列）
    try:
        saved = memory.write_feedback(
            feedback_id=feedback_id,
            proposal_id=req.proposal_id,
            user_id=req.user_id,
            action=req.action,
            score=req.score,
            comment=req.comment,
            line_id=req.line_id,  # M4-9: 产线隔离
        )
        if not saved:
            raise RuntimeError("write_feedback 返回 False")

        # 尝试更新案例置信度（Chroma 不可用时降级）
        if req.action == "adopted":
            memory.update_confidence(req.proposal_id, req.score)

        logger.info(f"反馈已保存: {feedback_id}, action={req.action}, score={req.score}, line={req.line_id}")
        return {"success": True, "feedback_id": feedback_id}
    except Exception as e:
        # M4-14: SQLite 故障降级，暂存内存队列，返回 200 + degraded
        logger.warning(f"feedback 写入降级，暂存队列: feedback_id={feedback_id}, error={e}")
        _feedback_queue.append({
            "feedback_id": feedback_id,
            "proposal_id": req.proposal_id,
            "user_id": req.user_id,
            "action": req.action,
            "score": req.score,
            "comment": req.comment,
            "line_id": req.line_id,
        })
        return _degraded_response({
            "success": True,
            "feedback_id": feedback_id,
            "degraded": True,
            "message": "反馈已暂存，将在服务恢复后持久化",
        })


@app.get("/api/v1/cases")
async def list_cases(
    defect_type: Optional[str] = None,
    line_id: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID（M4-10 权限过滤）"),
    limit: int = 20,
):
    """查询案例库（M4-9: 支持按产线过滤；M4-10: 权限隔离）"""
    # M4-10: 权限校验 + 自动过滤
    user_lines = get_user_lines(user_id)
    if line_id:
        _check_line_access(user_id, line_id)
        target_lines = [line_id]
    elif "*" in user_lines:
        target_lines = None  # admin 全部
    else:
        target_lines = user_lines  # 仅返回有权限的产线

    records = []
    try:
        if target_lines is None:
            records = memory.query_episodic(defect_type=defect_type, days=365)
        else:
            # M4-16: 单次 IN 查询替代 N+1 循环
            records = memory.query_episodic(
                defect_type=defect_type, days=365, line_id=target_lines
            )
    except Exception as e:
        # M4-14: SQLite 故障降级，返回空结果而非 500
        logger.warning(f"cases 查询降级: {e}")
        return _degraded_response({
            "total": 0, "cases": [], "degraded": True, "error": str(e)[:100]
        })
    return {
        "total": len(records),
        "cases": records[:limit],
    }


# ===== M3-12 记忆可视化端点 =====

@app.get("/api/v1/memory/stats")
async def memory_stats(
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """获取记忆统计概览（三层记忆总数 + 分布 + 最近记录）"""
    # M4-10: 非 admin 用户只统计有权限的产线
    stats = memory.get_memory_stats()
    user_lines = get_user_lines(user_id)
    if "*" not in user_lines:
        stats["filtered_by_user"] = user_lines
        stats["user_role"] = get_user_permissions(user_id)["role"]
    return stats


@app.get("/api/v1/memory/episodic")
async def list_episodic(
    limit: int = 100,
    line_id: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """列出全部短期记忆（SQLite episodic 表，M4-9: 产线过滤；M4-10: 权限隔离）"""
    user_lines = get_user_lines(user_id)
    if line_id:
        _check_line_access(user_id, line_id)
        records = memory.query_episodic(days=365, line_id=line_id)
    elif "*" in user_lines:
        records = memory.list_all_episodic(limit=limit)
    else:
        # M4-16: 单次 IN 查询替代 N+1 循环
        records = memory.query_episodic(days=365, line_id=user_lines)
    return {"total": len(records), "records": records}


@app.get("/api/v1/memory/semantic")
async def list_semantic(
    limit: int = 100,
    line_id: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """列出全部长期记忆（Chroma 案例库）"""
    user_lines = get_user_lines(user_id)
    if line_id:
        _check_line_access(user_id, line_id)
        # M4-16: Chroma where 服务端过滤
        records = memory.list_all_semantic(limit=limit, line_id=line_id)
    elif "*" in user_lines:
        records = memory.list_all_semantic(limit=limit)
    else:
        # M4-16: 非 admin 用 $in 服务端过滤，避免拉全量再 Python 过滤
        records = memory.list_all_semantic(limit=limit, line_id=user_lines)
    return {"total": len(records), "records": records}


@app.get("/api/v1/memory/feedback")
async def list_feedback(
    limit: int = 100,
    line_id: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """列出全部用户反馈（M4-9: 产线过滤；M4-10: 权限隔离）"""
    user_lines = get_user_lines(user_id)
    if line_id:
        _check_line_access(user_id, line_id)
        records = memory.query_feedback(days=365, line_id=line_id)
    elif "*" in user_lines:
        records = memory.list_all_feedback(limit=limit)
    else:
        # M4-16: 单次 IN 查询替代 N+1 循环
        records = memory.query_feedback(days=365, line_id=user_lines)
    return {"total": len(records), "records": records}


# ===== M3-9 知识冲突检测端点 =====

@app.get("/api/v1/memory/conflicts")
async def list_conflicts(
    limit: int = 100,
    line_id: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """列出全部知识冲突记录（按时间倒序，M4-9: 产线过滤；M4-10: 权限隔离）"""
    user_lines = get_user_lines(user_id)
    if line_id:
        _check_line_access(user_id, line_id)
        records = memory.list_conflicts(limit=limit, line_id=line_id)
    elif "*" in user_lines:
        records = memory.list_conflicts(limit=limit)
    else:
        # M4-16: 单次 IN 查询替代 N+1 循环
        records = memory.list_conflicts(limit=limit, line_id=user_lines)
    return {"total": len(records), "records": records}


# ===== M3-7 案例库 CRUD 端点 =====

class CaseCreateRequest(BaseModel):
    """新增案例请求"""
    case_id: Optional[str] = None  # 留空自动生成
    defect_type: str  # DefectType.value
    batch_id: str
    process_type: str = "heat_treatment"  # ProcessType.value
    temperature: Optional[float] = None
    holding_time: Optional[float] = None
    cooling_rate: Optional[float] = None
    root_cause: str
    solution: str = ""
    confidence: float = 0.5
    tags: list[str] = []
    line_id: str = "heat_treatment"  # M4-9: 产线ID
    user_id: str = "operator_01"     # M4-10: 提交用户


class CaseUpdateRequest(BaseModel):
    """更新案例请求（所有字段可选）"""
    root_cause: Optional[str] = None
    solution: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[list[str]] = None  # None=不更新，[]=清空
    user_id: str = "operator_01"      # M4-10: 操作用户


@app.post("/api/v1/memory/cases")
async def create_case(req: CaseCreateRequest):
    """新增长期记忆案例（M3-7 Create）"""
    # M4-10: 写权限校验
    _check_line_access(req.user_id, req.line_id, require_write=True)

    case_id = req.case_id or f"case-{uuid.uuid4().hex[:8]}"
    try:
        case = CaseRecord(
            case_id=case_id,
            defect_type=DefectType(req.defect_type),
            batch_params=BatchParams(
                batch_id=req.batch_id,
                process_type=ProcessType(req.process_type),
                temperature=req.temperature,
                holding_time=req.holding_time,
                cooling_rate=req.cooling_rate,
                start_time=datetime.now(),
            ),
            root_cause=req.root_cause,
            solution=req.solution,
            confidence=req.confidence,
            source="manual",
            tags=req.tags,
            line_id=req.line_id,  # M4-9: 产线隔离
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"参数非法: {e}")

    ok = memory.write_semantic(case)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="写入失败：Chroma 不可用或 ID 已存在",
        )

    # 检测冲突并返回告警
    conflicts = memory.detect_conflicts(case)
    return {
        "case_id": case.case_id,
        "created": True,
        "conflicts_detected": len(conflicts),
        "conflicts": [
            {
                "type": c.conflict_type,
                "existing_case_id": c.existing_case_id,
                "description": c.description,
            }
            for c in conflicts
        ],
    }


@app.get("/api/v1/memory/cases/{case_id}")
async def get_case(
    case_id: str,
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """获取单个案例（M3-7 Read）"""
    record = memory.get_semantic_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"案例 {case_id} 不存在")
    # M4-10: 校验该案例所属产线的访问权限
    case_line = (record.get("metadata") or {}).get("line_id", "heat_treatment")
    _check_line_access(user_id, case_line)
    return record


@app.put("/api/v1/memory/cases/{case_id}")
async def update_case(case_id: str, req: CaseUpdateRequest):
    """更新案例字段（M3-7 Update，部分更新）"""
    # M4-10: 写权限校验（需先查出案例所属产线）
    existing = memory.get_semantic_case(case_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"案例 {case_id} 不存在")
    case_line = (existing.get("metadata") or {}).get("line_id", "heat_treatment")
    _check_line_access(req.user_id, case_line, require_write=True)

    ok = memory.update_semantic(
        case_id=case_id,
        root_cause=req.root_cause,
        solution=req.solution,
        confidence=req.confidence,
        tags=req.tags,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"更新失败：案例 {case_id} 不存在或 Chroma 不可用",
        )
    return {"case_id": case_id, "updated": True}


@app.delete("/api/v1/memory/cases/{case_id}")
async def delete_case(
    case_id: str,
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """删除案例（M3-7 Delete，同时清理相关冲突记录）"""
    # M4-10: 删除权限校验（需先查出案例所属产线）
    existing = memory.get_semantic_case(case_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"案例 {case_id} 不存在")
    case_line = (existing.get("metadata") or {}).get("line_id", "heat_treatment")
    _check_delete_access(user_id, case_line)

    ok = memory.delete_semantic(case_id)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="删除失败：Chroma 不可用",
        )
    return {"case_id": case_id, "deleted": True}


# ===== M4-10: 多租户权限查询端点 =====


@app.get("/api/v1/auth/permissions")
async def get_permissions(
    user_id: str = Query(..., description="用户ID"),
):
    """查询用户权限（M4-10）

    返回角色、可访问产线列表、读写权限。
    前端用于动态渲染产线选择器、隐藏无权限操作。
    """
    perms = get_user_permissions(user_id)
    return {
        "user_id": user_id,
        "role": perms["role"],
        "allowed_lines": perms["allowed_lines"],
        "can_write": perms["can_write"],
        "can_delete": perms["can_delete"],
        "available_lines": list_available_lines(),
    }


@app.get("/api/v1/lines")
async def list_lines(
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
):
    """列出用户可访问的产线（M4-10）

    返回产线配置摘要，仅包含用户有权限访问的产线。
    """
    from agent.utils import load_line_config

    user_lines = get_user_lines(user_id)
    all_lines = list_available_lines()
    # admin 或 ["*"] 返回全部
    visible = all_lines if "*" in user_lines else [
        lid for lid in all_lines if lid in user_lines
    ]

    lines_summary = []
    for lid in visible:
        cfg = load_line_config(lid)
        lines_summary.append({
            "line_id": lid,
            "name": cfg.get("name", lid),
            "process_type": cfg.get("process_type", ""),
            "material": cfg.get("material", ""),
            "defect_types": cfg.get("defect_types", []),
        })
    return {"total": len(lines_summary), "lines": lines_summary}


# ===== M4-11: 跨产线统一看板 =====


@app.get("/api/v1/dashboard/overview")
async def dashboard_overview(
    user_id: str = Query("operator_01", description="用户ID（M4-10 权限过滤）"),
    days: int = Query(30, description="统计天数（近 N 天）"),
):
    """M4-11: 跨产线汇总看板

    返回各产线 KPI（案例数/缺陷分布/反馈/冲突/置信度/采纳率），
    非 admin 用户只返回有权限访问的产线数据。

    Returns:
        {
            "user_id": str,
            "days": int,
            "lines": [
                {
                    "line_id": str, "name": str,
                    "episodic_count": int, "feedback_count": int,
                    "conflict_count": int, "semantic_count": int,
                    "defect_distribution": {type: count},
                    "action_distribution": {action: count},
                    "adoption_rate": float, "avg_confidence": float,
                }
            ],
            "totals": {
                "total_episodic": int, "total_feedback": int,
                "total_conflicts": int, "total_semantic": int,
                "overall_adoption_rate": float, "overall_avg_confidence": float,
            }
        }
    """
    from agent.utils import load_line_config

    user_lines = get_user_lines(user_id)
    all_lines = list_available_lines()
    visible = all_lines if "*" in user_lines else [
        lid for lid in all_lines if lid in user_lines
    ]

    line_stats_list = []
    total_episodic = 0
    total_feedback = 0
    total_conflicts = 0
    total_semantic = 0
    total_adopted = 0
    confidence_sum = 0.0
    confidence_count = 0

    for lid in visible:
        cfg = load_line_config(lid)
        try:
            stats = memory.get_line_stats(lid, days=days)
        except Exception as e:
            # M4-14: SQLite/Chroma 故障降级，该产线返回零值统计
            logger.warning(f"dashboard 产线 {lid} 降级: {e}")
            stats = {
                "line_id": lid,
                "episodic_count": 0,
                "feedback_count": 0,
                "conflict_count": 0,
                "semantic_count": 0,
                "defect_distribution": {},
                "action_distribution": {},
                "adoption_rate": 0.0,
                "avg_confidence": 0.0,
                "degraded": True,
            }
        stats["name"] = cfg.get("name", lid)
        line_stats_list.append(stats)

        total_episodic += stats["episodic_count"]
        total_feedback += stats["feedback_count"]
        total_conflicts += stats["conflict_count"]
        total_semantic += stats["semantic_count"]
        total_adopted += stats["action_distribution"].get("adopted", 0)
        if stats["semantic_count"] > 0:
            confidence_sum += stats["avg_confidence"] * stats["semantic_count"]
            confidence_count += stats["semantic_count"]

    overall_adoption = (
        round(total_adopted / total_feedback, 3) if total_feedback > 0 else 0.0
    )
    overall_confidence = (
        round(confidence_sum / confidence_count, 3) if confidence_count > 0 else 0.0
    )

    # M4-14: 任一产线降级则顶层 degraded=True
    any_degraded = any(ls.get("degraded") for ls in line_stats_list)

    # M4-15: 降级时通过 X-Degraded header 标记，供 SLA 中间件统计
    result = {
        "user_id": user_id,
        "days": days,
        "lines": line_stats_list,
        "totals": {
            "total_episodic": total_episodic,
            "total_feedback": total_feedback,
            "total_conflicts": total_conflicts,
            "total_semantic": total_semantic,
            "overall_adoption_rate": overall_adoption,
            "overall_avg_confidence": overall_confidence,
        },
        "degraded": any_degraded,
    }
    if any_degraded:
        return _degraded_response(result)
    return result


# ===== M4-15: SLA 保障端点 =====

@app.get("/api/v1/sla/status")
async def sla_status(
    window_minutes: int = Query(60, description="统计窗口（分钟）"),
):
    """M4-15: 获取 SLA 状态（可用性、P95/P99 延迟、错误率、降级次数）

    SLA 目标：availability >= 99.5%

    Returns:
        {
            "total_requests": int,
            "availability": float,       # 0.0-1.0
            "error_rate": float,         # 0.0-1.0
            "p95_latency_ms": float,
            "p99_latency_ms": float,
            "avg_latency_ms": float,
            "degraded_count": int,
            "sla_target": 0.995,
            "sla_met": bool,
            "window_minutes": int,
        }
    """
    return sla_monitor.get_stats(window_minutes=window_minutes)


@app.get("/api/v1/sla/report")
async def sla_report(
    window_minutes: int = Query(60, description="统计窗口（分钟）"),
    user_id: str = Query("admin", description="用户ID（M4-10 权限校验）"),
):
    """M4-15: 生成完整 SLA 报告（含按端点细分）

    Returns:
        {
            "overall": {...},            # 全局 SLA 统计
            "by_endpoint": {path: {...}}, # 按端点细分
            "sla_target": 0.995,
            "sla_met": bool,
        }
    """
    # M4-10: 仅 admin 可查看完整 SLA 报告
    perms = get_user_permissions(user_id)
    if perms["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail=f"用户 {user_id} 无权查看 SLA 报告（需 admin 角色）",
        )

    overall = sla_monitor.get_stats(window_minutes=window_minutes)
    by_endpoint = sla_monitor.get_stats_by_endpoint(window_minutes=window_minutes)

    return {
        "overall": overall,
        "by_endpoint": by_endpoint,
        "sla_target": sla_monitor.SLA_TARGET,
        "sla_met": overall["sla_met"],
    }


# ===== M5-1 调参效果跟踪端点 =====

class EffectScheduleRequest(BaseModel):
    """M5-1: 创建效果跟踪请求"""
    proposal_id: str
    case_id: str
    batch_id_before: str
    line_id: str = "heat_treatment"
    metric_before: Optional[float] = None  # 调参前缺陷率（0-1）
    days_offset: int = 7  # T+N 天后跟踪
    note: Optional[str] = None
    user_id: str = "operator_01"


@app.post("/api/v1/effect/track")
async def schedule_effect_tracking(req: EffectScheduleRequest):
    """M5-1: 调度调参效果跟踪（Agent 产出建议后调用）"""
    _check_line_access(req.user_id, req.line_id, require_write=True)
    try:
        tracking_id = effect_tracker.schedule_tracking(
            proposal_id=req.proposal_id,
            case_id=req.case_id,
            batch_id_before=req.batch_id_before,
            line_id=req.line_id,
            metric_before=req.metric_before,
            days_offset=req.days_offset,
            note=req.note,
        )
        return {"success": True, "tracking_id": tracking_id, "status": "pending"}
    except Exception as e:
        logger.warning(f"effect/track 降级: {e}")
        return _degraded_response({
            "success": False, "degraded": True, "error": str(e)[:200]
        })


@app.get("/api/v1/effect")
async def list_effect_trackings(
    status: Optional[str] = None,
    line_id: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID（M4-10 权限过滤）"),
    days: int = 30,
    limit: int = 100,
):
    """M5-1: 列出效果跟踪记录（按产线权限过滤）"""
    user_lines = get_user_lines(user_id)
    if line_id:
        _check_line_access(user_id, line_id)
        target = line_id
    elif "*" in user_lines:
        target = None  # admin 全部
    else:
        target = user_lines  # M5-1: 多产线 IN 查询

    try:
        records = effect_tracker.list_trackings(
            line_id=target, status=status, days=days, limit=limit
        )
        return {"total": len(records), "records": records}
    except Exception as e:
        logger.warning(f"effect 列表降级: {e}")
        return _degraded_response({
            "total": 0, "records": [], "degraded": True, "error": str(e)[:100]
        })


@app.get("/api/v1/effect/stats")
async def effect_stats(
    line_id: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID（M4-10）"),
    days: int = 30,
):
    """M5-1: 效果跟踪统计概览"""
    user_lines = get_user_lines(user_id)
    if line_id:
        _check_line_access(user_id, line_id)
        target = line_id
    elif "*" in user_lines:
        target = None
    else:
        target = user_lines[0] if user_lines else None  # stats 单产线

    try:
        stats = effect_tracker.get_effect_stats(line_id=target, days=days)
        return stats
    except Exception as e:
        logger.warning(f"effect/stats 降级: {e}")
        return _degraded_response({
            "degraded": True, "error": str(e)[:100], "total": 0
        })


@app.get("/api/v1/effect/{tracking_id}")
async def get_effect_tracking(tracking_id: str):
    """M5-1: 查询单条效果跟踪记录"""
    rec = effect_tracker.get_tracking(tracking_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"跟踪记录 {tracking_id} 不存在")
    return rec


@app.post("/api/v1/effect/{tracking_id}/evaluate")
async def evaluate_effect(
    tracking_id: str,
    batch_id_after: Optional[str] = None,
    user_id: str = Query("operator_01", description="用户ID"),
):
    """M5-1: 触发效果跟踪评估（T+N 天后执行）

    查询调参后批次质量，对比调整前后指标，计算改善百分比。
    """
    rec = effect_tracker.get_tracking(tracking_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"跟踪记录 {tracking_id} 不存在")
    _check_line_access(user_id, rec["line_id"], require_write=True)

    try:
        result = effect_tracker.track_effect(
            tracking_id, batch_id_after=batch_id_after
        )
        if result is None:
            return _degraded_response({
                "success": False, "degraded": True, "error": "跟踪失败"
            })
        return {"success": True, **result}
    except Exception as e:
        logger.warning(f"effect/{tracking_id}/evaluate 降级: {e}")
        return _degraded_response({
            "success": False, "degraded": True, "error": str(e)[:200]
        })


@app.post("/api/v1/effect/{tracking_id}/attribute")
async def attribute_effect(
    tracking_id: str,
    user_id: str = Query("operator_01", description="用户ID"),
):
    """M5-2: 把跟踪效果归因到对应案例（更新 confidence）

    将真实生产效果（改善百分比）反馈到案例库 confidence，
    形成"效果→案例→下次检索"的闭环。
    幂等：已归因的记录直接返回上次结果。
    """
    rec = effect_tracker.get_tracking(tracking_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"跟踪记录 {tracking_id} 不存在")
    _check_line_access(user_id, rec["line_id"], require_write=True)

    try:
        result = effect_tracker.attribute_effect(tracking_id)
        if result is None:
            return _degraded_response({
                "success": False, "degraded": True,
                "error": "归因失败（可能未 tracked 或无 improvement_pct）"
            })
        return {"success": True, **result}
    except Exception as e:
        logger.warning(f"effect/{tracking_id}/attribute 降级: {e}")
        return _degraded_response({
            "success": False, "degraded": True, "error": str(e)[:200]
        })


@app.post("/api/v1/effect/run-due")
async def run_due_effect_trackings(
    user_id: str = Query("admin", description="仅 admin 可批量执行"),
    line_id: Optional[str] = None,
):
    """M5-1: 批量执行到期的待跟踪记录（定时任务调用）"""
    perms = get_user_permissions(user_id)
    if perms["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅 admin 可批量执行跟踪")
    try:
        result = effect_tracker.run_due_trackings(line_id=line_id)
        return {"success": True, **result}
    except Exception as e:
        logger.warning(f"effect/run-due 降级: {e}")
        return _degraded_response({
            "success": False, "degraded": True, "error": str(e)[:200]
        })


@app.websocket("/api/v1/stream")
async def stream(ws: WebSocket):
    """实时推送执行过程

    客户端连接后发送首条消息：{"query": "...", "batch_id": "B001"}
    服务端流式推送 LangGraph 各节点执行事件，最后发送 final 结果。
    """
    await ws.accept()
    try:
        await ws.send_json({"event": "connected", "message": "Stream connected"})

        # 等待客户端发送查询参数
        init_msg = await ws.receive_text()
        params = json.loads(init_msg)
        query = params.get("query", "请分析此批次的缺陷原因")
        batch_id = params.get("batch_id", "")
        max_replan = params.get("max_replan", 3)

        if not batch_id:
            await ws.send_json({"event": "error", "message": "缺少 batch_id"})
            return

        trace_id = f"ws_{uuid.uuid4().hex[:12]}"
        initial_state: AgentState = {
            "user_query": query,
            "batch_id": batch_id,
            "defect_record": None,
            "plan": [],
            "current_step": 0,
            "observations": [],
            "data_result": None,
            "mechanism_result": None,
            "knowledge_result": None,
            "decision_result": None,
            "review_result": None,
            "proposal": None,
            "final_answer": None,
            "retry_count": 0,
            "needs_replan": False,
            "max_replan": max_replan,
            "trace_id": trace_id,
            "session_id": trace_id,
        }

        await ws.send_json({"event": "start", "trace_id": trace_id, "batch_id": batch_id})

        # 构建全新 orchestrator 并流式执行
        orchestrator = build_orchestrator()
        config = {"configurable": {"thread_id": trace_id}}

        final_state = None
        try:
            async for event in orchestrator.astream_events(initial_state, config, version="v2"):
                evt_type = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # 只推送有意义的事件（节点开始/结束、工具调用）
                if evt_type in ("on_chain_start", "on_chain_end") and name:
                    # 节点执行事件
                    await ws.send_json({
                        "event": "node",
                        "node": name,
                        "status": "start" if evt_type == "on_chain_start" else "end",
                        "data": str(data)[:500],  # 截断避免消息过大
                    })
                elif evt_type == "on_tool_start":
                    await ws.send_json({
                        "event": "tool",
                        "tool": name,
                        "status": "start",
                    })
                elif evt_type == "on_tool_end":
                    await ws.send_json({
                        "event": "tool",
                        "tool": name,
                        "status": "end",
                        "output": str(data)[:500],
                    })

            # 获取最终状态
            final_state = await orchestrator.aget_state(config)
            final_values = final_state.values if final_state else {}

            decision = final_values.get("decision_result") or {}
            proposals = decision.get("proposals", [])

            await ws.send_json({
                "event": "final",
                "trace_id": trace_id,
                "final_answer": final_values.get("final_answer", ""),
                "proposals": proposals,
                "review": final_values.get("review_result", {}),
                "retry_count": final_values.get("retry_count", 0),
                "observations_count": len(final_values.get("observations", [])),
            })

            # 存储结果
            _results_store[trace_id] = final_values

        except Exception as e:
            logger.error(f"stream 执行失败: trace_id={trace_id}, error={e}")
            await ws.send_json({"event": "error", "message": f"执行失败: {e}"})

        await ws.send_json({"event": "done", "trace_id": trace_id})

    except WebSocketDisconnect:
        logger.info("WebSocket 已断开")
    except json.JSONDecodeError:
        await ws.send_json({"event": "error", "message": "首条消息必须是 JSON: {query, batch_id}"})
    except Exception as e:
        logger.error(f"WebSocket 异常: {e}")
        try:
            await ws.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass


# ===== M2-12 调试工具 =====

@app.get("/api/v1/debug/flows")
async def list_flows():
    """列出可用协作流程"""
    from agent.flow_config import list_flows as _list_flows, load_flow_config
    flows = _list_flows()
    result = []
    for name in flows:
        config = load_flow_config(name)
        result.append({
            "name": name,
            "description": config.description,
            "mode": config.mode,
            "parallel_agents": config.parallel_agents,
            "enable_arbitrate": config.enable_arbitrate,
        })
    return {"flows": result}


class DebugRunRequest(BaseModel):
    """调试执行请求"""
    query: str = "请分析此批次的缺陷原因"
    batch_id: str
    flow_name: str = "parallel"
    max_replan: int = 3


@app.post("/api/v1/debug/run")
async def debug_run(req: DebugRunRequest):
    """用指定流程执行，返回每步中间状态（M2-12 调试工具）

    与 /analyze 不同，此端点：
    - 支持指定 flow_name
    - 返回完整中间状态（不只是最终结果）
    - 用于调试和对比不同流程
    """
    from agent.orchestrator import build_orchestrator

    trace_id = f"debug_{uuid.uuid4().hex[:12]}"
    initial_state: AgentState = {
        "user_query": req.query,
        "batch_id": req.batch_id,
        "defect_record": None,
        "plan": [],
        "current_step": 0,
        "observations": [],
        "data_result": None,
        "mechanism_result": None,
        "knowledge_result": None,
        "arbitration_result": None,
        "decision_result": None,
        "review_result": None,
        "proposal": None,
        "final_answer": None,
        "retry_count": 0,
        "needs_replan": False,
        "max_replan": req.max_replan,
        "trace_id": trace_id,
        "session_id": trace_id,
    }

    orchestrator = build_orchestrator(req.flow_name)
    config = {"configurable": {"thread_id": trace_id}}

    try:
        final_state = await orchestrator.ainvoke(initial_state, config)

        return {
            "trace_id": trace_id,
            "flow_name": req.flow_name,
            "final_answer": final_state.get("final_answer", ""),
            "plan": final_state.get("plan", []),
            "observations": final_state.get("observations", []),
            "data_result": final_state.get("data_result"),
            "mechanism_result": final_state.get("mechanism_result"),
            "knowledge_result": final_state.get("knowledge_result"),
            "arbitration_result": final_state.get("arbitration_result"),
            "decision_result": final_state.get("decision_result"),
            "review_result": final_state.get("review_result"),
            "retry_count": final_state.get("retry_count", 0),
        }
    except Exception as e:
        logger.error(f"debug_run 失败: trace_id={trace_id}, error={e}")
        raise HTTPException(status_code=500, detail=f"执行失败: {e}")


@app.get("/api/v1/debug/trace/{trace_id}")
async def get_trace(trace_id: str):
    """查看某个会话的执行轨迹（M2-12 回放）

    从 _results_store 中获取已完成的归因结果，用于回放调试。
    """
    if trace_id not in _results_store:
        raise HTTPException(status_code=404, detail=f"trace_id {trace_id} 不存在")

    state = _results_store[trace_id]
    return {
        "trace_id": trace_id,
        "observations": state.get("observations", []),
        "data_result": state.get("data_result"),
        "mechanism_result": state.get("mechanism_result"),
        "knowledge_result": state.get("knowledge_result"),
        "arbitration_result": state.get("arbitration_result"),
        "decision_result": state.get("decision_result"),
        "review_result": state.get("review_result"),
        "final_answer": state.get("final_answer", ""),
        "retry_count": state.get("retry_count", 0),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
