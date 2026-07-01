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
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from agent.memory.memory_service import MemoryService
from agent.orchestrator import build_orchestrator
from agent.utils import setup_tracing
from models.entities import UserFeedback
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

# 全局 MemoryService
memory = MemoryService()

# 存储已完成的归因结果（M0 用内存存储，M1 切换为 checkpointer）
_results_store: dict[str, dict] = {}


# ===== 请求/响应模型 =====

class AnalyzeRequest(BaseModel):
    """归因请求"""
    query: str = "请分析此批次的缺陷原因"
    batch_id: str
    defect_type: Optional[str] = None
    measured_value: Optional[float] = None
    standard_value: Optional[float] = None


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
    logger.info(f"归因请求: trace_id={trace_id}, batch={req.batch_id}, query={req.query}")

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

    # 持久化到 SQLite
    saved = memory.write_feedback(
        feedback_id=feedback_id,
        proposal_id=req.proposal_id,
        user_id=req.user_id,
        action=req.action,
        score=req.score,
        comment=req.comment,
    )

    if not saved:
        raise HTTPException(status_code=500, detail="反馈持久化失败")

    # 尝试更新案例置信度（Chroma 不可用时降级）
    if req.action == "adopted":
        memory.update_confidence(req.proposal_id, req.score)

    logger.info(f"反馈已保存: {feedback_id}, action={req.action}, score={req.score}")
    return {"success": True, "feedback_id": feedback_id}


@app.get("/api/v1/cases")
async def list_cases(
    defect_type: Optional[str] = None,
    limit: int = 20,
):
    """查询案例库"""
    records = memory.query_episodic(
        defect_type=defect_type,
        days=365,
    )
    return {
        "total": len(records),
        "cases": records[:limit],
    }


# ===== M3-12 记忆可视化端点 =====

@app.get("/api/v1/memory/stats")
async def memory_stats():
    """获取记忆统计概览（三层记忆总数 + 分布 + 最近记录）"""
    return memory.get_memory_stats()


@app.get("/api/v1/memory/episodic")
async def list_episodic(limit: int = 100):
    """列出全部短期记忆（SQLite episodic 表）"""
    records = memory.list_all_episodic(limit=limit)
    return {"total": len(records), "records": records}


@app.get("/api/v1/memory/semantic")
async def list_semantic(limit: int = 100):
    """列出全部长期记忆（Chroma 案例库）"""
    records = memory.list_all_semantic(limit=limit)
    return {"total": len(records), "records": records}


@app.get("/api/v1/memory/feedback")
async def list_feedback(limit: int = 100):
    """列出全部用户反馈"""
    records = memory.list_all_feedback(limit=limit)
    return {"total": len(records), "records": records}


# ===== M3-9 知识冲突检测端点 =====

@app.get("/api/v1/memory/conflicts")
async def list_conflicts(limit: int = 100):
    """列出全部知识冲突记录（按时间倒序）"""
    records = memory.list_conflicts(limit=limit)
    return {"total": len(records), "records": records}


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
