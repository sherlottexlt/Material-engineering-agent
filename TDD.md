# MetaCraft Agent 技术设计文档（TDD）

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 创建日期 | 2026-06-29 |
| 对应 PRD | PRD.md v1.0 |
| 技术栈 | Python 3.11 + LangGraph + MCP + Chroma |
| 文档性质 | 技术设计文档 |

---

## 1. 系统概述

### 1.1 设计目标
将 PRD 中定义的 8 大功能模块（F1-F8）落地为可运行系统，重点解决：
1. Agent 自主行动与任务拆解的工程化实现
2. 多 Agent 协作流程的可控编排
3. 三层记忆架构的工程落地
4. 工具调用的标准化（MCP）
5. 全链路可观测与评估

### 1.2 设计原则
| 原则 | 含义 |
|------|------|
| **状态机优先** | 用 LangGraph 显式建模流程，避免隐式 ReAct 黑盒 |
| **工具标准化** | 所有外部能力封装为 MCP Server，工具可复用可替换 |
| **记忆分层隔离** | 工作/短期/长期记忆物理隔离，避免污染 |
| **人在回路** | 关键决策点必须有人工确认，Agent 不直接改产线 |
| **可观测优先** | 每一步都打 trace，可回溯、可复盘 |
| **降级容错** | 单工具/单 Agent 失败不阻断整体流程 |

---

## 2. 系统架构

### 2.1 分层架构

```
┌──────────────────────────────────────────────────────────┐
│  L5 接入层    │ Streamlit/React UI + FastAPI Gateway     │
├──────────────────────────────────────────────────────────┤
│  L4 编排层    │ LangGraph Orchestrator                   │
│               │  - Planner / Executor / Reflector        │
│               │  - Multi-Agent Coordinator              │
├──────────────────────────────────────────────────────────┤
│  L3 能力层    │ 6 个 Sub-Agent (角色)                    │
│               │  Data / Mechanism / Knowledge /          │
│               │  Decision / Review / Interaction         │
├──────────────────────────────────────────────────────────┤
│  L2 服务层    │ MCP Tools + RAG + Memory Service         │
├──────────────────────────────────────────────────────────┤
│  L1 数据层    │ Chroma + SQLite/PG + MES/SCADA           │
└──────────────────────────────────────────────────────────┘
```

### 2.2 组件清单

| 组件 | 职责 | 技术 |
|------|------|------|
| Gateway | API 网关、鉴权、限流 | FastAPI |
| Orchestrator | 主控编排、状态机 | LangGraph |
| Sub-Agents | 6 个角色 Agent | LangGraph subgraph |
| Tool Router | MCP 工具路由 | MCP Python SDK |
| RAG Service | 知识检索 | Chroma + LangChain |
| Memory Service | 三层记忆读写 | 自研 |
| Evaluator | 评估与反馈 | LangSmith + 自研 |
| Trace Collector | 全链路追踪 | Langfuse |

---

## 3. 核心数据模型

### 3.1 核心实体

```python
# models/entities.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal
from enum import Enum

class DefectType(str, Enum):
    HARDNESS_LOW = "hardness_low"
    HARDNESS_HIGH = "hardness_high"
    DEFORMATION = "deformation"
    CRACK = "crack"
    OTHER = "other"

class BatchParams(BaseModel):
    """批次工艺参数"""
    batch_id: str
    process_type: str  # 热处理/焊接/轧制
    temperature: Optional[float] = None  # ℃
    holding_time: Optional[float] = None  # 分钟
    cooling_rate: Optional[float] = None  # ℃/s
    pressure: Optional[float] = None  # MPa
    raw_material_batch: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None

class DefectRecord(BaseModel):
    """缺陷记录"""
    record_id: str
    batch_id: str
    defect_type: DefectType
    severity: float  # 0-1
    measured_value: Optional[float] = None
    standard_value: Optional[float] = None
    detected_at: datetime

class CaseRecord(base_model):
    """历史案例（长期记忆）"""
    case_id: str
    defect_type: DefectType
    batch_params: BatchParams
    root_cause: str
    solution: str
    effect: Optional[str] = None  # 调整后效果
    confidence: float = 0.5
    created_at: datetime
    source: Literal["manual", "auto"] = "auto"

class AdjustmentProposal(base_model):
    """参数调整建议"""
    proposal_id: str
    batch_id: str
    adjustments: dict[str, float]  # {参数名: 调整量}
    expected_effect: str
    risks: list[str]
    evidence: list[str]  # 证据链
    confidence: float
    created_at: datetime
```

### 3.2 Agent 状态模型（LangGraph State）

```python
# models/state.py
from typing import TypedDict, Annotated, Optional
import operator

class AgentState(TypedDict):
    # 输入
    user_query: str
    batch_id: Optional[str]
    defect_record: Optional[DefectRecord]

    # 任务规划
    plan: list[dict]  # [{"step_id": 1, "action": "...", "tool": "...", "status": "pending"}]
    current_step: int

    # 中间结果（累积式）
    observations: Annotated[list[dict], operator.add]
    # [{"step_id": 1, "tool": "...", "result": ..., "timestamp": ...}]

    # 各 Sub-Agent 输出
    data_result: Optional[dict]
    mechanism_result: Optional[dict]
    knowledge_result: Optional[dict]
    decision_result: Optional[dict]
    review_result: Optional[dict]

    # 最终输出
    proposal: Optional[AdjustmentProposal]
    final_answer: Optional[str]

    # 控制
    retry_count: int
    needs_replan: bool
    trace_id: str
```

---

## 4. Agent 七大要素技术实现

### 4.1 自主行动（LangGraph 状态机）

```python
# agent/orchestrator.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from models.state import AgentState
from agent.nodes import (
    planner, executor, reflector,
    data_agent, mechanism_agent, knowledge_agent,
    decision_agent, review_agent, interaction_agent,
    memory_writer
)

def build_orchestrator():
    g = StateGraph(AgentState)

    # 节点注册
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("reflector", reflector)
    g.add_node("data", data_agent)
    g.add_node("mechanism", mechanism_agent)
    g.add_node("knowledge", knowledge_agent)
    g.add_node("decision", decision_agent)
    g.add_node("review", review_agent)
    g.add_node("interaction", interaction_agent)
    g.add_node("memory_writer", memory_writer)

    # 边定义
    g.set_entry_point("planner")
    g.add_edge("planner", "executor")

    # Executor 路由：根据当前 step 分发到对应 Sub-Agent
    g.add_conditional_edges(
        "executor",
        route_step_to_agent,
        {
            "data": "data",
            "mechanism": "mechanism",
            "knowledge": "knowledge",
            "decision": "decision",
            "end": "reflector",
        }
    )

    # 三个并行 Agent 完成后进入 decision
    g.add_edge("data", "decision")
    g.add_edge("mechanism", "decision")
    g.add_edge("knowledge", "decision")

    # Decision → Review
    g.add_edge("decision", "review")

    # Review 路由：通过则进入交互，不通过则回退
    g.add_conditional_edges(
        "review",
        lambda s: "interaction" if s["review_result"]["approved"] else "decision",
        {"interaction": "interaction", "decision": "decision"}
    )

    # 交互后写记忆
    g.add_edge("interaction", "memory_writer")
    g.add_edge("memory_writer", END)

    # Reflector 路由：判断是否重新规划
    g.add_conditional_edges(
        "reflector",
        lambda s: "planner" if s["needs_replan"] and s["retry_count"] < 3 else END,
        {"planner": "planner", END: END}
    )

    return g.compile(checkpointer=MemorySaver())

def route_step_to_agent(state: AgentState) -> str:
    """根据当前子任务类型路由到对应 Agent"""
    if state["current_step"] >= len(state["plan"]):
        return "end"
    current = state["plan"][state["current_step"]]
    return current.get("agent", "end")
```

### 4.2 任务拆解（Planner 节点）

```python
# agent/nodes/planner.py
import json
from langchain_openai import ChatOpenAI
from models.state import AgentState

PLANNER_PROMPT = """你是材料工艺任务规划专家。
将用户问题拆解为 3-5 个有序子任务，每个子任务必须可独立执行、可验证。

可用 Agent：
- data: 查询批次工艺参数、历史缺陷
- mechanism: 调用机理模型验证假设
- knowledge: 检索工艺手册和历史案例
- decision: 综合多 Agent 结果生成建议

输出 JSON：
{{
  "plan": [
    {{"step_id": 1, "agent": "data", "action": "查询批次 {batch_id} 工艺参数", "tool": "query_batch_params"}},
    ...
  ]
}}

用户问题：{query}
批次ID：{batch_id}
"""

def planner(state: AgentState) -> dict:
    llm = ChatOpenAI(model="qwen-max", temperature=0)
    prompt = PLANNER_PROMPT.format(
        query=state["user_query"],
        batch_id=state.get("batch_id", "未提供")
    )
    resp = llm.invoke(prompt)
    plan = json.loads(resp.content)["plan"]
    return {"plan": plan, "current_step": 0, "retry_count": state.get("retry_count", 0)}
```

### 4.3 工具调用（MCP 标准化）

#### 4.3.1 MCP Server 定义

```python
# mcp_servers/mes_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio

server = Server("mes-tools")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_batch_params",
            description="根据批次ID查询工艺参数",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "批次编号"}
                },
                "required": ["batch_id"]
            }
        ),
        Tool(
            name="query_defect_history",
            description="查询历史缺陷记录",
            inputSchema={
                "type": "object",
                "properties": {
                    "defect_type": {"type": "string"},
                    "days_back": {"type": "integer", "default": 30}
                }
            }
        ),
        Tool(
            name="submit_adjustment",
            description="提交参数调整建议到审核流程",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string"},
                    "adjustments": {"type": "object"}
                },
                "required": ["batch_id", "adjustments"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_batch_params":
        # 实际接入 MES / SCADA
        params = await fetch_batch_from_mes(arguments["batch_id"])
        return [TextContent(type="text", text=params.json())]
    elif name == "query_defect_history":
        records = await query_defects(**arguments)
        return [TextContent(type="text", text=records)]
    # ...

if __name__ == "__main__":
    from mcp.server.stdio import stdio_server
    asyncio.run(stdio_server(server))
```

#### 4.3.2 机理模型工具

```python
# mcp_servers/metallurgy_server.py
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "run_metallurgy_model":
        # 调用材料机理模型（如 JMAK 方程预测硬度）
        predicted_hardness = jmak_model(
            temperature=arguments["temperature"],
            holding_time=arguments["holding_time"],
            cooling_rate=arguments["cooling_rate"]
        )
        return [TextContent(type="text", text=f"{{\"predicted_hardness\": {predicted_hardness}}}")]

def jmak_model(temperature, holding_time, cooling_rate):
    """JMAK 方程简化实现：预测转变分数与硬度"""
    # 实际工程中接你的材料模型
    k = 0.01  # 速率常数
    n = 1.5   # Avrami 指数
    fraction = 1 - math.exp(-k * (holding_time ** n) * (temperature / 850))
    predicted = 60 * fraction - 0.5 * cooling_rate
    return round(predicted, 2)
```

### 4.4 记忆管理（三层架构）

```python
# agent/memory/memory_service.py
from datetime import datetime, timedelta
from typing import Optional
import sqlite3
from chromadb import Client
from chromadb.config import Settings

class MemoryService:
    """三层记忆管理：工作 / 短期 / 长期"""

    def __init__(self):
        # 长期记忆：向量库
        self.chroma = Client(Settings(anonymized_telemetry=False))
        self.collection = self.chroma.get_or_create_collection("metacraft_cases")

        # 短期记忆：SQLite
        self.db = sqlite3.connect("memory/episodic.db")
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS episodic (
                record_id TEXT PRIMARY KEY,
                batch_id TEXT,
                defect_type TEXT,
                root_cause TEXT,
                solution TEXT,
                created_at TIMESTAMP,
                quality_score REAL DEFAULT 0.5
            )
        """)
        self.db.commit()

    # === 工作记忆：由 LangGraph State 管理，这里不重复 ===

    # === 短期记忆 ===
    def write_episodic(self, batch_id, defect_type, root_cause, solution):
        self.db.execute(
            "INSERT INTO episodic VALUES (?, ?, ?, ?, ?, ?, 0.5)",
            (f"ep_{int(datetime.now().timestamp())}",
             batch_id, defect_type, root_cause, solution,
             datetime.now())
        )
        self.db.commit()

    def query_episodic(self, batch_id=None, days=30):
        since = datetime.now() - timedelta(days=days)
        cur = self.db.execute(
            "SELECT * FROM episodic WHERE created_at >= ? AND (? IS NULL OR batch_id = ?)",
            (since, batch_id, batch_id)
        )
        return cur.fetchall()

    # === 长期记忆 ===
    def write_semantic(self, case: CaseRecord):
        self.collection.add(
            ids=[case.case_id],
            documents=[f"{case.defect_type}\n{case.root_cause}\n{case.solution}"],
            metadatas=[{
                "defect_type": case.defect_type,
                "confidence": case.confidence,
                "created_at": case.created_at.isoformat()
            }]
        )

    def search_semantic(self, query: str, top_k: int = 3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results

    # === 记忆权重更新（基于用户反馈）===
    def update_confidence(self, case_id: str, feedback_score: float):
        """根据用户反馈调整案例置信度"""
        self.collection.update(
            ids=[case_id],
            metadatas=[{"confidence": feedback_score}]
        )
```

### 4.5 角色设定（6 个 Sub-Agent Prompt）

```python
# agent/prompts/roles.py

DATA_AGENT_PROMPT = """你是【数据 Agent】，性格严谨、只陈述事实。
职责：查询批次工艺参数和历史缺陷，不做推断。
规则：
1. 只输出数据，不输出观点
2. 数据缺失时明确说明"未查询到"
3. 单位必须标注（℃、MPa、分钟、HRc）
"""

MECHANISM_AGENT_PROMPT = """你是【机理 Agent】，基于物理冶金原理分析。
职责：调用机理模型，验证假设是否成立。
规则：
1. 所有结论必须基于机理模型输出或物理定律
2. 明确区分"模型预测"与"经验推测"
3. 给出可证伪的假设
背景知识：JMAK 方程、相变动力学、Hall-Petch 关系等
"""

KNOWLEDGE_AGENT_PROMPT = """你是【知识 Agent】，类似图书管理员。
职责：检索工艺手册和历史案例，引用来源。
规则：
1. 必须标注来源（手册名称/案例ID）
2. 不臆造知识
3. 检索结果按相关性排序
"""

DECISION_AGENT_PROMPT = """你是【决策 Agent】，经验型老师傅角色。
职责：综合数据、机理、知识三方信息，给出排序建议。
规则：
1. 输出至少 2 个候选方案
2. 每个方案标注：调整项、调整量、预期效果、风险、依据
3. 方案排序按"可行性 × 置信度"
4. 不输出违反工艺约束的方案
"""

REVIEW_AGENT_PROMPT = """你是【审核 Agent】，质监员角色，挑刺、保守。
职责：审核决策 Agent 输出的合理性。
规则：
1. 检查证据链是否完整
2. 检查是否违反工艺约束
3. 检查置信度是否合理
4. 不通过必须给出具体原因
5. 默认倾向保守
"""

INTERACTION_AGENT_PROMPT = """你是【交互 Agent】，面向操作员沟通。
职责：把技术结论翻译成操作员易懂的语言。
规则：
1. 先结论，后展开证据（渐进披露）
2. 标注置信度（高/中/低）
3. 提供"一键确认/拒绝"选项
4. 不确定时主动说"我不确定，建议人工复核"
5. 语气友好但不啰嗦
"""
```

### 4.6 协作流程（Multi-Agent Coordinator）

```python
# agent/nodes/coordinator.py
from langgraph.graph import StateGraph, END
from models.state import AgentState

def build_collaboration_subgraph():
    """多 Agent 协作子图"""
    sg = StateGraph(AgentState)

    sg.add_node("data", data_agent)
    sg.add_node("mechanism", mechanism_agent)
    sg.add_node("knowledge", knowledge_agent)
    sg.add_node("decision", decision_agent)
    sg.add_node("review", review_agent)

    # 入口分发到三个并行 Agent
    sg.set_entry_point("data")
    # 使用 fan-out 实现（LangGraph 支持）
    # 三个完成后汇聚到 decision

    sg.add_edge("data", "decision")
    sg.add_edge("mechanism", "decision")
    sg.add_edge("knowledge", "decision")

    sg.add_edge("decision", "review")

    # Review 不通过回退
    sg.add_conditional_edges(
        "review",
        lambda s: "decision" if not s["review_result"]["approved"] else END,
        {"decision": "decision", END: END}
    )

    return sg.compile()
```

### 4.7 结果评估

```python
# agent/evaluator.py
from langchain_openai import ChatOpenAI
from langsmith import Client

EVALUATOR_PROMPT = """你是独立的评估 Agent，审核以下归因结果：
归因结论：{conclusion}
证据链：{evidence}
请评估：
1. 证据是否充分（1-5 分）
2. 推理是否合理（1-5 分）
3. 是否违反工艺常识（是/否）
4. 总体可信度（0-1）
输出 JSON。
"""

class Evaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.langsmith = Client()

    async def evaluate(self, state: AgentState) -> dict:
        # 1. LLM 自评
        prompt = EVALUATOR_PROMPT.format(
            conclusion=state.get("final_answer", ""),
            evidence=state.get("observations", [])
        )
        result = self.llm.invoke(prompt)
        score = parse_json(result.content)

        # 2. 记录到 LangSmith
        self.langsmith.create_example(
            inputs={"query": state["user_query"]},
            outputs={"answer": state["final_answer"]},
            dataset_name="metacraft_eval"
        )

        return {"eval_score": score, "review_result": score}

    def collect_feedback(self, proposal_id: str, feedback: str, score: float):
        """收集用户反馈，更新记忆权重"""
        # 写入反馈表
        # 更新对应案例的 confidence
        pass
```

---

## 5. 接口设计

### 5.1 REST API

| 端点 | 方法 | 描述 | 请求体 | 响应 |
|------|------|------|--------|------|
| `/api/v1/analyze` | POST | 提交归因请求 | `{batch_id, defect_type, query}` | `{trace_id, status}` |
| `/api/v1/result/{trace_id}` | GET | 查询归因结果 | - | `{proposal, trace}` |
| `/api/v1/feedback` | POST | 提交用户反馈 | `{proposal_id, score, comment}` | `{success}` |
| `/api/v1/cases` | GET | 查询案例库 | `?defect_type=&limit=` | `{cases[]}` |
| `/api/v1/stream` | WS | 实时推送执行过程 | - | `{step, status, result}` |

### 5.2 关键接口示例

```python
# api/routes.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from agent.orchestrator import build_orchestrator

app = FastAPI()
orchestrator = build_orchestrator()

@app.post("/api/v1/analyze")
async def analyze(req: AnalyzeRequest):
    state = {
        "user_query": req.query,
        "batch_id": req.batch_id,
        "defect_record": req.defect_record,
    }
    # 异步执行
    config = {"configurable": {"thread_id": req.trace_id}}
    result = await orchestrator.ainvoke(state, config)
    return {"trace_id": req.trace_id, "proposal": result.get("proposal")}

@app.websocket("/api/v1/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    # 流式推送 LangGraph 各节点执行状态
    async for event in orchestrator.astream_events(state, version="v1"):
        await ws.send_json({
            "step": event["name"],
            "status": event["event"],
            "data": event.get("data")
        })
```

---

## 6. MCP 工具清单

| 工具名 | 所属 MCP Server | 功能 | 数据源 |
|--------|-----------------|------|--------|
| query_batch_params | mes_server | 查批次工艺参数 | MES |
| query_defect_history | mes_server | 查历史缺陷 | 缺陷库 |
| submit_adjustment | mes_server | 提交参数调整 | MES API |
| run_metallurgy_model | metallurgy_server | 调机理模型 | Python 模型 |
| search_handbook | knowledge_server | 检索工艺手册 | Chroma |
| search_cases | knowledge_server | 检索历史案例 | Chroma |
| write_case | knowledge_server | 写入新案例 | Chroma |
| update_case_confidence | knowledge_server | 更新案例置信度 | Chroma |

---

## 7. 项目结构

```
metacraft-agent/
├── agent/
│   ├── orchestrator.py          # 主编排器
│   ├── nodes/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── reflector.py
│   │   ├── data_agent.py
│   │   ├── mechanism_agent.py
│   │   ├── knowledge_agent.py
│   │   ├── decision_agent.py
│   │   ├── review_agent.py
│   │   └── interaction_agent.py
│   ├── prompts/
│   │   └── roles.py
│   ├── memory/
│   │   └── memory_service.py
│   └── evaluator.py
├── mcp_servers/
│   ├── mes_server.py
│   ├── metallurgy_server.py
│   └── knowledge_server.py
├── models/
│   ├── entities.py
│   └── state.py
├── api/
│   ├── routes.py
│   └── auth.py
├── ui/
│   └── streamlit_app.py
├── data/
│   ├── handbooks/               # 工艺手册原文
│   └── seed_cases/              # 初始案例
├── eval/
│   ├── metrics.py
│   └── test_cases.json
├── tests/
│   ├── test_orchestrator.py
│   └── test_memory.py
├── config/
│   ├── settings.yaml
│   └── prompts.yaml
├── scripts/
│   ├── init_db.py
│   └── ingest_handbooks.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 8. 部署架构

### 8.1 部署模式

| 模式 | 适用场景 | LLM | 数据 |
|------|---------|-----|------|
| 云端 | MVP / 演示 | API 调用 | 脱敏后上云 |
| 混合 | V1+ | 本地小模型 + API 兜底 | 数据本地 |
| 全本地 | 严格合规 | 本地部署（Qwen-72B 等） | 全本地 |

### 8.2 Docker Compose

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [chroma, postgres]

  chroma:
    image: chromadb/chroma:latest
    ports: ["8001:8000"]
    volumes: ["./data/chroma:/chroma/chroma"]

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: metacraft
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes: ["./data/pg:/var/lib/postgresql/data"]

  langfuse:
    image: langfuse/langfuse:latest
    ports: ["3000:3000"]
    env_file: .env

  mcp_mes:
    build: ./mcp_servers
    command: python mes_server.py
```

---

## 9. 可观测与监控

### 9.1 三层监控

| 层级 | 工具 | 监控内容 |
|------|------|---------|
| 业务层 | 自建看板 | 归因准确率、采纳率、缺陷率 |
| Agent 层 | LangSmith + Langfuse | trace、token 用量、延迟、节点耗时 |
| 系统层 | Prometheus + Grafana | CPU/内存/磁盘、API QPS、错误率 |

### 9.2 关键指标

```python
# eval/metrics.py
class Metrics:
    # 任务级
    attribution_accuracy: float    # 归因准确率（vs 人工标注）
    adoption_rate: float           # 建议采纳率
    avg_response_time: float       # 平均响应时间

    # Agent 级
    token_per_task: int            # 单任务 token 消耗
    tool_call_success_rate: float  # 工具调用成功率
    replan_rate: float             # 重新规划率

    # 业务级
    defect_rate_change: float      # 缺陷率变化
    operator_trust_score: float    # 操作员信任度（问卷）
```

---

## 10. 安全设计

| 维度 | 措施 |
|------|------|
| 数据安全 | 工艺数据本地存储，传输加密（TLS） |
| 接口鉴权 | JWT + RBAC（操作员/工程师/管理员） |
| LLM 安全 | Prompt 注入防护、输出过滤、敏感词检测 |
| 操作审计 | 所有参数调整提交记录可追溯 |
| 权限隔离 | Agent 只读产线数据，写操作必须人工确认 |
| 容灾 | 数据库每日备份，Agent 状态可回放 |

---

## 11. 开发规范

### 11.1 代码规范
- Python 3.11+，类型注解强制
- 用 `ruff` 做 lint，`black` 做格式化
- 所有 LLM 调用必须走统一封装（便于切换模型）
- 所有 prompt 集中管理在 `config/prompts.yaml`

### 11.2 测试规范
- 单元测试覆盖率 ≥ 70%
- 关键流程必须有 E2E 测试
- 每个 Sub-Agent 单独可测试

### 11.3 Git 规范
- 分支：`main` / `dev` / `feature/*` / `fix/*`
- Commit：`<type>(<scope>): <desc>`，type ∈ feat/fix/docs/refactor/test

---

## 12. 里程碑与交付物

| 阶段 | 周期 | 交付物 | 验收 |
|------|------|--------|------|
| M0 | 2 周 | 环境搭建 + Hello World | LangGraph 跑通 |
| M1 | 10 周 | MVP（单产线单缺陷） | 准确率 ≥ 70% |
| M2 | 8 周 | 6 角色 Agent 协作 | 协作流程可视化 |
| M3 | 6 周 | 三层记忆 + 知识沉淀 | 检索准确率 ≥ 80% |
| M4 | 14 周 | 多产线接入 | 3 产线并行 |
| M5 | 12 周 | 自进化闭环 | 缺陷率 ↓ 15% |

---

## 13. 开放问题

1. 机理模型选型：自研简化模型 vs 接商业软件（如 Thermo-Calc）？
2. LLM 选型：通义千问 vs DeepSeek vs GPT-4o？涉及成本与合规
3. 案例冷启动：初始案例从哪来？是否需要人工标注 50 条种子案例？
4. 操作员反馈激励：如何提升反馈数据质量？
5. 多产线差异：不同产线工艺差异大，如何抽象通用 Agent？

---

**文档维护**：本文档随代码迭代同步更新，重大架构变更需技术评审。
