# MetaCraft Agent — 简历项目归纳

> 面向 AI Agent 开发岗位的项目经历描述。可直接粘贴到简历"项目经历"栏目。

---

## 项目经历：MetaCraft Agent — 材料加工产线智能工艺优化 Agent

**角色**：独立开发（全栈 + Agent 架构）
**时间**：2026.06 - 2026.07（持续迭代中）
**代码量**：21,000+ 行 Python / 387 个测试用例
**仓库**：https://github.com/sherlottexlt/Material-engineering-agent

### 一句话定位
面向材料加工产线（热处理等）的智能工艺优化 Agent，自主完成"缺陷归因 → 工艺参数推荐 → 操作员确认 → 效果跟踪"闭环，把资深工艺工程师经验沉淀为可执行系统。

### 技术栈
- **Agent 编排**：LangGraph（状态机建模，避免 ReAct 黑盒）
- **工具协议**：MCP（Model Context Protocol，工具标准化复用）
- **LLM 接入**：通义千问 / DeepSeek（统一调用层，可切换）
- **记忆架构**：Chroma 向量库 + SQLite（三层记忆：工作/短期/长期）
- **后端**：FastAPI（REST + WebSocket）+ LangSmith trace
- **前端**：Streamlit（对话式 + 协作可视化）
- **工程化**：Docker / docker-compose / pytest / 评估脚本

### 核心工作

**1. 设计并实现单 Agent 端到端闭环（M1 MVP）**
- 用 LangGraph 显式建模 7 节点状态机：Planner → Executor → Reflector → Decision → Review → Interaction，避免隐式 ReAct 黑盒，每步可追溯
- 实现 3 个 MCP Server（MES / 机理模型 / 知识 RAG），把外部能力标准化封装，工具可热插拔
- 50 条种子案例评估：宽松准确率 100%、严格准确率 54%、平均耗时 118.1s、审核通过率 38%

**2. 重构为多 Agent 并行协作架构（M2）**
- 将单 Agent 拆为 6 角色子图（data / mechanism / knowledge / decision / review / interaction），通过 coordinator 实现 fan-out 并行 + 汇聚
- 实现 arbitrate 仲裁节点，检测三方信息冲突（如硬度不匹配、知识空结果），多 Agent 交叉验证提升准确率
- 设计 5 种可配置流程（parallel / sequential / data_first / quick / knowledge_heavy），不同场景按需切换，流程配置化而非硬编码

**3. Prompt 工程与评估优化**
- 通过 50 例跑批评估定位 JMAK 机理模型 bug + review_agent prompt 缺陷
- 优化后所有 6 项指标提升：严格准确率 22% → 54%（+32pp）、平均耗时 165.6s → 118.1s（-28%）、审核通过率 16% → 38%（+22pp）
- 建立可复现评估流水线（eval/run_eval.py + eval/compare_eval.py），支持 baseline 对比

**4. 协作可视化与调试工具**
- Streamlit 实现协作流程图 + Agent 消息流时间线 + 仲裁结果展示 + 中间结果可展开查看
- FastAPI 提供 3 个调试端点（/debug/flows + /debug/run + /debug/trace），支持单步执行、流程选择器、回放
- 上下文摘要优化（_summarize_for_decision），token 消耗理论降低 30-40%

**5. 工程化与质量保障**
- 387 个测试用例（0 failures, 2 skipped），覆盖节点单测 / 集成 / 性能 / MCP / UI
- Docker compose 一键启动（Chroma + Postgres + Langfuse + API + UI）
- CI 跑通，全链路 LangSmith trace 可观测

### 关键挑战与解决方案

| 挑战 | 解决方案 |
|------|---------|
| 单 Agent 串行耗时过长 | fan-out 并行 3 个 Agent，理论加速 1.4x（118s → 预期 80-90s） |
| 多 Agent 结论冲突无处理 | 设计 arbitrate 仲裁节点，检测硬度/知识冲突，触发重规划 |
| LLM 输出 JSON 不稳定 | Pydantic 模型校验 + retry 机制 + 反射自评重规划 |
| 工艺手册 PDF/Word 解析 | 自研 ingest_handbooks.py，支持多格式 + BM25-like 检索 |
| 流程硬编码难扩展 | 抽象为 config/flows.yaml，5 种流程可配置 |

### 成果数据（可量化）

| 指标 | 数值 |
|------|------|
| 宽松准确率 | 100%（50/50） |
| 严格准确率 | 54% → 优化前 22%（+32pp） |
| 平均命中率 | 88% → 优化前 71%（+17pp） |
| 平均归因耗时 | 118.1s（< 5 分钟 SLA） |
| 审核通过率 | 38% → 优化前 16%（+22pp） |
| 测试用例数 | 387（0 failures） |
| 代码行数 | 21,000+ |
| Agent 角色 | 6 个（fan-out 并行协作） |
| MCP 工具 | 3 个 Server（MES / 机理 / 知识） |
| 可配置流程 | 5 种 |

### 项目文档（仓库内）
- [PRD.md](./PRD.md) — 产品需求文档
- [TDD.md](./TDD.md) — 技术设计文档
- [IMP.md](./IMP.md) — 实施计划（M0-M5，52 周）
- [data/eval_compare_final.md](./data/eval_compare_final.md) — 评估对比报告
- [data/m2_vs_m1_eval.md](./data/m2_vs_m1_eval.md) — M1 vs M2 对比报告

---

## 简历精简版（一栏 4-5 行）

> **MetaCraft Agent — 材料加工产线智能工艺优化 Agent**（2026.06-至今，独立开发）
> 基于 LangGraph + MCP 的多 Agent 协作系统，实现缺陷归因→参数推荐→人工确认闭环。设计 6 角色并行协作架构 + 仲裁节点，3 个 MCP Server 工具标准化。50 例评估准确率 100%/严格 54%，耗时 118s。387 测试用例，21k 行代码。技术栈：LangGraph / MCP / Chroma / FastAPI / Streamlit / Docker。
