# M4-13 故障演练报告

**测试时间**：2026-07-03T01:07:45.378231
**演练场景**：llm_failure, chroma_failure, sqlite_failure, combined_failure
**测试端点**：auth, cases, dashboard, feedback, analyze
**每端点重复**：1 次

**MES 说明**：MES 当前为 mock 模式（无真实接入），不构成单点故障，未单独演练

## 1. 基线（无故障）

| 端点 | 描述 | 状态码 | 平均耗时(ms) | 降级 |
|------|------|--------|-------------|------|
| auth | 权限查询（轻量，对照） | 200 | 12.82 | baseline |
| cases | 案例查询（读 SQLite + Chroma） | 200 | 4.49 | baseline |
| dashboard | 跨产线看板（读 SQLite + Chroma 全量扫描） | 200 | 9062.65 | baseline |
| feedback | 反馈写入（写 SQLite） | 200 | 4626.28 | baseline |
| analyze | 归因分析（调 LLM + 全链路） | N/A | 0 | skipped |

## 2. 场景：llm_failure（LLM 服务不可用）

**注入故障**：llm

| 端点 | 描述 | 状态码 | 平均耗时(ms) | 降级判定 | 错误信息 |
|------|------|--------|-------------|---------|---------|
| auth | 权限查询（轻量，对照） | 200 | 3.26 | normal |  |
| cases | 案例查询（读 SQLite + Chroma） | 200 | 5.45 | normal |  |
| dashboard | 跨产线看板（读 SQLite + Chroma 全量扫描） | 200 | 9091.61 | normal |  |
| feedback | 反馈写入（写 SQLite） | 200 | 4624.97 | normal |  |
| analyze | 归因分析（调 LLM + 全链路） | 200 | 18288.37 | degraded_ok |  |

## 2. 场景：chroma_failure（向量库（Chroma）不可用）

**注入故障**：chroma

| 端点 | 描述 | 状态码 | 平均耗时(ms) | 降级判定 | 错误信息 |
|------|------|--------|-------------|---------|---------|
| auth | 权限查询（轻量，对照） | 200 | 3.39 | normal |  |
| cases | 案例查询（读 SQLite + Chroma） | 200 | 5.77 | degraded_ok |  |
| dashboard | 跨产线看板（读 SQLite + Chroma 全量扫描） | 200 | 9130.53 | degraded_ok |  |
| feedback | 反馈写入（写 SQLite） | 200 | 4631.22 | normal |  |
| analyze | 归因分析（调 LLM + 全链路） | N/A | 0 | skipped |  |

## 2. 场景：sqlite_failure（SQLite 数据库不可用）

**注入故障**：sqlite

| 端点 | 描述 | 状态码 | 平均耗时(ms) | 降级判定 | 错误信息 |
|------|------|--------|-------------|---------|---------|
| auth | 权限查询（轻量，对照） | 200 | 5.18 | normal |  |
| cases | 案例查询（读 SQLite + Chroma） | -1 | 3.27 | failed | database is locked（故障注入） |
| dashboard | 跨产线看板（读 SQLite + Chroma 全量扫描） | -1 | 5.22 | failed | database is locked（故障注入） |
| feedback | 反馈写入（写 SQLite） | 500 | 3.32 | failed |  |
| analyze | 归因分析（调 LLM + 全链路） | N/A | 0 | skipped |  |

## 2. 场景：combined_failure（LLM + Chroma 同时故障）

**注入故障**：llm, chroma

| 端点 | 描述 | 状态码 | 平均耗时(ms) | 降级判定 | 错误信息 |
|------|------|--------|-------------|---------|---------|
| auth | 权限查询（轻量，对照） | 200 | 2.93 | normal |  |
| cases | 案例查询（读 SQLite + Chroma） | 200 | 3.79 | degraded_ok |  |
| dashboard | 跨产线看板（读 SQLite + Chroma 全量扫描） | 200 | 9077.4 | degraded_ok |  |
| feedback | 反馈写入（写 SQLite） | 200 | 4613.74 | normal |  |
| analyze | 归因分析（调 LLM + 全链路） | 200 | 4591.31 | degraded_ok |  |

## 3. 降级汇总

| 场景 | 端点 | 依赖故障 | 降级判定 |
|------|------|---------|---------|
| llm_failure | auth | 否 | normal |
| llm_failure | cases | 否 | normal |
| llm_failure | dashboard | 否 | normal |
| llm_failure | feedback | 否 | normal |
| llm_failure | analyze | 是 | degraded_ok |
| chroma_failure | auth | 否 | normal |
| chroma_failure | cases | 是 | degraded_ok |
| chroma_failure | dashboard | 是 | degraded_ok |
| chroma_failure | feedback | 否 | normal |
| chroma_failure | analyze | 是 | skipped |
| sqlite_failure | auth | 否 | normal |
| sqlite_failure | cases | 是 | failed |
| sqlite_failure | dashboard | 是 | failed |
| sqlite_failure | feedback | 是 | failed |
| sqlite_failure | analyze | 是 | skipped |
| combined_failure | auth | 否 | normal |
| combined_failure | cases | 是 | degraded_ok |
| combined_failure | dashboard | 是 | degraded_ok |
| combined_failure | feedback | 否 | normal |
| combined_failure | analyze | 是 | degraded_ok |

## 4. 验收标准对照

| 验收项 | 标准 | 结果 |
|--------|------|------|
| 故障命中端点降级率 | ≥ 90% | ❌ 未达标（6/9 = 66.7%） |

## 5. 弱点清单（故障命中但未降级）

| 场景 | 端点 | 说明 |
|------|------|------|
| sqlite_failure | cases | 故障命中依赖但返回 5xx，无降级 |
| sqlite_failure | dashboard | 故障命中依赖但返回 5xx，无降级 |
| sqlite_failure | feedback | 故障命中依赖但返回 5xx，无降级 |

> 以上弱点需在 M4-14 降级策略中补齐（加 try/except + 降级返回）。

## 6. 结论

- 故障命中端点降级率：66.7%（6/9）
- 未降级弱点数：3
- 弱点：sqlite_failure/cases, sqlite_failure/dashboard, sqlite_failure/feedback
- 建议：M4-14 对弱点端点补齐 try/except + 降级返回 + 全局异常处理。
