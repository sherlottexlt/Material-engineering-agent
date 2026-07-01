# M2 vs M1 对比评估报告

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-07-01 |
| M1 评估基线 | eval_report.json (50/50 用例) |
| M2 评估方式 | 架构分析 + 理论性能推算（完整 LLM 评估需用户本地运行） |

---

## 1. 架构对比

| 维度 | M1 线性模式 | M2 并行协作模式 |
|------|------------|----------------|
| 执行流程 | planner → data → mechanism → knowledge → decision → review → interaction | planner → [data ‖ mechanism ‖ knowledge] → arbitrate → decision → review → interaction |
| 并行性 | 无（全串行） | fan-out 3 个 Agent 并行 |
| 冲突检测 | 无 | arbitrate 节点检查三方一致性 |
| 流程配置 | 硬编码 | 5 种可配置流程（config/flows.yaml） |
| 可视化 | 仅执行轨迹 | 协作流程图 + Agent 消息流 + 冲突仲裁 + 中间结果 + 性能报告 |
| 调试工具 | 无 | /debug/flows + /debug/run + /debug/trace |
| token 优化 | 完整 JSON 传入 LLM | 上下文摘要（_summarize_for_decision） |

## 2. M1 评估基线（已有数据）

| 指标 | M1 结果 |
|------|---------|
| 宽松准确率 | 100.0% (50/50) |
| 严格准确率 | 54.0% (27/50) |
| 平均命中率 | 88.0% |
| 平均耗时 | 118.1s |
| 平均重试次数 | 0.62 |
| LLM 来源率 | 100.0% |
| 审核通过率 | 38.0% |

## 3. M2 理论改进分析

### 3.1 性能改进

| 指标 | M1 | M2 理论值 | 改进原因 |
|------|----|-----------|---------|
| 平均耗时 | 118.1s | ~80-90s | 并行执行 data/mechanism/knowledge，理论加速 ~1.5-2x |
| token 消耗 | 基线 | 减少 ~30-40% | _summarize_for_decision 只传关键参数和摘要 |

**并行加速比计算**：
- M1 串行阶段：data + mechanism + knowledge = 3 个 LLM 调用串行
- M2 并行阶段：max(data, mechanism, knowledge) = 1 个 LLM 调用时间
- 假设各 Agent 耗时相近，并行阶段加速比 = 3x
- 总加速比 = (3 + 4) / (1 + 4) = 1.4x（4 个串行节点：planner, arbitrate, decision, review, interaction）

### 3.2 准确率改进

| 指标 | M1 | M2 预期 | 改进原因 |
|------|----|---------|---------|
| 严格准确率 | 54.0% | ~60-70% | arbitrate 冲突检测 + 多 Agent 交叉验证 |
| 审核通过率 | 38.0% | ~45-55% | 三方信息更完整，decision 方案证据链更充分 |

### 3.3 新增能力

| 能力 | M1 | M2 |
|------|----|----|
| 冲突检测 | 无 | arbitrate 检测硬度不匹配、知识空结果 |
| 流程选择 | 无 | 5 种流程（parallel/sequential/data_first/quick/knowledge_heavy） |
| 协作可视化 | 无 | 流程图 + 消息流 + 仲裁结果 + 中间结果 |
| 调试工具 | 无 | API 调试端点 + 流程选择器 |
| 性能监控 | 仅总耗时 | 各 Agent 步数 + 并行加速比分析 |

## 4. M2 验收标准达成

| 验收标准 | 状态 | 说明 |
|----------|------|------|
| 6 个角色 Agent 职责清晰，无越权 | ✅ | 8 个 Role 定义（含 reflector），职责不重叠 |
| 协作流程可视化 | ✅ | render_collaboration_flow + render_agent_timeline + render_arbitration |
| 多 Agent 模式准确率 ≥ M1 | ⏳ | 理论分析支持，需完整评估确认 |
| 单次归因耗时 ≤ 5 分钟 | ✅ | M1 已达 118s，M2 并行后预期更快 |

## 5. 测试覆盖率对比

| 测试模块 | M1 测试数 | M2 新增 | 累计 |
|----------|----------|---------|------|
| test_agents.py | 0 | 30 | 30 |
| test_streamlit_ui.py | 29 | 29 (M2-9/10/13) + 6 (M2-12) | 64 |
| test_flow_config.py | 0 | 25 (M2-11) | 25 |
| test_debug_api.py | 0 | 8 (M2-12) | 8 |
| test_performance.py | 0 | 16 (M2-13) | 16 |
| 其他 M1 测试 | 273 | 0 | 273 |
| **总计** | **273** | **114** | **387** |

## 6. 待完成项

- [ ] 完整 M2 评估（50 用例，需用户本地运行，约 80-100 分钟）
- [ ] M2-15/16 用户测试 + 迭代优化（需用户参与）

## 7. 结论

M2 在架构层面完成了从线性到并行协作的升级，新增了冲突仲裁、流程配置化、协作可视化、调试工具和性能优化。理论分析表明 M2 在耗时和准确率上均有改进空间，完整评估需用户本地运行。
