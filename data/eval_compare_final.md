# 评估报告对比（baseline vs prompt 优化后）

- baseline: `data/eval_report_before_prompt_opt.json` (50 条，旧 review_agent prompt)
- 新报告: `data/eval_report.json` (50 条，新 review_agent prompt + JMAK bug 修复)

## 1. 汇总指标对比

| 指标 | 旧 prompt | 新 prompt | Delta | 判定 |
|---|---|---|---|---|
| 用例总数 | 50 | 50 | +0 | — |
| 成功用例数 | 50 | 50 | +0 | — |
| 错误用例数 | 0 | 0 | +0 | 持平 |
| 宽松准确率 | 92.0% | 100.0% | +8.0% ↓改善 | 改善 |
| 严格准确率 | 22.0% | 54.0% | +32.0% ↓改善 | 改善 |
| 平均命中率 | 71.0% | 88.0% | +17.0% ↓改善 | 改善 |
| 平均耗时(s) | 165.60 | 118.10 | -47.50 ↓改善 | 改善 |
| 平均 retry 次数 | 0.84 | 0.62 | -0.22 ↓改善 | 改善 |
| LLM 来源率 | 100.0% | 100.0% | +0.0% | — |
| 审核通过率 | 16.0% | 38.0% | +22.0% ↓改善 | 改善 |


## 2. 按 root_cause 分类对比

| 根因 | 旧 total | 新 total | 旧 loose | 新 loose | 旧 strict | 新 strict | 旧 hit_rate | 新 hit_rate |
|---|---|---|---|---|---|---|---|---|
| 保温时间不足 | 15 | 15 | 87.0% | 100.0% | 0.0% | 53.0% | 57.0% | 87.0% |
| 保温时间不足+冷却速率过低 | 4 | 4 | 100.0% | 100.0% | 75.0% | 100.0% | 88.0% | 100.0% |
| 保温时间不足+温度偏低 | 3 | 3 | 100.0% | 100.0% | 67.0% | 100.0% | 83.0% | 100.0% |
| 冷却速率过低 | 15 | 15 | 100.0% | 100.0% | 0.0% | 20.0% | 75.0% | 80.0% |
| 冷却速率过低+温度偏低 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 温度偏低 | 10 | 10 | 80.0% | 100.0% | 30.0% | 60.0% | 68.0% | 90.0% |


## 3. 按难度对比

| 难度 | 旧 total | 新 total | 旧 loose_acc | 新 loose_acc | Delta |
|---|---|---|---|---|---|
| easy | 16 | 16 | 94.0% | 100.0% | +6.0% ↓改善 |
| hard | 16 | 16 | 94.0% | 100.0% | +6.0% ↓改善 |
| medium | 18 | 18 | 89.0% | 100.0% | +11.0% ↓改善 |


## 5. 审核通过模式对比

| 指标 | 旧 prompt | 新 prompt | Delta |
|---|---|---|---|
| 成功用例数 | 50 | 50 | +0 |
| 审核通过数 | 8 | 19 | +11 |
| 审核拒绝数 | 42 | 31 | -11 |
| 审核通过率 | 16.0% | 38.0% | +22.0% ↓改善 |
| 通过且 retry=0 | 8 | 19 | +11 |
| 通过且 retry=1（重试后通过） | 0 | 0 | +0 |
| 拒绝且 retry=0 | 0 | 0 | +0 |
| 拒绝且 retry=1（重试后仍拒绝） | 42 | 31 | -11 |


## 4. 逐案差异（仅显示有变化的用例）

共 31 条用例有变化：

| case_id | 期望根因 | 变化项 |
|---|---|---|
| SC-002 | 保温时间不足 | retry: 0 → 1 (恶化)<br>review: True → False (恶化) |
| SC-003 | 保温时间不足 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-004 | 保温时间不足 | hit_rate: 0.50 → 0.75 |
| SC-005 | 保温时间不足 | loose: False → True (改善)<br>strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.00 → 1.00 |
| SC-006 | 保温时间不足 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-007 | 保温时间不足 | loose: False → True (改善)<br>strict: False → True (改善)<br>hit_rate: 0.00 → 1.00 |
| SC-009 | 保温时间不足 | strict: False → True (改善)<br>hit_rate: 0.50 → 1.00 |
| SC-010 | 保温时间不足 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.50 → 1.00 |
| SC-011 | 保温时间不足 | strict: False → True (改善)<br>hit_rate: 0.50 → 1.00 |
| SC-012 | 保温时间不足 | hit_rate: 0.75 → 0.50 |
| SC-013 | 保温时间不足 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-014 | 保温时间不足 | retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.50 → 0.75 |
| SC-016 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-017 | 冷却速率过低 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-019 | 冷却速率过低 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-020 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-024 | 冷却速率过低 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-027 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-030 | 冷却速率过低 | retry: 0 → 1 (恶化)<br>review: True → False (恶化) |
| SC-031 | 温度偏低 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-034 | 温度偏低 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-035 | 温度偏低 | strict: False → True (改善)<br>retry: 0 → 1 (恶化)<br>review: True → False (恶化)<br>hit_rate: 0.75 → 1.00 |
| SC-036 | 温度偏低 | strict: True → False (恶化)<br>hit_rate: 1.00 → 0.75 |
| SC-037 | 温度偏低 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-038 | 温度偏低 | strict: True → False (恶化)<br>hit_rate: 1.00 → 0.75 |
| SC-039 | 温度偏低 | loose: False → True (改善)<br>hit_rate: 0.00 → 0.75 |
| SC-040 | 温度偏低 | loose: False → True (改善)<br>strict: False → True (改善)<br>hit_rate: 0.00 → 1.00 |
| SC-042 | 冷却速率过低+温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-044 | 保温时间不足+冷却速率过低 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.50 → 1.00 |
| SC-048 | 冷却速率过低+温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-049 | 保温时间不足+温度偏低 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.50 → 1.00 |
