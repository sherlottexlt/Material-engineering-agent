# 评估报告对比

- 旧报告: `data/eval_report.json` (50 条)
- 新报告: `data/eval_report_m2.json` (50 条)

## 1. 汇总指标对比

| 指标 | 旧 prompt | 新 prompt | Delta | 判定 |
|---|---|---|---|---|
| 用例总数 | 50 | 50 | +0 | — |
| 成功用例数 | 50 | 50 | +0 | — |
| 错误用例数 | 0 | 0 | +0 | 持平 |
| 宽松准确率 | 100.0% | 100.0% | +0.0% | 持平 |
| 严格准确率 | 54.0% | 70.0% | +16.0% ↓改善 | 改善 |
| 平均命中率 | 88.0% | 92.5% | +4.5% ↓改善 | 改善 |
| 平均耗时(s) | 118.10 | 108.40 | -9.70 ↓改善 | 改善 |
| 平均 retry 次数 | 0.62 | 0.20 | -0.42 ↓改善 | 改善 |
| LLM 来源率 | 100.0% | 60.0% | -40.0% ↓改善 | — |
| 审核通过率 | 38.0% | 80.0% | +42.0% ↓改善 | 改善 |


## 2. 按 root_cause 分类对比

| 根因 | 旧 total | 新 total | 旧 loose | 新 loose | 旧 strict | 新 strict | 旧 hit_rate | 新 hit_rate |
|---|---|---|---|---|---|---|---|---|
| 保温时间不足 | 15 | 15 | 100.0% | 100.0% | 53.0% | 100.0% | 87.0% | 100.0% |
| 保温时间不足+冷却速率过低 | 4 | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 保温时间不足+温度偏低 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 冷却速率过低 | 15 | 15 | 100.0% | 100.0% | 20.0% | 0.0% | 80.0% | 75.0% |
| 冷却速率过低+温度偏低 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 温度偏低 | 10 | 10 | 100.0% | 100.0% | 60.0% | 100.0% | 90.0% | 100.0% |


## 3. 按难度对比

| 难度 | 旧 total | 新 total | 旧 loose_acc | 新 loose_acc | Delta |
|---|---|---|---|---|---|
| easy | 16 | 16 | 100.0% | 100.0% | +0.0% |
| hard | 16 | 16 | 100.0% | 100.0% | +0.0% |
| medium | 18 | 18 | 100.0% | 100.0% | +0.0% |


## 5. 审核通过模式对比

| 指标 | 旧 prompt | 新 prompt | Delta |
|---|---|---|---|
| 成功用例数 | 50 | 50 | +0 |
| 审核通过数 | 19 | 40 | +21 |
| 审核拒绝数 | 31 | 10 | -21 |
| 审核通过率 | 38.0% | 80.0% | +42.0% ↓改善 |
| 通过且 retry=0 | 19 | 40 | +21 |
| 通过且 retry=1（重试后通过） | 0 | 0 | +0 |
| 拒绝且 retry=0 | 0 | 0 | +0 |
| 拒绝且 retry=1（重试后仍拒绝） | 31 | 10 | -21 |


## 4. 逐案差异（仅显示有变化的用例）

共 35 条用例有变化：

| case_id | 期望根因 | 变化项 |
|---|---|---|
| SC-001 | 保温时间不足 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-002 | 保温时间不足 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-003 | 保温时间不足 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-004 | 保温时间不足 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-007 | 保温时间不足 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-008 | 保温时间不足 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-009 | 保温时间不足 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-011 | 保温时间不足 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-012 | 保温时间不足 | strict: False → True (改善)<br>hit_rate: 0.50 → 1.00 |
| SC-013 | 保温时间不足 | retry: 0 → 1 (恶化)<br>review: True → False (恶化) |
| SC-014 | 保温时间不足 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-015 | 保温时间不足 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-017 | 冷却速率过低 | strict: True → False (恶化)<br>hit_rate: 1.00 → 0.75 |
| SC-018 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-019 | 冷却速率过低 | strict: True → False (恶化)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 1.00 → 0.75 |
| SC-023 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-024 | 冷却速率过低 | strict: True → False (恶化)<br>retry: 0 → 1 (恶化)<br>review: True → False (恶化)<br>hit_rate: 1.00 → 0.75 |
| SC-025 | 冷却速率过低 | retry: 0 → 1 (恶化)<br>review: True → False (恶化) |
| SC-026 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-027 | 冷却速率过低 | retry: 0 → 1 (恶化)<br>review: True → False (恶化) |
| SC-029 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-030 | 冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-032 | 温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-033 | 温度偏低 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-035 | 温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-036 | 温度偏低 | strict: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-037 | 温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-038 | 温度偏低 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-039 | 温度偏低 | strict: False → True (改善)<br>retry: 1 → 0 (改善)<br>review: False → True (改善)<br>hit_rate: 0.75 → 1.00 |
| SC-040 | 温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-041 | 保温时间不足+冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-043 | 保温时间不足+温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-046 | 保温时间不足+温度偏低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-047 | 保温时间不足+冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
| SC-050 | 保温时间不足+冷却速率过低 | retry: 1 → 0 (改善)<br>review: False → True (改善) |
