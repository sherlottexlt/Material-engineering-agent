# M4-15 SLA 保障测试报告

**测试时间**：2026-07-03T02:56:09.519487
**测试总数**：15
**通过**：15
**失败**：0
**通过率**：100.0%

## 测试明细

| 测试 | 描述 | 状态 | 耗时(ms) | 错误 |
|------|------|------|---------|------|
| test_sla_monitor_record_and_stats | SLAMonitor 记录请求并查询统计 | PASS | 67.4 |  |
| test_sla_availability_calc | 可用性计算：5xx 算不可用 | PASS | 0.0 |  |
| test_sla_p95_p99 | P95/P99 百分位计算 | PASS | 0.0 |  |
| test_sla_degraded_count | 降级计数 | PASS | 0.0 |  |
| test_sla_empty_stats | 空记录返回默认值 | PASS | 0.0 |  |
| test_sla_window_filter | 时间窗口过滤 | PASS | 0.0 |  |
| test_sla_stats_by_endpoint | 按端点细分统计 | PASS | 0.0 |  |
| test_sla_reset | reset 清空记录 | PASS | 0.0 |  |
| test_sla_status_endpoint | /api/v1/sla/status 端点 | PASS | 7909.0 |  |
| test_sla_report_endpoint_admin | admin 可访问 /api/v1/sla/report | PASS | 4.6 |  |
| test_sla_report_permission_denied | 非 admin 无权访问 /api/v1/sla/report | PASS | 4.4 |  |
| test_sla_middleware_records_requests | SLA 中间件自动记录请求 | PASS | 3.6 |  |
| test_sla_middleware_records_degraded | 中间件记录降级请求（X-Degraded header） | PASS | 4.7 |  |
| test_sla_middleware_records_503 | 中间件记录 503 硬降级 | PASS | 3.0 |  |
| test_sla_99_5_target_met | 99.5% 可用性达标验证 | PASS | 423.9 |  |

## 验收标准对照

| 验收项 | 标准 | 结果 |
|--------|------|------|
| M4-15 SLA 测试通过率 | 100% | ✅ 达标（15/15）|
