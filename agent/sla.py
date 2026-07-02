"""M4-15: SLA 监控与可用性统计

记录请求指标（响应时间、状态码、降级标记），计算 SLA：
- 可用性 availability = (状态码 < 500 的请求数) / 总请求数
- P95/P99 延迟
- 错误率 error_rate = 5xx 请求数 / 总请求数
- 降级次数 degraded_count（响应头 X-Degraded=true 或状态码 503）

SLA 目标：availability >= 99.5%

对应 IMP.md 6.2 节 M4-15。
"""
import threading
import time
from collections import deque
from typing import Optional


class SLAMonitor:
    """SLA 监控器（线程安全的滚动窗口）

    用 deque 存储最近 N 条请求记录，支持按时间窗口查询统计。
    线程安全：所有读写操作加锁。
    """

    SLA_TARGET = 0.995  # 99.5% 可用性目标

    def __init__(self, window_size: int = 10000):
        """初始化

        Args:
            window_size: 滚动窗口大小（保留最近 N 条记录）
        """
        # 滚动窗口：保留最近 window_size 条请求记录
        self._records: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        degraded: bool = False,
        user_id: Optional[str] = None,
        line_id: Optional[str] = None,
    ) -> None:
        """记录一次请求

        Args:
            method: HTTP 方法（GET/POST/...）
            path: 请求路径
            status_code: 响应状态码
            duration_ms: 响应耗时（毫秒）
            degraded: 是否降级（响应头 X-Degraded=true 或 503）
            user_id: 用户ID（可选）
            line_id: 产线ID（可选）
        """
        with self._lock:
            self._records.append({
                "timestamp": time.time(),
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "degraded": degraded,
                "user_id": user_id,
                "line_id": line_id,
            })

    def get_stats(self, window_minutes: int = 60) -> dict:
        """获取最近 window_minutes 分钟内的 SLA 统计

        Args:
            window_minutes: 时间窗口（分钟），默认 60

        Returns:
            {
                "total_requests": int,
                "availability": float,       # 0.0-1.0
                "error_rate": float,         # 0.0-1.0
                "p95_latency_ms": float,
                "p99_latency_ms": float,
                "avg_latency_ms": float,
                "degraded_count": int,
                "sla_target": float,         # 0.995
                "sla_met": bool,             # availability >= sla_target
                "window_minutes": int,
            }
        """
        cutoff = time.time() - window_minutes * 60
        with self._lock:
            records = [r for r in self._records if r["timestamp"] >= cutoff]

        total = len(records)
        if total == 0:
            return {
                "total_requests": 0,
                "availability": 1.0,
                "error_rate": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "degraded_count": 0,
                "sla_target": self.SLA_TARGET,
                "sla_met": True,
                "window_minutes": window_minutes,
            }

        # 可用性：状态码 < 500 的比例（2xx+3xx+4xx 都算可用，5xx 算不可用）
        success = sum(1 for r in records if r["status_code"] < 500)
        availability = success / total

        # 错误率：5xx 比例
        errors = sum(1 for r in records if r["status_code"] >= 500)
        error_rate = errors / total

        # P95/P99/avg 延迟
        latencies = sorted(r["duration_ms"] for r in records)
        p95 = self._percentile(latencies, 95)
        p99 = self._percentile(latencies, 99)
        avg = sum(latencies) / len(latencies)

        # 降级次数
        degraded_count = sum(1 for r in records if r["degraded"])

        return {
            "total_requests": total,
            "availability": round(availability, 4),
            "error_rate": round(error_rate, 4),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "avg_latency_ms": round(avg, 2),
            "degraded_count": degraded_count,
            "sla_target": self.SLA_TARGET,
            "sla_met": availability >= self.SLA_TARGET,
            "window_minutes": window_minutes,
        }

    def get_stats_by_endpoint(self, window_minutes: int = 60) -> dict:
        """按端点细分的 SLA 统计

        Returns:
            {path: {total, availability, p95_latency_ms, error_rate, degraded_count}}
        """
        cutoff = time.time() - window_minutes * 60
        with self._lock:
            records = [r for r in self._records if r["timestamp"] >= cutoff]

        by_path: dict[str, list] = {}
        for r in records:
            by_path.setdefault(r["path"], []).append(r)

        result = {}
        for path, recs in by_path.items():
            total = len(recs)
            success = sum(1 for r in recs if r["status_code"] < 500)
            errors = sum(1 for r in recs if r["status_code"] >= 500)
            latencies = sorted(r["duration_ms"] for r in recs)
            result[path] = {
                "total_requests": total,
                "availability": round(success / total, 4) if total else 1.0,
                "error_rate": round(errors / total, 4) if total else 0.0,
                "p95_latency_ms": round(self._percentile(latencies, 95), 2),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "degraded_count": sum(1 for r in recs if r["degraded"]),
            }
        return result

    @staticmethod
    def _percentile(sorted_list: list, p: float) -> float:
        """计算百分位数（sorted_list 必须已排序）

        Args:
            sorted_list: 已排序的数值列表
            p: 百分位（0-100）

        Returns:
            百分位数值
        """
        if not sorted_list:
            return 0.0
        k = (len(sorted_list) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(sorted_list) - 1)
        if f == c:
            return sorted_list[f]
        return sorted_list[f] + (sorted_list[c] - sorted_list[f]) * (k - f)

    def reset(self) -> None:
        """清空记录（测试用）"""
        with self._lock:
            self._records.clear()


# 全局单例
sla_monitor = SLAMonitor()
