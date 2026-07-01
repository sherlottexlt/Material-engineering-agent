"""
MetaCraft Agent 评估指标
对应 TDD 第 9 节

三层评估：
1. 任务级：归因准确率、采纳率、响应时间
2. Agent 级：token 消耗、工具成功率、重规划率
3. 业务级：缺陷率变化、操作员信任度
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class TaskMetrics:
    """任务级指标"""
    trace_id: str
    attribution_accuracy: Optional[float] = None  # 归因准确率
    adoption_status: Optional[str] = None          # adopted/rejected/partial
    response_time_seconds: Optional[float] = None  # 响应时间
    token_consumption: int = 0                     # token 消耗
    tool_calls: int = 0                            # 工具调用次数
    tool_failures: int = 0                         # 工具失败次数
    replan_count: int = 0                          # 重规划次数
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessMetrics:
    """业务级指标"""
    period: str  # daily/weekly/monthly
    defect_rate_before: float
    defect_rate_after: float
    adoption_rate: float           # 建议采纳率
    avg_response_time: float       # 平均响应时间
    operator_trust_score: float    # 操作员信任度（问卷 1-5）
    cases_collected: int           # 新增案例数
    roi: Optional[float] = None    # 投资回报率


class MetricsCollector:
    """指标收集器"""

    def __init__(self, db_path: str = "data/metrics.db"):
        self.db_path = db_path
        # TODO: 初始化 SQLite 存储指标
        self._task_metrics: list[TaskMetrics] = []
        self._start_times: dict[str, float] = {}

    def start_task(self, trace_id: str):
        """记录任务开始时间"""
        self._start_times[trace_id] = time.time()

    def end_task(self, trace_id: str, **kwargs) -> TaskMetrics:
        """记录任务结束"""
        elapsed = time.time() - self._start_times.pop(trace_id, time.time())
        metrics = TaskMetrics(
            trace_id=trace_id,
            response_time_seconds=round(elapsed, 2),
            **kwargs,
        )
        self._task_metrics.append(metrics)
        logger.info(f"任务指标已记录: {trace_id}, 耗时 {elapsed:.1f}s")
        return metrics

    def calculate_accuracy(self, predictions: list[dict], ground_truth: list[dict]) -> float:
        """计算归因准确率

        Args:
            predictions: Agent 预测的根因列表
            ground_truth: 人工标注的真实根因列表

        Returns:
            准确率 0-1
        """
        if not ground_truth:
            return 0.0

        correct = 0
        for pred, truth in zip(predictions, ground_truth):
            # TODO: 实现更精确的匹配逻辑（语义相似度）
            if pred.get("root_cause") == truth.get("root_cause"):
                correct += 1

        return correct / len(ground_truth)

    def calculate_adoption_rate(self, feedbacks: list[dict]) -> float:
        """计算建议采纳率"""
        if not feedbacks:
            return 0.0
        adopted = sum(1 for f in feedbacks if f.get("action") == "adopted")
        return adopted / len(feedbacks)

    def export_report(self, output_path: str = "data/eval_report.json"):
        """导出评估报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_tasks": len(self._task_metrics),
            "avg_response_time": (
                sum(m.response_time_seconds for m in self._task_metrics if m.response_time_seconds)
                / max(len([m for m in self._task_metrics if m.response_time_seconds]), 1)
            ),
            "total_token_consumption": sum(m.token_consumption for m in self._task_metrics),
            "avg_tool_failures": (
                sum(m.tool_failures for m in self._task_metrics)
                / max(len(self._task_metrics), 1)
            ),
            "replan_rate": (
                len([m for m in self._task_metrics if m.replan_count > 0])
                / max(len(self._task_metrics), 1)
            ),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"评估报告已导出: {output_path}")
        return report
