"""
MetaCraft Agent 调参效果跟踪（M5-1）

T+N 天自动跟踪调参后批次质量，对比调整前后指标。

数据流：
1. Agent 产出 proposal → 调用 schedule_tracking() 创建 pending 记录
2. T+N 天后调用 track_effect() → 查询调参后批次质量 → 计算改善百分比
3. 效果数据反馈到案例库（M5-2 效果归因）

由于无真实 MES 接入，提供 quality_fetcher 回调可注入；
默认 _default_quality_fetcher 基于 batch_id 哈希生成稳定质量指标。
"""
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from loguru import logger

from agent.memory.memory_service import MemoryService


@dataclass
class TrackingRecord:
    """效果跟踪记录"""
    tracking_id: str
    proposal_id: str
    case_id: str
    line_id: str
    batch_id_before: str
    batch_id_after: str
    metric_before: Optional[float]
    metric_after: Optional[float]
    improvement_pct: Optional[float]
    status: str  # pending / tracked / skipped
    days_offset: int
    scheduled_at: datetime
    tracked_at: Optional[datetime]
    note: Optional[str]


# 质量指标获取函数签名：batch_id, line_id -> 缺陷率（0-1，越低越好）
QualityFetcher = Callable[[str, str], Optional[float]]


def _default_quality_fetcher(batch_id: str, line_id: str) -> Optional[float]:
    """默认质量指标获取器（无真实 MES 时用）

    基于 batch_id 哈希生成稳定的缺陷率（0.02-0.30），
    保证同一批次多次查询结果一致，便于测试。
    """
    h = hashlib.md5(f"{line_id}:{batch_id}".encode()).hexdigest()
    # 取前 8 位 hex 转 int，映射到 0.02-0.30
    val = int(h[:8], 16) % 2800 / 10000.0 + 0.02
    return round(val, 4)


class EffectTracker:
    """调参效果跟踪器（M5-1）

    用法：
        tracker = EffectTracker(memory)
        # Agent 产出建议后调度跟踪
        tracker.schedule_tracking(
            proposal_id="P001",
            case_id="case_001",
            batch_id_before="B001",
            line_id="heat_treatment",
            metric_before=0.15,
            days_offset=7,
        )
        # T+N 天后执行跟踪
        result = tracker.track_effect(tracking_id)
    """

    def __init__(
        self,
        memory: MemoryService,
        quality_fetcher: Optional[QualityFetcher] = None,
    ):
        self.memory = memory
        self.db = memory.db
        # 质量指标获取器（可注入，默认模拟）
        self.quality_fetcher = quality_fetcher or _default_quality_fetcher

    # ===== 调度跟踪 =====

    def schedule_tracking(
        self,
        proposal_id: str,
        case_id: str,
        batch_id_before: str,
        line_id: str = "heat_treatment",
        metric_before: Optional[float] = None,
        days_offset: int = 7,
        note: Optional[str] = None,
    ) -> str:
        """创建待跟踪记录

        Agent 产出建议后调用，记录调参前批次和指标，
        在 T+days_offset 天后可执行跟踪。

        Args:
            proposal_id: 建议ID
            case_id: 关联案例ID
            batch_id_before: 调参前批次ID
            line_id: 产线ID
            metric_before: 调参前质量指标（缺陷率，0-1）
            days_offset: T+N 天后跟踪
            note: 备注

        Returns:
            tracking_id
        """
        tracking_id = f"trk_{uuid.uuid4().hex[:10]}"
        scheduled_at = datetime.now() + timedelta(days=days_offset)

        self.db.execute(
            """INSERT INTO effect_tracking
            (tracking_id, proposal_id, case_id, line_id,
             batch_id_before, batch_id_after,
             metric_before, metric_after, improvement_pct,
             status, days_offset, scheduled_at, tracked_at, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tracking_id, proposal_id, case_id, line_id,
             batch_id_before, None,
             metric_before, None, None,
             "pending", days_offset, scheduled_at, None, note),
        )
        self.db.commit()
        logger.info(
            f"M5-1 跟踪已调度: {tracking_id}, proposal={proposal_id}, "
            f"line={line_id}, T+{days_offset}d, scheduled_at={scheduled_at}"
        )
        return tracking_id

    # ===== 执行跟踪 =====

    def track_effect(
        self,
        tracking_id: str,
        batch_id_after: Optional[str] = None,
    ) -> Optional[dict]:
        """执行效果跟踪

        查询调参后批次质量，对比调整前后指标，计算改善百分比。

        Args:
            tracking_id: 跟踪记录ID
            batch_id_after: 调参后批次ID（可选，None 时用 batch_id_before 衍生）

        Returns:
            跟踪结果字典，包含 metric_after / improvement_pct / status
        """
        rec = self.get_tracking(tracking_id)
        if rec is None:
            logger.warning(f"跟踪记录不存在: {tracking_id}")
            return None

        if rec["status"] == "tracked":
            logger.info(f"跟踪记录已执行过: {tracking_id}")
            return rec

        # 调参后批次：未指定时用 batch_id_before + "_after" 衍生
        bid_after = batch_id_after or f"{rec['batch_id_before']}_after"

        try:
            metric_after = self.quality_fetcher(bid_after, rec["line_id"])
        except Exception as e:
            logger.error(f"获取调参后批次质量失败: {e}")
            metric_after = None

        metric_before = rec["metric_before"]
        improvement_pct = None
        if metric_before is not None and metric_after is not None:
            # 改善百分比：(before - after) / before * 100，正值表示缺陷率下降
            if metric_before > 0:
                improvement_pct = round(
                    (metric_before - metric_after) / metric_before * 100, 2
                )

        tracked_at = datetime.now()
        self.db.execute(
            """UPDATE effect_tracking
            SET batch_id_after = ?, metric_after = ?, improvement_pct = ?,
                status = 'tracked', tracked_at = ?
            WHERE tracking_id = ?""",
            (bid_after, metric_after, improvement_pct, tracked_at, tracking_id),
        )
        self.db.commit()

        result = {
            "tracking_id": tracking_id,
            "proposal_id": rec["proposal_id"],
            "case_id": rec["case_id"],
            "line_id": rec["line_id"],
            "batch_id_before": rec["batch_id_before"],
            "batch_id_after": bid_after,
            "metric_before": metric_before,
            "metric_after": metric_after,
            "improvement_pct": improvement_pct,
            "status": "tracked",
            "tracked_at": tracked_at.isoformat(),
        }
        logger.info(
            f"M5-1 跟踪完成: {tracking_id}, "
            f"before={metric_before}, after={metric_after}, "
            f"improvement={improvement_pct}%"
        )
        return result

    # ===== 查询 =====

    def get_tracking(self, tracking_id: str) -> Optional[dict]:
        """查询单条跟踪记录"""
        cur = self.db.execute(
            "SELECT * FROM effect_tracking WHERE tracking_id = ?",
            (tracking_id,),
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            return None
        rec = dict(zip(cols, row))
        # 序列化 datetime
        for k in ("scheduled_at", "tracked_at"):
            if rec.get(k) and isinstance(rec[k], str) is False:
                rec[k] = rec[k].isoformat() if hasattr(rec[k], "isoformat") else str(rec[k])
        return rec

    def list_trackings(
        self,
        line_id: Optional[str | list[str]] = None,
        status: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> list[dict]:
        """列出跟踪记录

        Args:
            line_id: 产线ID过滤（支持 list[str]，M5-1 多产线 IN 查询）
            status: 状态过滤（pending/tracked/skipped）
            days: 近 N 天
            limit: 最大返回数

        Returns:
            跟踪记录列表
        """
        since = datetime.now() - timedelta(days=days)
        query = "SELECT * FROM effect_tracking WHERE scheduled_at >= ?"
        params: list = [since]

        if line_id:
            if isinstance(line_id, (list, tuple)):
                placeholders = ",".join("?" for _ in line_id)
                query += f" AND line_id IN ({placeholders})"
                params.extend(line_id)
            else:
                query += " AND line_id = ?"
                params.append(line_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY scheduled_at DESC LIMIT ?"
        params.append(limit)

        cur = self.db.execute(query, params)
        cols = [d[0] for d in cur.description]
        records = []
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            for k in ("scheduled_at", "tracked_at"):
                if rec.get(k) and hasattr(rec[k], "isoformat"):
                    rec[k] = rec[k].isoformat()
            records.append(rec)
        return records

    def list_pending(self, line_id: Optional[str] = None) -> list[dict]:
        """列出已到跟踪时间但未执行的记录（status=pending 且 scheduled_at<=now）"""
        now = datetime.now()
        query = (
            "SELECT * FROM effect_tracking "
            "WHERE status = 'pending' AND scheduled_at <= ?"
        )
        params: list = [now]
        if line_id:
            query += " AND line_id = ?"
            params.append(line_id)
        query += " ORDER BY scheduled_at ASC"

        cur = self.db.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ===== 统计 =====

    def get_effect_stats(self, line_id: Optional[str] = None, days: int = 30) -> dict:
        """获取效果跟踪统计（M5-1 概览）

        Args:
            line_id: 产线ID过滤（None 则全部）
            days: 近 N 天

        Returns:
            {
                "total": int,           # 总跟踪数
                "tracked": int,         # 已跟踪数
                "pending": int,         # 待跟踪数
                "skipped": int,         # 跳过数
                "avg_improvement": float,  # 平均改善百分比
                "positive_count": int,  # 改善>0 的数量
                "negative_count": int,  # 改善<0 的数量
            }
        """
        since = datetime.now() - timedelta(days=days)
        base = "FROM effect_tracking WHERE scheduled_at >= ?"
        params = [since]

        if line_id:
            base += " AND line_id = ?"
            params.append(line_id)

        # 总数
        cur = self.db.execute(f"SELECT COUNT(*) {base}", params)
        total = cur.fetchone()[0]

        # 各状态计数
        stats = {"total": total, "tracked": 0, "pending": 0, "skipped": 0,
                 "avg_improvement": 0.0, "positive_count": 0, "negative_count": 0}
        if total == 0:
            return stats

        cur = self.db.execute(
            f"SELECT status, COUNT(*) {base} GROUP BY status", params
        )
        for status, cnt in cur.fetchall():
            stats[status] = cnt

        # 已跟踪记录的改善统计
        cur = self.db.execute(
            f"""SELECT improvement_pct {base}
               AND status = 'tracked' AND improvement_pct IS NOT NULL""",
            params
        )
        improvements = [row[0] for row in cur.fetchall()]
        if improvements:
            stats["avg_improvement"] = round(sum(improvements) / len(improvements), 2)
            stats["positive_count"] = sum(1 for x in improvements if x > 0)
            stats["negative_count"] = sum(1 for x in improvements if x < 0)

        return stats

    # ===== 批量执行 =====

    def run_due_trackings(self, line_id: Optional[str] = None) -> dict:
        """执行所有到期的待跟踪记录（scheduled_at <= now 且 status=pending）

        用于定时任务（如 cron / APScheduler）每日扫描。

        Args:
            line_id: 限定产线（None 则全部）

        Returns:
            {"executed": int, "succeeded": int, "failed": int}
        """
        pending = self.list_pending(line_id=line_id)
        executed = 0
        succeeded = 0
        failed = 0
        for rec in pending:
            try:
                result = self.track_effect(rec["tracking_id"])
                if result is not None:
                    succeeded += 1
                else:
                    failed += 1
                executed += 1
            except Exception as e:
                logger.error(f"跟踪 {rec['tracking_id']} 失败: {e}")
                failed += 1
                executed += 1
        logger.info(f"M5-1 批量跟踪: 执行 {executed}，成功 {succeeded}，失败 {failed}")
        return {"executed": executed, "succeeded": succeeded, "failed": failed}
