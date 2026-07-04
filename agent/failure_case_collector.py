"""
MetaCraft Agent 失败案例归集（M5-4）

自动收集低分案例 + 反效果跟踪 + 被拒绝反馈，归集到失败案例库。
用于分析失败原因，作为 M5-5 Prompt 自动优化的输入。

三类失败来源：
1. low_confidence: 案例库中 confidence < min_confidence 的案例
2. negative_effect: 效果跟踪中 improvement_pct < 0（反效果）的记录
3. rejected_feedback: 用户反馈 action=rejected 的建议

数据流：
1. collect_all() 扫描三类来源，去重后存入 failure_cases 表
2. list_failures() / get_failure_stats() 查询统计
3. update_failure_status() 标记分析进度（open → analyzed → resolved）
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from agent.memory.memory_service import MemoryService


class FailureCaseCollector:
    """失败案例归集器（M5-4）

    用法：
        collector = FailureCaseCollector(memory)
        # 收集失败案例
        result = collector.collect_all(line_id="heat_treatment", days=30)
        # 查询
        failures = collector.list_failures(line_id="heat_treatment", category="low_confidence")
        stats = collector.get_failure_stats(line_id="heat_treatment", days=30)
    """

    def __init__(self, memory: MemoryService):
        self.memory = memory
        self.db = memory.db

    # ===== 收集 =====

    def collect_low_confidence_cases(
        self,
        line_id: Optional[str | list[str]] = None,
        min_confidence: float = 0.3,
        limit: int = 100,
    ) -> list[dict]:
        """从 Chroma 收集低 confidence 案例

        Args:
            line_id: 产线过滤（支持 list[str]）
            min_confidence: 置信度阈值，低于此值视为低质
            limit: 最大收集数

        Returns:
            低分案例列表
        """
        self.memory._ensure_chroma()
        if self.memory._collection is None:
            return []

        all_cases = self.memory.list_all_semantic(limit=limit * 3, line_id=line_id)
        low_conf = [
            c for c in all_cases
            if float(c.get("metadata", {}).get("confidence", 0.5)) < min_confidence
        ]
        return low_conf[:limit]

    def collect_negative_effect_trackings(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
    ) -> list[dict]:
        """收集反效果跟踪记录（improvement_pct < 0）

        Args:
            line_id: 产线过滤
            days: 近 N 天

        Returns:
            反效果跟踪列表
        """
        from agent.effect_tracker import EffectTracker
        tracker = EffectTracker(self.memory)
        records = tracker.list_trackings(line_id=line_id, status="tracked", days=days, limit=500)
        return [r for r in records if (r.get("improvement_pct") or 0) < 0]

    def collect_rejected_feedback(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
    ) -> list[dict]:
        """收集被拒绝的反馈（action=rejected）

        Args:
            line_id: 产线过滤
            days: 近 N 天

        Returns:
            被拒绝反馈列表
        """
        feedbacks = self.memory.query_feedback(line_id=line_id, days=days)
        return [fb for fb in feedbacks if fb.get("action") == "rejected"]

    def collect_all(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
        min_confidence: float = 0.3,
    ) -> dict:
        """综合收集三类失败案例，去重后存入 failure_cases 表

        幂等：同一 case_id+category 不会重复收集（检查 failure_cases 表）。

        Args:
            line_id: 产线过滤（None 则全部）
            days: 近 N 天（用于 negative_effect 和 rejected_feedback）
            min_confidence: 低分阈值

        Returns:
            {"low_confidence": N, "negative_effect": N, "rejected_feedback": N,
             "total_collected": N, "duplicates_skipped": N}
        """
        collected = 0
        duplicates = 0
        counts = {"low_confidence": 0, "negative_effect": 0, "rejected_feedback": 0}

        # 1. 低分案例
        low_conf_cases = self.collect_low_confidence_cases(
            line_id=line_id, min_confidence=min_confidence
        )
        for case in low_conf_cases:
            meta = case.get("metadata", {})
            case_id = case.get("id", "")
            if self._save_failure(
                case_id=case_id,
                tracking_id=None,
                line_id=meta.get("line_id", "heat_treatment"),
                category="low_confidence",
                confidence=meta.get("confidence", 0.5),
                improvement_pct=meta.get("last_effect_improvement"),
                root_cause=meta.get("root_cause", ""),
                solution=meta.get("solution", ""),
                failure_reason=f"confidence={meta.get('confidence', 0.5)} < {min_confidence}",
            ):
                collected += 1
                counts["low_confidence"] += 1
            else:
                duplicates += 1

        # 2. 反效果跟踪
        negative_trackings = self.collect_negative_effect_trackings(
            line_id=line_id, days=days
        )
        for rec in negative_trackings:
            if self._save_failure(
                case_id=rec.get("case_id", ""),
                tracking_id=rec.get("tracking_id", ""),
                line_id=rec.get("line_id", "heat_treatment"),
                category="negative_effect",
                confidence=None,
                improvement_pct=rec.get("improvement_pct"),
                root_cause="",
                solution="",
                failure_reason=f"improvement_pct={rec.get('improvement_pct')}% < 0",
            ):
                collected += 1
                counts["negative_effect"] += 1
            else:
                duplicates += 1

        # 3. 被拒绝反馈
        rejected = self.collect_rejected_feedback(line_id=line_id, days=days)
        for fb in rejected:
            if self._save_failure(
                case_id=fb.get("proposal_id", ""),
                tracking_id=None,
                line_id=fb.get("line_id", "heat_treatment"),
                category="rejected_feedback",
                confidence=None,
                improvement_pct=None,
                root_cause="",
                solution="",
                failure_reason=f"action=rejected, score={fb.get('score')}",
            ):
                collected += 1
                counts["rejected_feedback"] += 1
            else:
                duplicates += 1

        logger.info(
            f"M5-4 失败案例归集: {counts}, 共收集 {collected}，跳过重复 {duplicates}"
        )
        return {
            **counts,
            "total_collected": collected,
            "duplicates_skipped": duplicates,
        }

    def _save_failure(
        self,
        case_id: str,
        tracking_id: Optional[str],
        line_id: str,
        category: str,
        confidence: Optional[float] = None,
        improvement_pct: Optional[float] = None,
        root_cause: str = "",
        solution: str = "",
        failure_reason: str = "",
    ) -> bool:
        """保存失败案例（幂等：case_id+category 已存在则跳过）

        Returns:
            True=新建，False=已存在跳过
        """
        if not case_id and not tracking_id:
            return False

        # 幂等检查：同 case_id + category 已存在则跳过
        existing = self.db.execute(
            "SELECT failure_id FROM failure_cases WHERE case_id = ? AND category = ?",
            (case_id, category),
        ).fetchone()
        if existing:
            return False

        failure_id = f"fail_{uuid.uuid4().hex[:10]}"
        self.db.execute(
            """INSERT INTO failure_cases
            (failure_id, case_id, tracking_id, line_id, category,
             confidence, improvement_pct, root_cause, solution,
             failure_reason, collected_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (failure_id, case_id, tracking_id, line_id, category,
             confidence, improvement_pct, root_cause, solution,
             failure_reason, datetime.now()),
        )
        self.db.commit()
        return True

    # ===== 查询 =====

    def list_failures(
        self,
        line_id: Optional[str | list[str]] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> list[dict]:
        """列出失败案例

        Args:
            line_id: 产线过滤（支持 list[str]）
            category: 类别过滤（low_confidence/negative_effect/rejected_feedback）
            status: 状态过滤（open/analyzed/resolved）
            days: 近 N 天
            limit: 最大返回数
        """
        since = datetime.now() - timedelta(days=days)
        query = "SELECT * FROM failure_cases WHERE collected_at >= ?"
        params: list = [since]

        if line_id:
            if isinstance(line_id, (list, tuple)):
                placeholders = ",".join("?" for _ in line_id)
                query += f" AND line_id IN ({placeholders})"
                params.extend(line_id)
            else:
                query += " AND line_id = ?"
                params.append(line_id)

        if category:
            query += " AND category = ?"
            params.append(category)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY collected_at DESC LIMIT ?"
        params.append(limit)

        cur = self.db.execute(query, params)
        cols = [d[0] for d in cur.description]
        records = []
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            if rec.get("collected_at") and hasattr(rec["collected_at"], "isoformat"):
                rec["collected_at"] = rec["collected_at"].isoformat()
            records.append(rec)
        return records

    def get_failure(self, failure_id: str) -> Optional[dict]:
        """查询单条失败案例"""
        cur = self.db.execute(
            "SELECT * FROM failure_cases WHERE failure_id = ?",
            (failure_id,),
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            return None
        rec = dict(zip(cols, row))
        if rec.get("collected_at") and hasattr(rec["collected_at"], "isoformat"):
            rec["collected_at"] = rec["collected_at"].isoformat()
        return rec

    def get_failure_stats(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
    ) -> dict:
        """失败案例统计

        Returns:
            {"total": N, "by_category": {cat: count}, "by_status": {status: count}}
        """
        since = datetime.now() - timedelta(days=days)
        base = "FROM failure_cases WHERE collected_at >= ?"
        params = [since]

        if line_id:
            if isinstance(line_id, (list, tuple)):
                placeholders = ",".join("?" for _ in line_id)
                base += f" AND line_id IN ({placeholders})"
                params.extend(line_id)
            else:
                base += " AND line_id = ?"
                params.append(line_id)

        cur = self.db.execute(f"SELECT COUNT(*) {base}", params)
        total = cur.fetchone()[0]

        stats = {"total": total, "by_category": {}, "by_status": {}}
        if total == 0:
            return stats

        # 按类别
        cur = self.db.execute(
            f"SELECT category, COUNT(*) {base} GROUP BY category", params
        )
        for cat, cnt in cur.fetchall():
            stats["by_category"][cat] = cnt

        # 按状态
        cur = self.db.execute(
            f"SELECT status, COUNT(*) {base} GROUP BY status", params
        )
        for st, cnt in cur.fetchall():
            stats["by_status"][st] = cnt

        return stats

    # ===== 状态更新 =====

    def update_failure_status(
        self, failure_id: str, status: str, note: Optional[str] = None
    ) -> bool:
        """更新失败案例状态（open → analyzed → resolved）

        Args:
            failure_id: 失败案例ID
            status: 新状态（open/analyzed/resolved）
            note: 可选备注（追加到 failure_reason）

        Returns:
            是否成功
        """
        existing = self.get_failure(failure_id)
        if existing is None:
            return False

        if note:
            old_reason = existing.get("failure_reason", "") or ""
            new_reason = f"{old_reason}\n[{status}] {note}".strip()
            self.db.execute(
                "UPDATE failure_cases SET status = ?, failure_reason = ? WHERE failure_id = ?",
                (status, new_reason, failure_id),
            )
        else:
            self.db.execute(
                "UPDATE failure_cases SET status = ? WHERE failure_id = ?",
                (status, failure_id),
            )
        self.db.commit()
        logger.info(f"M5-4 失败案例 {failure_id} 状态更新: {status}")
        return True
