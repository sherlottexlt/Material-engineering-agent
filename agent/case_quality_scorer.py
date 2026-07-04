"""M5-6 案例质量评分模块

自动评估 episodic 记忆的质量，让 quality_score 字段真正反映案例价值，
为 M5-7 知识主动学习提供质量筛选依据，并让 M3-13 cleanup 阈值真正生效。

评分维度（4 维加权）：
- 信息完整度 completeness (40%): root_cause/solution 是否非空 + 长度足够
- 可复用性   reusability   (25%): defect_type 标准化 + solution 可操作
- 时效性     timeliness    (15%): 案例新鲜度（越新分越高）
- 验证状态   validation    (20%): failure_cases 出现过则减分（说明案例失效）

公式：quality_score = 0.4 * completeness + 0.25 * reusability
                    + 0.15 * timeliness + 0.2 * validation
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger


# 评分维度权重
WEIGHT_COMPLETENESS = 0.40
WEIGHT_REUSABILITY = 0.25
WEIGHT_TIMELINESS = 0.15
WEIGHT_VALIDATION = 0.20

# 完整度阈值
MIN_ROOT_CAUSE_LEN = 20
MIN_SOLUTION_LEN = 20

# 时效性衰减天数（1 年衰减到 0）
TIMELINESS_DECAY_DAYS = 365

# 验证状态：出现失败记录的减分
VALIDATION_FAILURE_PENALTY = 0.5  # 出现失败 → 0.5 - 0.5 = 0
VALIDATION_BASELINE = 0.5         # 无失败记录 → 基线 0.5
VALIDATION_BONUS = 0.5            # （预留）被采纳反馈加分

# 标准缺陷类型关键词（用于可复用性判断）
STANDARD_DEFECT_KEYWORDS = (
    "hardness", "crack", "deformation", "oxidation", "decarburization",
    "porosity", "inclusion", "warpage", "wear", "corrosion",
    "硬度", "裂纹", "变形", "氧化", "脱碳", "气孔", "夹杂", "磨损", "腐蚀",
)

# 可操作关键词（solution 含这些词说明可执行）
ACTIONABLE_KEYWORDS = (
    "调整", "更换", "检查", "提升", "降低", "增加", "减少", "更换",
    "保温", "冷却", "淬火", "回火", "正火", "退火", "渗碳",
    "adjust", "replace", "check", "increase", "decrease",
    "temperature", "time", "rate", "≥", "≤", "±",
)


class CaseQualityScorer:
    """案例质量评分器

    评分流程：
    1. 从 episodic 表读取案例
    2. 对每条案例计算 4 维子分
    3. 加权求和得到 quality_score（0.0-1.0）
    4. 批量更新回 episodic.quality_score 字段

    用法：
        scorer = CaseQualityScorer(memory)
        result = scorer.score_all(line_id="heat_treatment", days=365)
        # result = {"total": 100, "updated": 95, "skipped": 5, "avg_score": 0.72, ...}
    """

    def __init__(self, memory):
        self.memory = memory
        self.db = memory.db

    # ===== 单案例评分 =====

    def score_case(self, record: dict) -> dict:
        """评估单个案例质量

        Args:
            record: episodic 记录 dict，需含字段
                record_id, batch_id, defect_type, root_cause, solution,
                created_at, quality_score, line_id

        Returns:
            {
                "record_id": str,
                "old_score": float,
                "new_score": float,
                "dimensions": {
                    "completeness": float,
                    "reusability": float,
                    "timeliness": float,
                    "validation": float,
                },
                "reasons": list[str],  # 评分依据说明
            }
        """
        reasons: list[str] = []

        completeness = self._score_completeness(record, reasons)
        reusability = self._score_reusability(record, reasons)
        timeliness = self._score_timeliness(record, reasons)
        validation = self._score_validation(record, reasons)

        new_score = (
            WEIGHT_COMPLETENESS * completeness
            + WEIGHT_REUSABILITY * reusability
            + WEIGHT_TIMELINESS * timeliness
            + WEIGHT_VALIDATION * validation
        )
        new_score = round(max(0.0, min(1.0, new_score)), 4)

        return {
            "record_id": record.get("record_id", ""),
            "old_score": float(record.get("quality_score", 0.5) or 0.5),
            "new_score": new_score,
            "dimensions": {
                "completeness": round(completeness, 4),
                "reusability": round(reusability, 4),
                "timeliness": round(timeliness, 4),
                "validation": round(validation, 4),
            },
            "reasons": reasons,
        }

    # ===== 维度子分计算 =====

    def _score_completeness(self, record: dict, reasons: list[str]) -> float:
        """信息完整度（0-1）

        - root_cause 非空: +0.5
        - solution 非空: +0.5
        - root_cause 长度 >= 20: +0.3
        - solution 长度 >= 20: +0.3
        归一化到 [0, 1]（最大 1.6 → 1.0）
        """
        root_cause = (record.get("root_cause") or "").strip()
        solution = (record.get("solution") or "").strip()

        score = 0.0
        if root_cause:
            score += 0.5
            if len(root_cause) >= MIN_ROOT_CAUSE_LEN:
                score += 0.3
                reasons.append(f"root_cause 完整（{len(root_cause)} 字符）")
            else:
                reasons.append(f"root_cause 偏短（{len(root_cause)} 字符）")
        else:
            reasons.append("root_cause 为空")

        if solution:
            score += 0.5
            if len(solution) >= MIN_SOLUTION_LEN:
                score += 0.3
                reasons.append(f"solution 完整（{len(solution)} 字符）")
            else:
                reasons.append(f"solution 偏短（{len(solution)} 字符）")
        else:
            reasons.append("solution 为空")

        return min(1.0, score / 1.6)

    def _score_reusability(self, record: dict, reasons: list[str]) -> float:
        """可复用性（0-1）

        - defect_type 非空: +0.4
        - defect_type 含标准缺陷关键词: +0.3
        - solution 含可操作关键词: +0.3
        归一化到 [0, 1]（最大 1.0）
        """
        defect_type = (record.get("defect_type") or "").strip()
        solution = (record.get("solution") or "").strip()

        score = 0.0
        if defect_type:
            score += 0.4
            defect_lower = defect_type.lower()
            if any(kw.lower() in defect_lower for kw in STANDARD_DEFECT_KEYWORDS):
                score += 0.3
                reasons.append(f"defect_type 标准化（{defect_type}）")
            else:
                reasons.append(f"defect_type 非标准（{defect_type}）")
        else:
            reasons.append("defect_type 为空")

        if solution and any(kw in solution for kw in ACTIONABLE_KEYWORDS):
            score += 0.3
            reasons.append("solution 含可操作关键词")
        elif solution:
            reasons.append("solution 缺可操作关键词")

        return min(1.0, score)

    def _score_timeliness(self, record: dict, reasons: list[str]) -> float:
        """时效性（0-1）

        案例年龄 days = (now - created_at).days
        衰减分数：max(0, 1 - days / 365)
        """
        created_at = record.get("created_at")
        if not created_at:
            reasons.append("created_at 为空，时效性记 0")
            return 0.0

        # 兼容字符串和 datetime 对象
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                try:
                    created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    reasons.append(f"created_at 格式无法解析: {created_at}")
                    return 0.0

        now = datetime.now()
        if isinstance(created_at, datetime):
            age_days = (now - created_at).days
        else:
            reasons.append(f"created_at 类型异常: {type(created_at)}")
            return 0.0

        if age_days < 0:
            age_days = 0  # 未来时间视为最新

        score = max(0.0, 1.0 - age_days / TIMELINESS_DECAY_DAYS)
        reasons.append(f"案例年龄 {age_days} 天，时效性 {score:.2f}")
        return score

    def _score_validation(self, record: dict, reasons: list[str]) -> float:
        """验证状态（0-1）

        - 在 failure_cases 表出现（有失败记录）: 0.0（案例失效）
        - 未出现: 0.5（基线）
        - （预留）被采纳反馈: +0.5 → 1.0
        """
        record_id = record.get("record_id", "")
        if not record_id:
            reasons.append("record_id 为空，验证状态记基线")
            return VALIDATION_BASELINE

        # 查询 failure_cases 表是否有关联记录
        has_failure = self._has_failure_record(record_id)
        if has_failure:
            reasons.append("案例在 failure_cases 表有失败记录，验证状态降为 0")
            return max(0.0, VALIDATION_BASELINE - VALIDATION_FAILURE_PENALTY)

        reasons.append("案例无失败记录，验证状态基线 0.5")
        return VALIDATION_BASELINE

    def _has_failure_record(self, record_id: str) -> bool:
        """检查 record_id 是否在 failure_cases 表中出现"""
        try:
            cur = self.db.execute(
                "SELECT 1 FROM failure_cases WHERE case_id = ? LIMIT 1",
                (record_id,),
            )
            return cur.fetchone() is not None
        except Exception as e:
            logger.warning(f"查询 failure_cases 失败: {e}")
            return False

    # ===== 批量评分 =====

    def score_all(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 365,
        limit: int = 1000,
        dry_run: bool = False,
    ) -> dict:
        """批量评分并更新 quality_score 字段

        Args:
            line_id: 产线ID过滤（可选，支持 str 或 list[str]）
            days: 评分范围（最近 N 天的案例）
            limit: 最大处理数量
            dry_run: 仅评分不写入数据库

        Returns:
            {
                "total": int,         # 处理的案例总数
                "updated": int,       # 实际更新的数量
                "skipped": int,       # 跳过的数量（dry_run 或 写入失败）
                "avg_score": float,   # 平均新评分
                "by_tier": {"high": int, "medium": int, "low": int},
                "sample": list[dict], # 前 5 条评分结果示例
            }
        """
        records = self.memory.query_episodic(days=days, line_id=line_id)
        # query_episodic 已按 created_at DESC 排序，再按 limit 截断
        records = records[:limit]

        results = []
        for rec in records:
            try:
                result = self.score_case(rec)
                results.append((rec, result))
            except Exception as e:
                logger.warning(f"评分失败 record_id={rec.get('record_id')}: {e}")

        updated = 0
        scores: list[float] = []
        for rec, result in results:
            new_score = result["new_score"]
            scores.append(new_score)
            if not dry_run:
                ok = self.memory.update_quality_score(
                    result["record_id"], new_score
                )
                if ok:
                    updated += 1

        by_tier = {"high": 0, "medium": 0, "low": 0}
        for s in scores:
            if s >= 0.7:
                by_tier["high"] += 1
            elif s >= 0.4:
                by_tier["medium"] += 1
            else:
                by_tier["low"] += 1

        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "total": len(results),
            "updated": updated,
            "skipped": len(results) - updated,
            "avg_score": avg_score,
            "by_tier": by_tier,
            "sample": [r for _, r in results[:5]],
        }

    # ===== 查询代理（方便 API 调用）=====

    def get_low_quality_cases(
        self,
        line_id: Optional[str | list[str]] = None,
        threshold: float = 0.4,
        limit: int = 100,
    ) -> list[dict]:
        """获取低质量案例列表（代理到 memory.get_low_quality_cases）"""
        return self.memory.get_low_quality_cases(
            line_id=line_id, threshold=threshold, limit=limit
        )

    def get_quality_stats(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
    ) -> dict:
        """质量分布统计（代理到 memory.get_quality_distribution）"""
        return self.memory.get_quality_distribution(line_id=line_id, days=days)
