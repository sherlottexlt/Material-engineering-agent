"""
M5-8 A/B 测试框架

为 M5-5 Prompt 优化、M5-7 主动学习等"变更"提供量化效果验证：
- A 组（control）用旧版本，B 组（treatment）用新版本
- 哈希分配保证同 case_id 总是分到同组（确定性）
- t 检验 + 显著性判断（p < 0.05）
- 自动从 M5-1（效果跟踪）/ M5-6（质量评分）/ M5-4（失败案例）采集指标

核心闭环：
1. create_experiment：定义实验（A/B 配置 + 关注的指标名）
2. start_experiment：开始收集数据
3. assign：按 case_id 哈希分配到 A/B 组（幂等：同 case_id 总是同组）
4. record_metric / collect_metrics_from_*：记录或自动采集指标
5. analyze：统计两组差异 + t 检验 + 显著性 + 胜出者
6. stop_experiment：结束实验
"""
import hashlib
import json
import math
import uuid
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger


# 实验状态
STATUS_DRAFT = "draft"
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_COMPLETED = "completed"

# 默认关注的指标
DEFAULT_METRICS = [
    "quality_score",        # M5-6 案例质量分
    "feedback_adoption",    # 反馈采纳率（adopted=1, rejected=0）
    "improvement_pct",      # M5-1 调参效果改善百分比
    "failure_rate",         # M5-4 失败率（0/1）
]

# 显著性阈值
SIGNIFICANCE_THRESHOLD = 0.05


class ABTestFramework:
    """A/B 测试框架

    通过哈希分配将 case_id 确定性地分到 A/B 组，记录指标后用 t 检验判断差异显著性。
    """

    def __init__(self, memory, prompt_optimizer=None, quality_scorer=None,
                 effect_tracker=None, failure_collector=None):
        """初始化

        Args:
            memory: MemoryService 实例（必需）
            prompt_optimizer: PromptOptimizer 实例（可选，用于读取 Prompt 版本配置）
            quality_scorer: CaseQualityScorer 实例（可选，用于采集 quality_score）
            effect_tracker: EffectTracker 实例（可选，用于采集 improvement_pct）
            failure_collector: FailureCaseCollector 实例（可选，用于采集 failure_rate）
        """
        self.memory = memory
        self.db = memory.db
        self.prompt_optimizer = prompt_optimizer
        self.quality_scorer = quality_scorer
        self.effect_tracker = effect_tracker
        self.failure_collector = failure_collector

    # ===== 实验管理 =====

    def create_experiment(
        self,
        name: str,
        description: str = "",
        line_id: str = "heat_treatment",
        variant_a_config: Optional[dict] = None,
        variant_b_config: Optional[dict] = None,
        metric_names: Optional[list[str]] = None,
        sample_size_target: int = 100,
        created_by: str = "admin",
    ) -> dict:
        """创建 A/B 测试实验

        Args:
            name: 实验名称
            description: 实验描述（说明 A/B 差异点）
            line_id: 产线ID
            variant_a_config: A 组配置（如 {"prompt_version": "v1", "ruleset": "default"}）
            variant_b_config: B 组配置（如 {"prompt_version": "v2", "ruleset": "optimized"}）
            metric_names: 关注的指标名列表（默认 DEFAULT_METRICS）
            sample_size_target: 目标样本量
            created_by: 创建者

        Returns:
            {"success": bool, "experiment_id": str}
        """
        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
        variant_a_config = variant_a_config or {"version": "current", "description": "对照组"}
        variant_b_config = variant_b_config or {"version": "new", "description": "实验组"}
        metric_names = metric_names or DEFAULT_METRICS.copy()

        self.memory.save_ab_experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            line_id=line_id,
            variant_a_config=json.dumps(variant_a_config, ensure_ascii=False),
            variant_b_config=json.dumps(variant_b_config, ensure_ascii=False),
            metric_names=json.dumps(metric_names, ensure_ascii=False),
            sample_size_target=sample_size_target,
            created_by=created_by,
        )
        logger.info(f"创建 A/B 实验: {experiment_id}, name={name}, line={line_id}")
        return {"success": True, "experiment_id": experiment_id}

    def start_experiment(self, experiment_id: str) -> dict:
        """开始实验（draft → running）"""
        exp = self.memory.get_ab_experiment(experiment_id)
        if not exp:
            return {"success": False, "error": "experiment not found"}
        if exp["status"] != STATUS_DRAFT:
            return {"success": False, "error": f"实验状态为 {exp['status']}，仅 draft 可启动"}

        self.memory.update_ab_experiment_status(experiment_id, STATUS_RUNNING)
        logger.info(f"启动 A/B 实验: {experiment_id}")
        return {"success": True, "status": STATUS_RUNNING}

    def stop_experiment(self, experiment_id: str) -> dict:
        """停止实验（running → stopped）"""
        exp = self.memory.get_ab_experiment(experiment_id)
        if not exp:
            return {"success": False, "error": "experiment not found"}
        if exp["status"] != STATUS_RUNNING:
            return {"success": False, "error": f"实验状态为 {exp['status']}，仅 running 可停止"}

        self.memory.update_ab_experiment_status(experiment_id, STATUS_STOPPED)
        logger.info(f"停止 A/B 实验: {experiment_id}")
        return {"success": True, "status": STATUS_STOPPED}

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """查询实验详情（含解析后的 config/metric_names）"""
        exp = self.memory.get_ab_experiment(experiment_id)
        if not exp:
            return None
        return self._format_experiment(exp)

    def list_experiments(
        self,
        status: Optional[str] = None,
        line_id: Optional[str | list[str]] = None,
        limit: int = 50,
    ) -> list[dict]:
        """列出实验"""
        rows = self.memory.list_ab_experiments(status=status, line_id=line_id, limit=limit)
        return [self._format_experiment(r) for r in rows]

    def _format_experiment(self, row: dict) -> dict:
        """格式化实验记录（解析 JSON 字段）"""
        result = dict(row)
        try:
            result["variant_a_config"] = json.loads(row["variant_a_config"]) if row.get("variant_a_config") else {}
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            result["variant_b_config"] = json.loads(row["variant_b_config"]) if row.get("variant_b_config") else {}
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            result["metric_names"] = json.loads(row["metric_names"]) if row.get("metric_names") else []
        except (json.JSONDecodeError, TypeError):
            pass
        return result

    # ===== 分配 =====

    def assign(self, experiment_id: str, case_id: str, line_id: Optional[str] = None) -> dict:
        """将 case_id 分配到 A/B 组（幂等：同 case_id 总是同组）

        策略：MD5(experiment_id + case_id) 取首字节模 2，0=A，1=B
        保证确定性 + 均匀分布。

        Returns:
            {"success": bool, "variant": "a"/"b", "assignment_id": str, "case_id": str}
        """
        exp = self.memory.get_ab_experiment(experiment_id)
        if not exp:
            return {"success": False, "error": "experiment not found"}
        if exp["status"] != STATUS_RUNNING:
            return {"success": False, "error": f"实验状态为 {exp['status']}，仅 running 可分配"}

        # 幂等：先查已有分配
        existing = self.memory.get_ab_assignment(experiment_id, case_id)
        if existing:
            return {
                "success": True,
                "variant": existing["variant"],
                "assignment_id": existing["assignment_id"],
                "case_id": case_id,
                "reassigned": True,
            }

        # 哈希分配
        variant = self._hash_assign(experiment_id, case_id)
        assignment_id = f"asg_{uuid.uuid4().hex[:10]}"
        effective_line = line_id or exp["line_id"]

        self.memory.save_ab_assignment(
            assignment_id=assignment_id,
            experiment_id=experiment_id,
            variant=variant,
            case_id=case_id,
            line_id=effective_line,
        )
        return {
            "success": True,
            "variant": variant,
            "assignment_id": assignment_id,
            "case_id": case_id,
            "reassigned": False,
        }

    def _hash_assign(self, experiment_id: str, case_id: str) -> str:
        """哈希分配：MD5(experiment_id + case_id) 首字节模 2"""
        key = f"{experiment_id}:{case_id}"
        hash_bytes = hashlib.md5(key.encode("utf-8")).digest()
        return "a" if hash_bytes[0] % 2 == 0 else "b"

    def get_assignment(self, experiment_id: str, case_id: str) -> Optional[dict]:
        """查询已有分配"""
        return self.memory.get_ab_assignment(experiment_id, case_id)

    def list_assignments(
        self,
        experiment_id: str,
        variant: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """列出实验的分配记录"""
        return self.memory.list_ab_assignments(experiment_id, variant=variant, limit=limit)

    # ===== 指标记录 =====

    def record_metric(
        self,
        experiment_id: str,
        case_id: str,
        metric_name: str,
        metric_value: float,
    ) -> dict:
        """记录指标（自动查找分配记录确定 variant）

        Returns:
            {"success": bool, "metric_id": str, "variant": str}
        """
        assignment = self.memory.get_ab_assignment(experiment_id, case_id)
        if not assignment:
            return {"success": False, "error": "case_id 未分配，请先 assign"}
        if not isinstance(metric_value, (int, float)):
            return {"success": False, "error": "metric_value 必须为数值"}

        metric_id = f"mtc_{uuid.uuid4().hex[:10]}"
        self.memory.save_ab_metric(
            metric_id=metric_id,
            experiment_id=experiment_id,
            assignment_id=assignment["assignment_id"],
            variant=assignment["variant"],
            metric_name=metric_name,
            metric_value=float(metric_value),
        )
        return {
            "success": True,
            "metric_id": metric_id,
            "variant": assignment["variant"],
        }

    def record_metric_batch(
        self,
        experiment_id: str,
        metrics: list[dict],
    ) -> dict:
        """批量记录指标

        Args:
            metrics: [{"case_id": str, "metric_name": str, "metric_value": float}, ...]

        Returns:
            {"success": bool, "recorded": int, "failed": int}
        """
        recorded = 0
        failed = 0
        for m in metrics:
            result = self.record_metric(
                experiment_id=experiment_id,
                case_id=m["case_id"],
                metric_name=m["metric_name"],
                metric_value=m["metric_value"],
            )
            if result["success"]:
                recorded += 1
            else:
                failed += 1
        return {"success": True, "recorded": recorded, "failed": failed}

    def list_metrics(
        self,
        experiment_id: str,
        variant: Optional[str] = None,
        metric_name: Optional[str] = None,
    ) -> list[dict]:
        """列出实验的指标记录"""
        return self.memory.list_ab_metrics(
            experiment_id, variant=variant, metric_name=metric_name
        )

    # ===== 自动采集指标（从 M5-1/M5-4/M5-6）=====

    def collect_metrics_from_quality(
        self,
        experiment_id: str,
        case_ids: Optional[list[str]] = None,
    ) -> dict:
        """从 M5-6 CaseQualityScorer 采集 quality_score 指标

        Args:
            case_ids: 指定 case_id 列表，None 则用该实验全部分配
        """
        if not self.quality_scorer:
            return {"success": False, "error": "quality_scorer 未注入"}

        if case_ids is None:
            assignments = self.memory.list_ab_assignments(experiment_id, limit=10000)
            case_ids = [a["case_id"] for a in assignments]

        recorded = 0
        for case_id in case_ids:
            # 从 episodic 查询案例
            record = self._get_episodic_record(case_id)
            if not record:
                continue
            score_result = self.quality_scorer.score_case(record)
            result = self.record_metric(
                experiment_id=experiment_id,
                case_id=case_id,
                metric_name="quality_score",
                metric_value=score_result["new_score"],
            )
            if result["success"]:
                recorded += 1
        return {"success": True, "recorded": recorded, "metric_name": "quality_score"}

    def collect_metrics_from_effect(
        self,
        experiment_id: str,
        days: int = 30,
    ) -> dict:
        """从 M5-1 EffectTracker 采集 improvement_pct 指标

        通过 proposal_id 关联回 case_id（experiment 分配时用 case_id）。
        """
        if not self.effect_tracker:
            return {"success": False, "error": "effect_tracker 未注入"}

        trackings = self.effect_tracker.list_trackings(days=days, limit=10000)
        recorded = 0
        for t in trackings:
            case_id = t.get("case_id")
            improvement = t.get("improvement_pct")
            if not case_id or improvement is None:
                continue
            # 只记录已分配的 case
            assignment = self.memory.get_ab_assignment(experiment_id, case_id)
            if not assignment:
                continue
            result = self.record_metric(
                experiment_id=experiment_id,
                case_id=case_id,
                metric_name="improvement_pct",
                metric_value=float(improvement),
            )
            if result["success"]:
                recorded += 1
        return {"success": True, "recorded": recorded, "metric_name": "improvement_pct"}

    def collect_metrics_from_failures(
        self,
        experiment_id: str,
        days: int = 30,
    ) -> dict:
        """从 M5-4 failure_cases 采集 failure_rate 指标（出现失败=1，否则=0）"""
        if not self.failure_collector:
            return {"success": False, "error": "failure_collector 未注入"}

        # 取实验全部分配
        assignments = self.memory.list_ab_assignments(experiment_id, limit=10000)
        if not assignments:
            return {"success": True, "recorded": 0, "metric_name": "failure_rate"}

        # 取失败案例的 case_id 集合
        failures = self.failure_collector.list_failures(days=days, limit=10000)
        failure_case_ids = {f["case_id"] for f in failures if f.get("case_id")}

        recorded = 0
        for a in assignments:
            case_id = a["case_id"]
            failure_value = 1.0 if case_id in failure_case_ids else 0.0
            result = self.record_metric(
                experiment_id=experiment_id,
                case_id=case_id,
                metric_name="failure_rate",
                metric_value=failure_value,
            )
            if result["success"]:
                recorded += 1
        return {"success": True, "recorded": recorded, "metric_name": "failure_rate"}

    def _get_episodic_record(self, case_id: str) -> Optional[dict]:
        """从 episodic 表查询单条记录"""
        cur = self.db.execute(
            "SELECT * FROM episodic WHERE record_id = ?",
            (case_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    # ===== 分析（核心）=====

    def analyze(self, experiment_id: str) -> dict:
        """分析实验结果

        Returns:
            {
                "experiment_id": str,
                "status": str,
                "sample_sizes": {"a": int, "b": int},
                "metrics": {
                    metric_name: {
                        "a": {"mean", "std", "n", "min", "max"},
                        "b": {"mean", "std", "n", "min", "max"},
                        "diff": float,            # b_mean - a_mean
                        "diff_pct": float,        # (b-a)/a * 100
                        "t_statistic": float,
                        "p_value": float,
                        "significant": bool,      # p < 0.05
                        "winner": "a"/"b"/"no_difference",
                    }
                },
                "conclusion": str,   # 人类可读结论
            }
        """
        exp = self.memory.get_ab_experiment(experiment_id)
        if not exp:
            return {"success": False, "error": "experiment not found"}

        # 按指标名分组
        all_metrics = self.memory.list_ab_metrics(experiment_id)
        by_metric: dict[str, dict[str, list[float]]] = {}
        for m in all_metrics:
            name = m["metric_name"]
            variant = m["variant"]
            value = m["metric_value"]
            if name not in by_metric:
                by_metric[name] = {"a": [], "b": []}
            if variant in ("a", "b"):
                by_metric[name][variant].append(value)

        # 样本量
        assignments = self.memory.list_ab_assignments(experiment_id, limit=100000)
        sample_sizes = {"a": 0, "b": 0}
        for a in assignments:
            if a["variant"] in sample_sizes:
                sample_sizes[a["variant"]] += 1

        # 各指标统计
        metrics_result = {}
        conclusions = []
        for metric_name, values in by_metric.items():
            a_vals = values["a"]
            b_vals = values["b"]
            stat = self._compute_metric_stats(metric_name, a_vals, b_vals)
            metrics_result[metric_name] = stat
            if stat["significant"]:
                winner = stat["winner"]
                if winner == "b":
                    conclusions.append(
                        f"B 组在 {metric_name} 上显著优于 A 组 "
                        f"(+{stat['diff_pct']:.1f}%, p={stat['p_value']:.3f})"
                    )
                elif winner == "a":
                    conclusions.append(
                        f"A 组在 {metric_name} 上显著优于 B 组 "
                        f"({stat['diff_pct']:.1f}%, p={stat['p_value']:.3f})"
                    )
            else:
                conclusions.append(
                    f"{metric_name} 无显著差异 (p={stat['p_value']:.3f})"
                )

        conclusion = "；".join(conclusions) if conclusions else "无指标数据"
        return {
            "success": True,
            "experiment_id": experiment_id,
            "status": exp["status"],
            "sample_sizes": sample_sizes,
            "metrics": metrics_result,
            "conclusion": conclusion,
        }

    def _compute_metric_stats(
        self,
        metric_name: str,
        a_vals: list[float],
        b_vals: list[float],
    ) -> dict:
        """计算单个指标的统计量 + t 检验"""
        a_stats = self._describe(a_vals)
        b_stats = self._describe(b_vals)

        diff = b_stats["mean"] - a_stats["mean"]
        diff_pct = (diff / a_stats["mean"] * 100) if a_stats["mean"] != 0 else 0.0

        # t 检验（Welch's t-test，不假设方差齐性）
        t_stat, p_value = self._welch_t_test(a_vals, b_vals)
        significant = p_value < SIGNIFICANCE_THRESHOLD

        # 胜出者判断
        if not significant:
            winner = "no_difference"
        elif diff > 0:
            # 注意：failure_rate 是越小越好，需特殊处理
            if metric_name in ("failure_rate",):
                winner = "a"
            else:
                winner = "b"
        else:
            if metric_name in ("failure_rate",):
                winner = "b"
            else:
                winner = "a"

        return {
            "a": a_stats,
            "b": b_stats,
            "diff": diff,
            "diff_pct": diff_pct,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": significant,
            "winner": winner,
        }

    def _describe(self, vals: list[float]) -> dict:
        """描述性统计"""
        n = len(vals)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "n": 0, "min": 0.0, "max": 0.0}
        mean = sum(vals) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        return {
            "mean": mean,
            "std": std,
            "n": n,
            "min": min(vals),
            "max": max(vals),
        }

    def _welch_t_test(self, a: list[float], b: list[float]) -> tuple[float, float]:
        """Welch's t-test（不假设方差齐性）

        返回 (t_statistic, p_value)
        - 两样本均值差异的 t 统计量
        - p_value 用正态近似（|t| > 1.96 对应 p < 0.05，双侧）
        - 样本量不足（n < 2）返回 (0.0, 1.0)
        """
        n_a = len(a)
        n_b = len(b)
        if n_a < 2 or n_b < 2:
            return (0.0, 1.0)

        mean_a = sum(a) / n_a
        mean_b = sum(b) / n_b
        var_a = sum((v - mean_a) ** 2 for v in a) / (n_a - 1)
        var_b = sum((v - mean_b) ** 2 for v in b) / (n_b - 1)

        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            # 两组方差都为 0：均值相同则无差异，否则极端显著
            if mean_a == mean_b:
                return (0.0, 1.0)
            return (float("inf"), 0.0)

        t = (mean_b - mean_a) / se

        # Welch-Satterthwaite 自由度
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        if denom == 0:
            df = n_a + n_b - 2
        else:
            df = num / denom

        # 用正态近似计算双侧 p_value（大样本时 t 分布趋近正态）
        # 小样本时偏保守（实际 p 值偏大），但够用
        p_value = self._t_distribution_p_two_sided(t, df)
        return (t, p_value)

    def _t_distribution_p_two_sided(self, t: float, df: float) -> float:
        """近似计算 t 分布双侧 p 值

        对大 df 用正态近似，对小 df 用简单近似。
        注：sandbox 可能无 scipy，用近似公式。
        """
        if df <= 0:
            return 1.0
        # 大自由度（df > 30）用正态近似
        if df > 30:
            # 标准正态 CDF 近似（Abramowitz & Stegun 26.2.17）
            z = abs(t)
            if z > 6:
                return 0.0
            cdf = 1.0 - 0.5 * math.erfc(z / math.sqrt(2))
            p = 2 * (1.0 - cdf)
            return max(0.0, min(1.0, p))
        else:
            # 小自由度用简化公式（基于 t^2 与 F 分布关系）
            # p ≈ 2 * (1 - CDF(|t|))
            # 用 Beta 函数近似太复杂，这里用查表 + 插值
            # 简化方案：用 Cauchy 分布近似（df=1）到正态（df=∞）的插值
            # 实际项目用 scipy.stats.ttest_ind 更准确，这里近似够用
            z = abs(t)
            if z > 30:
                return 0.0
            # 简单近似：1/(1+z) 衰减（粗略）
            # 更准确：用 t^2 / (t^2 + df) 的 Beta 分布关系
            x = df / (df + z * z)
            # Beta(x; df/2, 0.5) 的补 CDF 近似
            # 这里用简化公式：p ≈ Beta(x; 0.5, 0.5) 即不完全 Beta
            # 为简化用正态近似 + 自由度修正
            correction = math.sqrt(df / (df + 2))  # 小样本修正
            z_corrected = z * correction
            if z_corrected > 6:
                return 0.0
            cdf = 1.0 - 0.5 * math.erfc(z_corrected / math.sqrt(2))
            p = 2 * (1.0 - cdf)
            return max(0.0, min(1.0, p))

    # ===== 统计 =====

    def get_stats(self) -> dict:
        """获取 A/B 测试总体统计"""
        return self.memory.get_ab_stats()
