"""
MetaCraft Agent Prompt 自动优化（M5-5）

基于 M5-4 失败案例库自动优化 Prompt，形成"失败案例 → Prompt 优化 → 案例质量提升"闭环。

核心流程：
1. analyze_failure_patterns(): 从 failure_cases 表分析失败模式，映射到需优化的 Prompt 角色
2. generate_optimization(): 基于失败模式生成 Prompt 改进规则（规则优先，可选 LLM 辅助）
3. apply_optimization(): 备份当前 prompts.yaml → 应用新 prompt → 记录版本历史
4. rollback_optimization(): 从历史快照恢复 → 标记 status=rolled_back
5. list_optimizations() / get_optimization(): 查询优化历史
6. get_current_prompts(): 查看当前 prompts

失败类别 → 优化角色映射：
- low_confidence    → decision_agent（决策不准）+ review_agent（审核标准）
- rejected_feedback → decision_agent（方案被拒）
- negative_effect   → decision_agent（方案反效果）

幂等性：
- 同 role+failure_category 已有 applied 记录时，generate 不重复生成（返回已存在记录）
- apply 只能作用于 status=draft 的记录
- rollback 只能作用于 status=applied 的记录
"""
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from agent.memory.memory_service import MemoryService


# ===== 角色 → prompts.yaml key 映射 =====
ROLE_TO_PROMPT_KEY = {
    "planner": "planner",
    "data": "data_agent",
    "mechanism": "mechanism_agent",
    "knowledge": "knowledge_agent",
    "decision": "decision_agent",
    "review": "review_agent",
    "interaction": "interaction_agent",
    "reflector": "reflector",
}

# ===== 失败类别 → 优化角色 + 规则模板 =====
# 每种失败类别对应需优化的角色列表，及优化规则模板
FAILURE_CATEGORY_OPTIMIZATION = {
    "low_confidence": {
        "roles": ["decision", "review"],
        "rules": {
            "decision": (
                "\n\n【M5-5 自动优化规则（基于 {failure_count} 条 low_confidence 失败案例）】\n"
                "1. 每个方案的 evidence 必须包含至少 2 条具体数值比对\n"
                "2. confidence 取值需基于证据支持度：\n"
                "   - 证据充分（≥3 条数值比对）→ 0.8-0.95\n"
                "   - 证据一般（1-2 条数值比对）→ 0.6-0.8\n"
                "   - 证据不足（无数值比对）→ 0.4-0.6（禁止 > 0.6）\n"
                "3. 根因判定必须严格按决策树顺序，禁止跳步\n"
                "4. 失败案例警示：{sample_failure_reasons}\n"
            ),
            "review": (
                "\n\n【M5-5 自动优化规则（基于 {failure_count} 条 low_confidence 失败案例）】\n"
                "1. 审核时优先检查 confidence 是否与证据支持度匹配\n"
                "2. confidence > 0.8 但 evidence 不足 3 条数值比对 → 拒绝（理由：置信度虚高）\n"
                "3. confidence < 0.5 但 evidence 充分 → 拒绝（理由：置信度虚低）\n"
                "4. 失败案例警示：{sample_failure_reasons}\n"
            ),
        },
    },
    "rejected_feedback": {
        "roles": ["decision"],
        "rules": {
            "decision": (
                "\n\n【M5-5 自动优化规则（基于 {failure_count} 条 rejected_feedback 失败案例）】\n"
                "1. 方案必须可执行：调整量不得超过标准值的 ±15%\n"
                "2. 避免给出过于激进的调整（如温度 +50℃），优先小幅渐进调整\n"
                "3. 每个方案必须标注 risks，且 risks 不得为空\n"
                "4. 失败案例警示：{sample_failure_reasons}\n"
            ),
        },
    },
    "negative_effect": {
        "roles": ["decision"],
        "rules": {
            "decision": (
                "\n\n【M5-5 自动优化规则（基于 {failure_count} 条 negative_effect 失败案例）】\n"
                "1. 调整量不得超过标准值的 ±10%，避免过度调整导致反效果\n"
                "2. 对于多参数偏离的批次，优先调整偏离幅度最大的单一参数，避免多参数同时调整\n"
                "3. expected_effect 必须量化（如 硬度提升 2-3 HRc），不得用模糊描述\n"
                "4. 失败案例警示（反效果记录）：{sample_failure_reasons}\n"
            ),
        },
    },
}

# 自动优化规则区块的标记（用于替换已存在的规则，保证幂等）
OPTIMIZATION_RULE_MARKER = "【M5-5 自动优化规则"


class PromptOptimizer:
    """Prompt 自动优化器（M5-5）

    用法：
        optimizer = PromptOptimizer(memory)
        # 分析失败模式
        patterns = optimizer.analyze_failure_patterns(line_id="heat_treatment")
        # 生成优化（status=draft）
        for pattern in patterns:
            opt = optimizer.generate_optimization(pattern)
        # 应用优化
        optimizer.apply_optimization(opt["optimization_id"])
        # 回滚
        optimizer.rollback_optimization(opt["optimization_id"])
    """

    def __init__(
        self,
        memory: MemoryService,
        prompts_path: Optional[Path] = None,
        history_dir: Optional[Path] = None,
    ):
        self.memory = memory
        self.db = memory.db
        # prompts.yaml 路径（默认 config/prompts.yaml）
        self.prompts_path = prompts_path or Path(__file__).parent.parent / "config" / "prompts.yaml"
        # 历史快照目录
        self.history_dir = history_dir or self.prompts_path.parent / "prompts_history"

    # ===== 1. 失败模式分析 =====

    def analyze_failure_patterns(
        self,
        line_id: Optional[str] = None,
        days: int = 30,
    ) -> list[dict]:
        """从 failure_cases 表分析失败模式，映射到需优化的 Prompt 角色

        Args:
            line_id: 产线过滤（None 则全部）
            days: 近 N 天

        Returns:
            失败模式列表，每项：
            {
                "role": "decision",  # 需优化的角色
                "failure_category": "low_confidence",
                "failure_count": 5,
                "trigger_reason": "决策不准：5 条 low_confidence 案例",
                "sample_failures": [...],  # 最多 3 条样本
            }
        """
        # 查询 failure_cases 按类别统计
        from agent.failure_case_collector import FailureCaseCollector
        collector = FailureCaseCollector(self.memory)
        stats = collector.get_failure_stats(line_id=line_id, days=days)
        by_category = stats.get("by_category", {})

        if not by_category:
            return []

        patterns = []
        for category, count in by_category.items():
            if count == 0:
                continue
            opt_config = FAILURE_CATEGORY_OPTIMIZATION.get(category)
            if not opt_config:
                continue

            # 取该类别的失败样本（最多 3 条，用于规则填充）
            failures = collector.list_failures(
                line_id=line_id, category=category, days=days, limit=3
            )
            sample_reasons = [f.get("failure_reason", "") for f in failures if f.get("failure_reason")]
            sample_text = " | ".join(sample_reasons[:3]) if sample_reasons else "无具体失败原因"

            # 映射到需优化的角色
            for role in opt_config["roles"]:
                patterns.append({
                    "role": role,
                    "failure_category": category,
                    "failure_count": count,
                    "trigger_reason": f"{opt_config.get('trigger_desc', category)}：{count} 条 {category} 案例",
                    "sample_failures": failures,
                    "sample_failure_reasons": sample_text,
                })

        logger.info(f"M5-5 失败模式分析: 识别 {len(patterns)} 个优化点")
        return patterns

    # ===== 2. 生成优化 =====

    def generate_optimization(
        self,
        pattern: dict,
        force: bool = False,
    ) -> Optional[dict]:
        """基于失败模式生成 Prompt 优化（status=draft）

        幂等：同 role+failure_category 已有 applied 记录时，不重复生成（除非 force=True）

        Args:
            pattern: analyze_failure_patterns 返回的失败模式
            force: 强制生成（忽略幂等检查）

        Returns:
            优化记录 dict，或 None（幂等跳过时）
        """
        role = pattern["role"]
        category = pattern["failure_category"]

        # 幂等检查：同 role+category 已有 applied 记录则跳过
        if not force:
            existing = self.db.execute(
                "SELECT optimization_id, version FROM prompt_optimizations "
                "WHERE role = ? AND failure_category = ? AND status = 'applied' "
                "ORDER BY version DESC LIMIT 1",
                (role, category),
            ).fetchone()
            if existing:
                logger.info(
                    f"M5-5 跳过优化（已存在 applied 记录）: role={role}, category={category}, "
                    f"optimization_id={existing[0]}"
                )
                return self.get_optimization(existing[0])

        # 获取当前 prompt
        current_prompts = self._load_prompts()
        prompt_key = ROLE_TO_PROMPT_KEY.get(role)
        if not prompt_key or prompt_key not in current_prompts:
            logger.warning(f"M5-5 角色 {role} 无对应 prompt key，跳过")
            return None

        old_prompt = current_prompts[prompt_key]

        # 生成新 prompt（规则优先）
        opt_config = FAILURE_CATEGORY_OPTIMIZATION.get(category, {})
        rule_template = opt_config.get("rules", {}).get(role)
        if not rule_template:
            logger.warning(f"M5-5 角色 {role} + 类别 {category} 无规则模板，跳过")
            return None

        rule_text = rule_template.format(
            failure_count=pattern["failure_count"],
            sample_failure_reasons=pattern.get("sample_failure_reasons", "无"),
        )

        new_prompt = self._merge_optimization_rule(old_prompt, rule_text)
        change_summary = self._summarize_change(old_prompt, new_prompt, pattern)

        # 生成版本号（当前最大版本 + 1）
        version = self._next_version()

        optimization_id = f"opt_{uuid.uuid4().hex[:10]}"
        self.db.execute(
            """INSERT INTO prompt_optimizations
            (optimization_id, version, role, failure_category, failure_count,
             trigger_reason, old_prompt, new_prompt, change_summary,
             status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)""",
            (optimization_id, version, role, category,
             pattern["failure_count"], pattern["trigger_reason"],
             old_prompt, new_prompt, change_summary, datetime.now()),
        )
        self.db.commit()

        logger.info(
            f"M5-5 生成优化: id={optimization_id}, version={version}, "
            f"role={role}, category={category}, status=draft"
        )
        return self.get_optimization(optimization_id)

    def _merge_optimization_rule(self, old_prompt: str, rule_text: str) -> str:
        """合并优化规则到原 prompt

        幂等：若原 prompt 已有 M5-5 自动优化规则区块，替换它；否则追加

        Args:
            old_prompt: 原 prompt
            rule_text: 优化规则文本

        Returns:
            新 prompt
        """
        # 检查是否已有 M5-5 自动优化规则区块
        if OPTIMIZATION_RULE_MARKER in old_prompt:
            # 替换已有区块（从标记开始到 prompt 末尾）
            idx = old_prompt.find(OPTIMIZATION_RULE_MARKER)
            return old_prompt[:idx].rstrip() + rule_text
        else:
            # 追加新区块
            return old_prompt.rstrip() + rule_text

    def _summarize_change(self, old_prompt: str, new_prompt: str, pattern: dict) -> str:
        """生成变更摘要"""
        added_len = len(new_prompt) - len(old_prompt)
        return (
            f"角色={pattern['role']}, 类别={pattern['failure_category']}, "
            f"失败案例数={pattern['failure_count']}, "
            f"新增规则字符数={added_len}"
        )

    def _next_version(self) -> int:
        """获取下一个版本号"""
        cur = self.db.execute(
            "SELECT MAX(version) FROM prompt_optimizations"
        )
        row = cur.fetchone()
        return (row[0] or 0) + 1

    # ===== 3. 应用优化 =====

    def apply_optimization(self, optimization_id: str) -> Optional[dict]:
        """应用优化（备份当前 prompts.yaml → 写入新 prompt → 记录历史）

        Args:
            optimization_id: 优化记录ID

        Returns:
            更新后的优化记录，或 None（不存在或状态不对）
        """
        record = self.get_optimization(optimization_id)
        if record is None:
            return None
        if record["status"] != "draft":
            logger.warning(
                f"M5-5 应用失败（状态非 draft）: id={optimization_id}, status={record['status']}"
            )
            return None

        # 1. 备份当前 prompts.yaml 到 history
        self.history_dir.mkdir(parents=True, exist_ok=True)
        snapshot_filename = (
            f"v{record['version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        snapshot_path = self.history_dir / snapshot_filename
        shutil.copy2(self.prompts_path, snapshot_path)

        # 2. 更新 prompts.yaml 中对应角色的 prompt
        current_prompts = self._load_prompts()
        prompt_key = ROLE_TO_PROMPT_KEY.get(record["role"])
        if not prompt_key:
            logger.error(f"M5-5 应用失败（角色无对应 key）: role={record['role']}")
            return None

        current_prompts[prompt_key] = record["new_prompt"]
        self._save_prompts(current_prompts)

        # 3. 清除 load_prompts 的 lru_cache
        from agent.utils import load_prompts
        load_prompts.cache_clear()

        # 4. 更新优化记录
        self.db.execute(
            """UPDATE prompt_optimizations
            SET status = 'applied', applied_at = ?, snapshot_path = ?
            WHERE optimization_id = ?""",
            (datetime.now(), str(snapshot_path), optimization_id),
        )
        self.db.commit()

        logger.info(
            f"M5-5 应用优化: id={optimization_id}, role={record['role']}, "
            f"snapshot={snapshot_path}"
        )
        return self.get_optimization(optimization_id)

    # ===== 4. 回滚 =====

    def rollback_optimization(self, optimization_id: str) -> Optional[dict]:
        """回滚优化（从历史快照恢复 prompts.yaml）

        Args:
            optimization_id: 优化记录ID

        Returns:
            更新后的优化记录，或 None（不存在或状态不对）
        """
        record = self.get_optimization(optimization_id)
        if record is None:
            return None
        if record["status"] != "applied":
            logger.warning(
                f"M5-5 回滚失败（状态非 applied）: id={optimization_id}, status={record['status']}"
            )
            return None

        snapshot_path = record.get("snapshot_path")
        if not snapshot_path or not Path(snapshot_path).exists():
            logger.error(
                f"M5-5 回滚失败（快照不存在）: id={optimization_id}, snapshot={snapshot_path}"
            )
            return None

        # 从快照恢复
        shutil.copy2(snapshot_path, self.prompts_path)

        # 清除 lru_cache
        from agent.utils import load_prompts
        load_prompts.cache_clear()

        # 更新优化记录
        self.db.execute(
            "UPDATE prompt_optimizations SET status = 'rolled_back', rolled_back_at = ? "
            "WHERE optimization_id = ?",
            (datetime.now(), optimization_id),
        )
        self.db.commit()

        logger.info(f"M5-5 回滚优化: id={optimization_id}, restored from {snapshot_path}")
        return self.get_optimization(optimization_id)

    # ===== 5. 查询 =====

    def list_optimizations(
        self,
        role: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """列出优化记录

        Args:
            role: 角色过滤
            status: 状态过滤（draft/applied/rolled_back）
            limit: 最大返回数
        """
        query = "SELECT * FROM prompt_optimizations WHERE 1=1"
        params: list = []
        if role:
            query += " AND role = ?"
            params.append(role)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY version DESC LIMIT ?"
        params.append(limit)

        cur = self.db.execute(query, params)
        cols = [d[0] for d in cur.description]
        records = []
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            for k in ("created_at", "applied_at", "rolled_back_at"):
                if rec.get(k) and hasattr(rec[k], "isoformat"):
                    rec[k] = rec[k].isoformat()
            records.append(rec)
        return records

    def get_optimization(self, optimization_id: str) -> Optional[dict]:
        """查询单条优化记录"""
        cur = self.db.execute(
            "SELECT * FROM prompt_optimizations WHERE optimization_id = ?",
            (optimization_id,),
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            return None
        rec = dict(zip(cols, row))
        for k in ("created_at", "applied_at", "rolled_back_at"):
            if rec.get(k) and hasattr(rec[k], "isoformat"):
                rec[k] = rec[k].isoformat()
        return rec

    # ===== 6. 当前 prompts =====

    def get_current_prompts(self, role: Optional[str] = None) -> dict:
        """查看当前 prompts

        Args:
            role: 角色过滤（None 则返回全部）

        Returns:
            {role_key: prompt_text} 或 {role_key: prompt_text}（单角色）
        """
        prompts = self._load_prompts()
        if role:
            prompt_key = ROLE_TO_PROMPT_KEY.get(role)
            if not prompt_key:
                return {}
            return {prompt_key: prompts.get(prompt_key, "")}
        return prompts

    # ===== 辅助方法 =====

    def _load_prompts(self) -> dict:
        """加载 prompts.yaml（不使用 lru_cache，确保读到最新内容）"""
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_prompts(self, prompts: dict) -> None:
        """保存 prompts.yaml"""
        with open(self.prompts_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(prompts, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
