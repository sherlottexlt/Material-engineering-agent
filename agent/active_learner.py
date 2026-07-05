"""M5-7 知识主动学习模块

Agent 主动询问不确定的案例，从专家回答中提炼通用规则，更新 handbook 知识库。

主动学习闭环：
1. identify_candidates: 识别学习候选
   - low_quality: M5-6 低质量案例（quality_score < threshold）
   - failure_case: M5-4 失败案例（status=open）
   - high_frequency: 高频缺陷类型（统计 episodic 中 defect_type 频次）
2. generate_question: 针对候选生成专家询问问题（规则模板，不依赖 LLM）
3. submit_answer: 专家提交回答
4. extract_rule: 从回答中提炼通用规则（格式化为 handbook chunk）
5. add_rule_to_handbook: 将规则写入 handbook_index.json（让 search_handbook 能检索到）

幂等性：同 source_type+source_id 已存在则跳过，避免重复生成候选。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


# 学习候选来源类型
SOURCE_LOW_QUALITY = "low_quality"
SOURCE_FAILURE_CASE = "failure_case"
SOURCE_HIGH_FREQUENCY = "high_frequency"

# 候选状态
STATUS_PENDING = "pending"
STATUS_ANSWERED = "answered"
STATUS_LEARNED = "learned"
STATUS_SKIPPED = "skipped"

# 默认参数
DEFAULT_LOW_QUALITY_THRESHOLD = 0.4
DEFAULT_FAILURE_DAYS = 30
DEFAULT_FREQUENCY_TOP_N = 5
DEFAULT_FREQUENCY_MIN_COUNT = 3  # 至少出现 3 次才算高频


class ActiveLearner:
    """主动学习器

    用法：
        learner = ActiveLearner(memory)
        # 1. 识别候选
        result = learner.identify_candidates(line_id="heat_treatment")
        # 2. 专家查看候选问题并提交回答
        learner.submit_answer(candidate_id, answer="根因是淬火温度不足...", auto_extract=True)
        # 3. 规则自动提炼并写入 handbook 索引
    """

    def __init__(
        self,
        memory,
        handbook_index_path: Optional[str | Path] = None,
    ):
        self.memory = memory
        self.db = memory.db
        # handbook 索引路径（默认 data/handbook_index.json）
        if handbook_index_path is None:
            self.handbook_index_path = (
                Path(__file__).parent.parent / "data" / "handbook_index.json"
            )
        else:
            self.handbook_index_path = Path(handbook_index_path)

    # ===== 候选识别 =====

    def identify_candidates(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
        max_count: int = 20,
        low_quality_threshold: float = DEFAULT_LOW_QUALITY_THRESHOLD,
        frequency_top_n: int = DEFAULT_FREQUENCY_TOP_N,
        frequency_min_count: int = DEFAULT_FREQUENCY_MIN_COUNT,
    ) -> dict:
        """识别学习候选并写入 learning_candidates 表

        Args:
            line_id: 产线ID过滤（可选，支持 str 或 list[str]）
            days: 失败案例/高频缺陷统计天数
            max_count: 最大候选数量
            low_quality_threshold: 低质量案例阈值
            frequency_top_n: 高频缺陷返回前 N 个
            frequency_min_count: 高频缺陷最低出现次数

        Returns:
            {
                "total": int,
                "by_source": {"low_quality": int, "failure_case": int, "high_frequency": int},
                "skipped_duplicate": int,
                "candidate_ids": list[str],
            }
        """
        by_source = {
            SOURCE_LOW_QUALITY: 0,
            SOURCE_FAILURE_CASE: 0,
            SOURCE_HIGH_FREQUENCY: 0,
        }
        skipped_duplicate = 0
        candidate_ids: list[str] = []

        # 1. 低质量案例
        low_quality = self.memory.get_low_quality_cases(
            line_id=line_id, threshold=low_quality_threshold, limit=max_count
        )
        for rec in low_quality:
            if len(candidate_ids) >= max_count:
                break
            source_id = rec.get("record_id", "")
            if not source_id:
                continue
            # 幂等检查
            if self._candidate_exists(SOURCE_LOW_QUALITY, source_id):
                skipped_duplicate += 1
                continue
            question = self.generate_question(
                {
                    "source_type": SOURCE_LOW_QUALITY,
                    "source_id": source_id,
                    "defect_type": rec.get("defect_type", ""),
                    "record": rec,
                }
            )
            cid = self._new_candidate_id()
            ok = self.memory.save_learning_candidate(
                candidate_id=cid,
                source_type=SOURCE_LOW_QUALITY,
                source_id=source_id,
                line_id=rec.get("line_id", "heat_treatment"),
                defect_type=rec.get("defect_type"),
                question=question,
            )
            if ok:
                by_source[SOURCE_LOW_QUALITY] += 1
                candidate_ids.append(cid)

        # 2. 失败案例（status=open）
        failures = self._query_open_failures(line_id=line_id, days=days, limit=max_count)
        for f in failures:
            if len(candidate_ids) >= max_count:
                break
            source_id = f.get("failure_id", "") or f.get("case_id", "")
            if not source_id:
                continue
            if self._candidate_exists(SOURCE_FAILURE_CASE, source_id):
                skipped_duplicate += 1
                continue
            question = self.generate_question(
                {
                    "source_type": SOURCE_FAILURE_CASE,
                    "source_id": source_id,
                    "defect_type": f.get("root_cause", "") or f.get("category", ""),
                    "failure": f,
                }
            )
            cid = self._new_candidate_id()
            ok = self.memory.save_learning_candidate(
                candidate_id=cid,
                source_type=SOURCE_FAILURE_CASE,
                source_id=source_id,
                line_id=f.get("line_id", "heat_treatment"),
                defect_type=f.get("category"),
                question=question,
            )
            if ok:
                by_source[SOURCE_FAILURE_CASE] += 1
                candidate_ids.append(cid)

        # 3. 高频缺陷类型
        frequencies = self.memory.get_defect_frequency(
            line_id=line_id, days=days, top_n=frequency_top_n
        )
        for freq in frequencies:
            if len(candidate_ids) >= max_count:
                break
            count = freq.get("count", 0)
            if count < frequency_min_count:
                continue  # 频次不够，跳过
            defect_type = freq.get("defect_type", "")
            source_id = defect_type  # 高频缺陷用 defect_type 作为 source_id
            if self._candidate_exists(SOURCE_HIGH_FREQUENCY, source_id):
                skipped_duplicate += 1
                continue
            question = self.generate_question(
                {
                    "source_type": SOURCE_HIGH_FREQUENCY,
                    "source_id": source_id,
                    "defect_type": defect_type,
                    "frequency": freq,
                    "days": days,
                }
            )
            cid = self._new_candidate_id()
            ok = self.memory.save_learning_candidate(
                candidate_id=cid,
                source_type=SOURCE_HIGH_FREQUENCY,
                source_id=source_id,
                line_id=freq.get("line_id", "heat_treatment"),
                defect_type=defect_type,
                question=question,
            )
            if ok:
                by_source[SOURCE_HIGH_FREQUENCY] += 1
                candidate_ids.append(cid)

        return {
            "total": len(candidate_ids),
            "by_source": by_source,
            "skipped_duplicate": skipped_duplicate,
            "candidate_ids": candidate_ids,
        }

    def _query_open_failures(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
        limit: int = 100,
    ) -> list[dict]:
        """查询 status=open 的失败案例"""
        from datetime import timedelta
        since = datetime.now() - timedelta(days=days)
        query = "SELECT * FROM failure_cases WHERE status = 'open' AND collected_at >= ?"
        params: list = [since]
        if line_id:
            if isinstance(line_id, (list, tuple)):
                placeholders = ",".join("?" for _ in line_id)
                query += f" AND line_id IN ({placeholders})"
                params.extend(line_id)
            else:
                query += " AND line_id = ?"
                params.append(line_id)
        query += " ORDER BY collected_at DESC LIMIT ?"
        params.append(limit)

        try:
            cur = self.db.execute(query, params)
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"查询失败案例失败: {e}")
            return []

    def _candidate_exists(self, source_type: str, source_id: str) -> bool:
        """检查候选是否已存在（幂等）"""
        try:
            cur = self.db.execute(
                "SELECT 1 FROM learning_candidates WHERE source_type = ? AND source_id = ? LIMIT 1",
                (source_type, source_id),
            )
            return cur.fetchone() is not None
        except Exception:
            return False

    def _new_candidate_id(self) -> str:
        """生成候选ID"""
        return f"lc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # ===== 问题生成（规则模板，不依赖 LLM）=====

    def generate_question(self, context: dict) -> str:
        """针对候选生成专家询问问题

        Args:
            context: {
                source_type: 'low_quality' / 'failure_case' / 'high_frequency',
                source_id: str,
                defect_type: str,
                record: dict,        # low_quality 时提供
                failure: dict,       # failure_case 时提供
                frequency: dict,     # high_frequency 时提供
                days: int,           # high_frequency 时提供
            }

        Returns:
            问题文本
        """
        source_type = context.get("source_type", "")
        defect_type = context.get("defect_type", "") or "未知缺陷"

        if source_type == SOURCE_LOW_QUALITY:
            record = context.get("record", {})
            record_id = context.get("source_id", "")
            issues: list[str] = []
            root_cause = (record.get("root_cause") or "").strip()
            solution = (record.get("solution") or "").strip()
            if not root_cause:
                issues.append("root_cause 为空")
            elif len(root_cause) < 20:
                issues.append(f"root_cause 偏短（{len(root_cause)} 字符）")
            if not solution:
                issues.append("solution 为空")
            elif len(solution) < 20:
                issues.append(f"solution 偏短（{len(solution)} 字符）")
            issues_text = "、".join(issues) if issues else "信息不完整"

            return (
                f"【低质量案例学习】案例 {record_id}（缺陷类型: {defect_type}）"
                f"质量评分低于阈值（{issues_text}）。\n"
                f"请补充该案例的详细根因分析和可执行的解决方案，"
                f"以便提炼为通用知识规则。"
            )

        elif source_type == SOURCE_FAILURE_CASE:
            failure = context.get("failure", {})
            failure_id = context.get("source_id", "")
            category = failure.get("category", "未知")
            failure_reason = failure.get("failure_reason", "无详细原因")
            confidence = failure.get("confidence")
            improvement = failure.get("improvement_pct")

            details = []
            if confidence is not None:
                details.append(f"置信度 {confidence}")
            if improvement is not None:
                details.append(f"效果改善 {improvement}%")
            details_text = f"（{', '.join(details)}）" if details else ""

            return (
                f"【失败案例学习】案例 {failure_id} 出现 {category} 失败{details_text}。\n"
                f"失败原因: {failure_reason}\n"
                f"请确认根因并给出改进建议，以避免类似案例再次失败。"
            )

        elif source_type == SOURCE_HIGH_FREQUENCY:
            freq = context.get("frequency", {})
            count = freq.get("count", 0)
            days = context.get("days", 30)

            return (
                f"【高频缺陷学习】缺陷类型 {defect_type} 在最近 {days} 天出现 {count} 次。\n"
                f"请提供该缺陷类型的标准根因分析模板和解决方案模板，"
                f"以便 Agent 在后续分析中复用。"
            )

        else:
            return f"【通用学习】请提供案例 {context.get('source_id', '')} 的详细分析。"

    # ===== 规则提炼 + 手册写入 =====

    def extract_rule(self, candidate: dict, answer: str) -> str:
        """从专家回答中提炼通用规则（格式化为 handbook chunk）

        Args:
            candidate: learning_candidates 记录
            answer: 专家回答文本

        Returns:
            规则文本（handbook chunk 格式）
        """
        defect_type = candidate.get("defect_type", "") or "通用"
        source_type = candidate.get("source_type", "")
        question = candidate.get("question", "")
        line_id = candidate.get("line_id", "heat_treatment")

        source_label = {
            SOURCE_LOW_QUALITY: "低质量案例补全",
            SOURCE_FAILURE_CASE: "失败案例改进",
            SOURCE_HIGH_FREQUENCY: "高频缺陷模板",
        }.get(source_type, "主动学习")

        rule_text = (
            f"## 主动学习规则 - {defect_type}\n\n"
            f"**来源**: {source_label}（产线: {line_id}）\n"
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"### 问题\n{question}\n\n"
            f"### 专家回答\n{answer}\n\n"
            f"### 提炼规则\n"
            f"针对 {defect_type} 缺陷，建议参考上述专家回答中的根因分析和解决方案。"
        )
        return rule_text

    def add_rule_to_handbook(
        self,
        rule_text: str,
        defect_type: str,
        source: str = "active_learning",
    ) -> dict:
        """将规则写入 handbook_index.json

        Args:
            rule_text: 规则文本（handbook chunk 格式）
            defect_type: 缺陷类型（用于 chunk metadata）
            source: 来源标签

        Returns:
            {
                "success": bool,
                "chunk_id": str,
                "total_chunks": int,
                "error": str (失败时),
            }
        """
        try:
            # 读取现有索引
            if self.handbook_index_path.exists():
                with open(self.handbook_index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "source_files": [],
                    "total_chunks": 0,
                    "chunks": [],
                }

            # 生成新 chunk_id
            existing_ids = {c.get("chunk_id", "") for c in index.get("chunks", [])}
            chunk_num = len(index.get("chunks", []))
            chunk_id = f"al_{chunk_num:04d}"
            while chunk_id in existing_ids:
                chunk_num += 1
                chunk_id = f"al_{chunk_num:04d}"

            new_chunk = {
                "chunk_id": chunk_id,
                "source": f"{source}:{defect_type}",
                "page": None,
                "section": f"主动学习规则 - {defect_type}",
                "content": rule_text,
                "char_count": len(rule_text),
            }

            index.setdefault("chunks", []).append(new_chunk)
            index["total_chunks"] = len(index["chunks"])
            index["updated_at"] = datetime.now().isoformat()

            # 写回文件
            self.handbook_index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.handbook_index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

            logger.info(
                f"主动学习规则已写入手册: chunk_id={chunk_id}, defect_type={defect_type}"
            )
            return {
                "success": True,
                "chunk_id": chunk_id,
                "total_chunks": index["total_chunks"],
            }
        except Exception as e:
            logger.error(f"写入手册失败: {e}")
            return {
                "success": False,
                "chunk_id": "",
                "total_chunks": 0,
                "error": str(e)[:200],
            }

    # ===== 提交回答闭环 =====

    def submit_answer(
        self,
        candidate_id: str,
        answer: str,
        auto_extract: bool = True,
    ) -> dict:
        """提交专家回答 → 提炼规则 → 写入手册

        Args:
            candidate_id: 候选ID
            answer: 专家回答
            auto_extract: 是否自动提炼规则并写入手册

        Returns:
            {
                "success": bool,
                "candidate_id": str,
                "status": str,
                "rule_extracted": bool,
                "handbook_updated": bool,
                "chunk_id": str (写入手册时),
                "error": str (失败时),
            }
        """
        candidate = self.memory.get_learning_candidate(candidate_id)
        if not candidate:
            return {
                "success": False,
                "candidate_id": candidate_id,
                "status": "not_found",
                "rule_extracted": False,
                "handbook_updated": False,
                "error": f"候选 {candidate_id} 不存在",
            }

        if candidate.get("status") not in (STATUS_PENDING, STATUS_ANSWERED):
            return {
                "success": False,
                "candidate_id": candidate_id,
                "status": candidate.get("status"),
                "rule_extracted": False,
                "handbook_updated": False,
                "error": f"候选状态为 {candidate.get('status')}，无法提交回答",
            }

        # 1. 保存回答，标记 answered
        self.memory.update_learning_candidate(
            candidate_id=candidate_id,
            answer=answer,
            status=STATUS_ANSWERED,
        )

        rule_extracted = False
        handbook_updated = False
        chunk_id = ""

        # 2. 可选：提炼规则 + 写入手册
        if auto_extract and answer.strip():
            rule_text = self.extract_rule(candidate, answer)

            # 保存规则到候选记录
            self.memory.update_learning_candidate(
                candidate_id=candidate_id,
                rule_text=rule_text,
            )
            rule_extracted = True

            # 写入手册
            defect_type = candidate.get("defect_type", "") or "通用"
            result = self.add_rule_to_handbook(
                rule_text=rule_text,
                defect_type=defect_type,
                source="active_learning",
            )
            if result.get("success"):
                handbook_updated = True
                chunk_id = result.get("chunk_id", "")
                # 标记已写入手册 + learned
                self.memory.update_learning_candidate(
                    candidate_id=candidate_id,
                    status=STATUS_LEARNED,
                    rule_added_to_handbook=1,
                )
            else:
                # 写入手册失败，仍标记 learned（规则已提炼）
                self.memory.update_learning_candidate(
                    candidate_id=candidate_id,
                    status=STATUS_LEARNED,
                )

        return {
            "success": True,
            "candidate_id": candidate_id,
            "status": STATUS_LEARNED if (auto_extract and rule_extracted) else STATUS_ANSWERED,
            "rule_extracted": rule_extracted,
            "handbook_updated": handbook_updated,
            "chunk_id": chunk_id,
        }

    # ===== 查询代理（方便 API 调用）=====

    def list_candidates(
        self,
        status: Optional[str] = None,
        line_id: Optional[str | list[str]] = None,
        source_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """列出学习候选（代理到 memory.list_learning_candidates）"""
        return self.memory.list_learning_candidates(
            status=status,
            line_id=line_id,
            source_type=source_type,
            limit=limit,
        )

    def get_candidate(self, candidate_id: str) -> Optional[dict]:
        """查询单条候选（代理到 memory.get_learning_candidate）"""
        return self.memory.get_learning_candidate(candidate_id)

    def skip_candidate(self, candidate_id: str) -> dict:
        """跳过候选（status → skipped）"""
        candidate = self.memory.get_learning_candidate(candidate_id)
        if not candidate:
            return {
                "success": False,
                "candidate_id": candidate_id,
                "error": f"候选 {candidate_id} 不存在",
            }

        if candidate.get("status") != STATUS_PENDING:
            return {
                "success": False,
                "candidate_id": candidate_id,
                "status": candidate.get("status"),
                "error": f"候选状态为 {candidate.get('status')}，无法跳过",
            }

        ok = self.memory.update_learning_candidate(
            candidate_id=candidate_id,
            status=STATUS_SKIPPED,
        )
        return {
            "success": ok,
            "candidate_id": candidate_id,
            "status": STATUS_SKIPPED if ok else candidate.get("status"),
        }

    def get_learning_stats(
        self,
        line_id: Optional[str | list[str]] = None,
        days: int = 30,
    ) -> dict:
        """学习统计（代理到 memory.get_learning_stats）"""
        return self.memory.get_learning_stats(line_id=line_id, days=days)
