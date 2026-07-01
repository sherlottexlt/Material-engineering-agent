"""
MetaCraft Agent 三层记忆管理
对应 TDD 第 4.4 节

三层架构：
- 工作记忆（Working）：LangGraph State，会话级
- 短期记忆（Episodic）：SQLite，近 30 天批次与归因
- 长期记忆（Semantic）：Chroma 向量库，工艺手册 + 历史案例
"""
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from models.entities import CaseRecord

# 默认数据路径
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "metacraft.db"
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent.parent / "data" / "chroma"


# ===== M3-9: 知识冲突检测 =====

# 冲突类型常量
CONFLICT_HARD = "hard"          # 硬冲突：相同缺陷 + 相似参数 + 不同根因
CONFLICT_SOFT = "soft"          # 软冲突：相同根因 + 不同方案
CONFLICT_CONFIDENCE = "confidence"  # 置信度冲突：相同场景 + 置信度差异大


@dataclass
class ConflictRecord:
    """知识冲突记录（M3-9）

    当新案例与已有案例存在矛盾时生成，不阻止写入，仅告警。
    """
    conflict_id: str
    new_case_id: str
    existing_case_id: str
    conflict_type: str  # hard / soft / confidence
    description: str
    created_at: datetime


class MemoryService:
    """三层记忆管理服务

    工作记忆由 LangGraph State 管理（不在此类）
    本类管理短期记忆（SQLite）和长期记忆（Chroma）
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        chroma_path: str | Path | None = None,
        chroma_collection: str = "metacraft_cases",
        retention_days: int = 30,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.collection_name = chroma_collection
        self.chroma_path = Path(chroma_path) if chroma_path else DEFAULT_CHROMA_PATH
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # 初始化 SQLite
        self.db = sqlite3.connect(str(self.db_path))
        self._init_db()

        # 初始化 Chroma（延迟导入，避免无 chroma 时报错）
        self._chroma_client = None
        self._collection = None
        self._ensure_chroma()

    def _init_db(self):
        """初始化短期记忆表结构"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS episodic (
                record_id TEXT PRIMARY KEY,
                batch_id TEXT,
                defect_type TEXT,
                root_cause TEXT,
                solution TEXT,
                created_at TIMESTAMP,
                quality_score REAL DEFAULT 0.5
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                proposal_id TEXT,
                user_id TEXT,
                action TEXT,
                score REAL,
                comment TEXT,
                created_at TIMESTAMP
            )
        """)
        # M3-9: 知识冲突记录表
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS conflicts (
                conflict_id TEXT PRIMARY KEY,
                new_case_id TEXT,
                existing_case_id TEXT,
                conflict_type TEXT,
                description TEXT,
                created_at TIMESTAMP
            )
        """)
        self.db.commit()

    def _ensure_chroma(self):
        """初始化 Chroma 客户端（PersistentClient，数据持久化到磁盘）"""
        if self._collection is None:
            try:
                import chromadb
                self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
                self._collection = self._chroma_client.get_or_create_collection(
                    self.collection_name
                )
                logger.info(f"Chroma 集合已就绪: {self.collection_name} @ {self.chroma_path}")
            except Exception as e:
                logger.warning(f"Chroma 未就绪（降级模式）: {e}")
                self._collection = None

    # ===== 短期记忆（Episodic）=====

    def write_episodic(
        self,
        batch_id: str,
        defect_type: str,
        root_cause: str,
        solution: str,
        quality_score: float = 0.5,
    ) -> str:
        """写入短期记忆

        Args:
            batch_id: 批次ID
            defect_type: 缺陷类型
            root_cause: 根因
            solution: 解决方案
            quality_score: 质量评分

        Returns:
            record_id
        """
        record_id = f"ep_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}"
        self.db.execute(
            "INSERT INTO episodic VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record_id, batch_id, defect_type, root_cause, solution,
             datetime.now(), quality_score),
        )
        self.db.commit()
        logger.info(f"短期记忆已写入: {record_id}")
        return record_id

    def query_episodic(
        self,
        batch_id: Optional[str] = None,
        defect_type: Optional[str] = None,
        days: int = 30,
    ) -> list[dict]:
        """查询短期记忆

        Args:
            batch_id: 批次ID（可选）
            defect_type: 缺陷类型（可选）
            days: 查询天数

        Returns:
            记录列表
        """
        since = datetime.now() - timedelta(days=days)
        query = "SELECT * FROM episodic WHERE created_at >= ?"
        params = [since]

        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        if defect_type:
            query += " AND defect_type = ?"
            params.append(defect_type)

        query += " ORDER BY created_at DESC"
        cur = self.db.execute(query, params)
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ===== 长期记忆（Semantic）=====

    def write_semantic(self, case: CaseRecord) -> bool:
        """写入长期记忆（向量库）

        Args:
            case: 案例记录

        Returns:
            是否成功
        """
        self._ensure_chroma()
        if self._collection is None:
            logger.warning("Chroma 不可用，跳过长期记忆写入")
            return False

        document = f"{case.defect_type.value}\n{case.root_cause}\n{case.solution}"
        # M3-9: metadata 扩展工艺参数（Chroma 不支持 None，仅存非空字段）
        metadata = {
            "defect_type": case.defect_type.value,
            "confidence": case.confidence,
            "created_at": case.created_at.isoformat(),
            "source": case.source,
        }
        bp = case.batch_params
        if bp.temperature is not None:
            metadata["temperature"] = bp.temperature
        if bp.holding_time is not None:
            metadata["holding_time"] = bp.holding_time
        if bp.cooling_rate is not None:
            metadata["cooling_rate"] = bp.cooling_rate

        self._collection.add(
            ids=[case.case_id],
            documents=[document],
            metadatas=[metadata],
        )
        logger.info(f"长期记忆已写入: {case.case_id}")

        # M3-9: 写入后检测知识冲突（不阻止写入，仅告警 + 记录）
        try:
            conflicts = self.detect_conflicts(case)
            for c in conflicts:
                self.save_conflict(c)
                logger.warning(f"知识冲突告警[{c.conflict_type}]: {c.description}")
        except Exception as e:
            logger.error(f"冲突检测失败（不影响写入）: {e}")

        return True

    def search_semantic(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索长期记忆

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相似案例列表
        """
        self._ensure_chroma()
        if self._collection is None:
            logger.warning("Chroma 不可用，返回空结果")
            return []

        results = self._collection.query(query_texts=[query], n_results=top_k)
        return self._format_search_results(results)

    def _format_search_results(self, results: dict) -> list[dict]:
        """格式化检索结果"""
        formatted = []
        if not results or not results.get("documents"):
            return formatted

        for i, doc in enumerate(results["documents"][0]):
            formatted.append({
                "document": doc,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "id": results["ids"][0][i] if results.get("ids") else None,
            })
        return formatted

    # ===== 用户反馈 =====

    def write_feedback(
        self,
        feedback_id: str,
        proposal_id: str,
        user_id: str,
        action: str,
        score: float,
        comment: Optional[str] = None,
    ) -> bool:
        """持久化用户反馈

        Args:
            feedback_id: 反馈ID
            proposal_id: 关联建议ID
            user_id: 用户ID
            action: adopted / rejected / partial
            score: 0-1 评分
            comment: 评论文本

        Returns:
            是否成功
        """
        try:
            self.db.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
                (feedback_id, proposal_id, user_id, action, score,
                 comment, datetime.now()),
            )
            self.db.commit()
            logger.info(f"反馈已持久化: {feedback_id}, action={action}, score={score}")
            return True
        except Exception as e:
            logger.error(f"写入反馈失败: {e}")
            return False

    def query_feedback(
        self,
        proposal_id: Optional[str] = None,
        user_id: Optional[str] = None,
        days: int = 30,
    ) -> list[dict]:
        """查询用户反馈"""
        since = datetime.now() - timedelta(days=days)
        query = "SELECT * FROM feedback WHERE created_at >= ?"
        params = [since]
        if proposal_id:
            query += " AND proposal_id = ?"
            params.append(proposal_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " ORDER BY created_at DESC"
        cur = self.db.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ===== 记忆权重更新（基于反馈）=====

    def update_confidence(self, case_id: str, feedback_score: float) -> bool:
        """根据用户反馈调整案例置信度

        Args:
            case_id: 案例ID
            feedback_score: 反馈评分 0-1

        Returns:
            是否成功
        """
        self._ensure_chroma()
        if self._collection is None:
            return False

        try:
            # 获取原 metadata
            existing = self._collection.get(ids=[case_id])
            if not existing["metadatas"]:
                logger.warning(f"案例不存在: {case_id}")
                return False

            metadata = existing["metadatas"][0]
            # 加权平均更新置信度
            old_conf = metadata.get("confidence", 0.5)
            new_conf = old_conf * 0.7 + feedback_score * 0.3
            metadata["confidence"] = new_conf

            self._collection.update(ids=[case_id], metadatas=[metadata])
            logger.info(f"案例 {case_id} 置信度更新: {old_conf} → {new_conf}")
            return True
        except Exception as e:
            logger.error(f"更新置信度失败: {e}")
            return False

    # ===== 列表查询（M3-12 记忆可视化）=====

    def list_all_episodic(self, limit: int = 100) -> list[dict]:
        """列出全部短期记忆（按时间倒序）

        Args:
            limit: 最大返回数量

        Returns:
            记录列表（含 record_id/batch_id/defect_type/root_cause/solution/created_at/quality_score）
        """
        cur = self.db.execute(
            "SELECT * FROM episodic ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def list_all_feedback(self, limit: int = 100) -> list[dict]:
        """列出全部用户反馈（按时间倒序）"""
        cur = self.db.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def list_all_semantic(self, limit: int = 100) -> list[dict]:
        """列出全部长期记忆（Chroma 案例）

        Args:
            limit: 最大返回数量

        Returns:
            案例列表（含 id/document/metadata）
        """
        self._ensure_chroma()
        if self._collection is None:
            return []

        try:
            # Chroma get 不传 ids 时返回全部（受 limit 限制）
            result = self._collection.get(limit=limit, include=["metadatas", "documents"])
            ids = result.get("ids", [])
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])
            return [
                {
                    "id": ids[i] if i < len(ids) else "",
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
                for i in range(len(ids))
            ]
        except Exception as e:
            logger.error(f"列出长期记忆失败: {e}")
            return []

    def get_memory_stats(self) -> dict:
        """获取记忆统计概览

        Returns:
            {
                "episodic_count": int,
                "feedback_count": int,
                "semantic_count": int,
                "defect_type_distribution": {defect_type: count},
                "action_distribution": {action: count},
                "avg_confidence": float,
                "recent_episodic": [...],  # 最近 5 条
                "recent_feedback": [...],  # 最近 5 条
            }
        """
        # 短期记忆统计
        cur = self.db.execute("SELECT COUNT(*) FROM episodic")
        episodic_count = cur.fetchone()[0]

        # 反馈统计
        cur = self.db.execute("SELECT COUNT(*) FROM feedback")
        feedback_count = cur.fetchone()[0]

        # 缺陷类型分布
        cur = self.db.execute(
            "SELECT defect_type, COUNT(*) as cnt FROM episodic "
            "WHERE defect_type IS NOT NULL GROUP BY defect_type ORDER BY cnt DESC"
        )
        defect_type_dist = {row[0]: row[1] for row in cur.fetchall()}

        # 反馈动作分布
        cur = self.db.execute(
            "SELECT action, COUNT(*) as cnt FROM feedback "
            "WHERE action IS NOT NULL GROUP BY action ORDER BY cnt DESC"
        )
        action_dist = {row[0]: row[1] for row in cur.fetchall()}

        # 长期记忆统计
        semantic_count = 0
        avg_confidence = 0.0
        semantic_records = self.list_all_semantic(limit=500)
        if semantic_records:
            semantic_count = len(semantic_records)
            confidences = [
                r["metadata"].get("confidence", 0.5)
                for r in semantic_records
                if isinstance(r.get("metadata"), dict)
            ]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)

        return {
            "episodic_count": episodic_count,
            "feedback_count": feedback_count,
            "semantic_count": semantic_count,
            "defect_type_distribution": defect_type_dist,
            "action_distribution": action_dist,
            "avg_confidence": round(avg_confidence, 3),
            "recent_episodic": self.list_all_episodic(limit=5),
            "recent_feedback": self.list_all_feedback(limit=5),
        }

    # ===== 知识冲突检测（M3-9）=====

    def detect_conflicts(self, case: CaseRecord, top_k: int = 10) -> list[ConflictRecord]:
        """检测新案例与已有案例的知识冲突

        冲突类型：
        - hard: 相同缺陷 + 相似参数 + 不同根因（数据错误或工艺漂移）
        - soft: 相同根因 + 不同方案（工艺改进，正常但需记录）
        - confidence: 相同场景 + 置信度差异 > 0.3

        Args:
            case: 待写入的新案例
            top_k: 检索候选数量

        Returns:
            冲突记录列表（空列表表示无冲突）
        """
        self._ensure_chroma()
        if self._collection is None:
            return []

        # 用缺陷类型 + 根因关键词检索相似案例
        query = f"{case.defect_type.value} {case.root_cause}"
        candidates = self.search_semantic(query, top_k=top_k)

        conflicts = []
        now = datetime.now()
        for cand in candidates:
            # 排除自身（相同 case_id）
            if cand.get("id") == case.case_id:
                continue

            meta = cand.get("metadata", {})
            # 只比较相同缺陷类型
            if meta.get("defect_type") != case.defect_type.value:
                continue

            # 工艺参数相似度判断（旧数据无工艺参数时降级为相似）
            if not self._params_similar(case, meta):
                continue

            existing_id = cand.get("id", "unknown")
            document = cand.get("document", "")
            existing_root = self._extract_field(document, "root_cause")
            existing_solution = self._extract_field(document, "solution")
            existing_conf = float(meta.get("confidence", 0.5))

            # 硬冲突：相似参数 + 不同根因
            if not self._root_cause_similar(case.root_cause, existing_root):
                conflicts.append(ConflictRecord(
                    conflict_id=f"cf_{int(now.timestamp())}_{uuid.uuid4().hex[:6]}",
                    new_case_id=case.case_id,
                    existing_case_id=existing_id,
                    conflict_type=CONFLICT_HARD,
                    description=f"相同缺陷({case.defect_type.value})+相似参数，根因不同："
                                f"新='{case.root_cause}' vs 旧='{existing_root}'",
                    created_at=now,
                ))
                continue  # 硬冲突已记录，跳过软冲突检测

            # 软冲突：相同根因 + 不同方案
            if case.solution and existing_solution and case.solution != existing_solution:
                conflicts.append(ConflictRecord(
                    conflict_id=f"cf_{int(now.timestamp())}_{uuid.uuid4().hex[:6]}",
                    new_case_id=case.case_id,
                    existing_case_id=existing_id,
                    conflict_type=CONFLICT_SOFT,
                    description=f"相同根因，方案不同："
                                f"新='{case.solution}' vs 旧='{existing_solution}'",
                    created_at=now,
                ))

            # 置信度冲突：置信度差异 > 0.3
            if abs(case.confidence - existing_conf) > 0.3:
                conflicts.append(ConflictRecord(
                    conflict_id=f"cf_{int(now.timestamp())}_{uuid.uuid4().hex[:6]}",
                    new_case_id=case.case_id,
                    existing_case_id=existing_id,
                    conflict_type=CONFLICT_CONFIDENCE,
                    description=f"相同场景，置信度差异大："
                                f"新={case.confidence} vs 旧={existing_conf}",
                    created_at=now,
                ))

        return conflicts

    def _params_similar(
        self,
        case: CaseRecord,
        meta: dict,
        temp_tol: float = 10.0,
        time_tol: float = 15.0,
    ) -> bool:
        """判断工艺参数是否相似

        Args:
            case: 新案例
            meta: 已有案例的 metadata
            temp_tol: 温度容差 (°C)
            time_tol: 保温时间容差 (min)

        Returns:
            是否相似（旧数据无工艺参数时降级为相似）
        """
        has_temp = isinstance(meta.get("temperature"), (int, float))
        has_time = isinstance(meta.get("holding_time"), (int, float))

        # 旧数据无工艺参数，降级为相似
        if not has_temp and not has_time:
            return True

        # 温度比较
        if has_temp and case.batch_params.temperature is not None:
            if abs(float(meta["temperature"]) - case.batch_params.temperature) > temp_tol:
                return False

        # 保温时间比较
        if has_time and case.batch_params.holding_time is not None:
            if abs(float(meta["holding_time"]) - case.batch_params.holding_time) > time_tol:
                return False

        return True

    def _root_cause_similar(self, root1: str, root2: str) -> bool:
        """判断两个根因是否相似（字符级 bigram 重叠度 ≥ 50%）

        用 bigram 而非空格分词，适配中文无空格的根因文本。

        Args:
            root1: 根因文本1
            root2: 根因文本2

        Returns:
            是否相似
        """
        if not root1 or not root2:
            return False
        # 字符级 bigram 集合
        bigrams1 = {root1[i:i + 2] for i in range(len(root1) - 1)}
        bigrams2 = {root2[i:i + 2] for i in range(len(root2) - 1)}
        if not bigrams1 or not bigrams2:
            return False
        overlap = len(bigrams1 & bigrams2)
        threshold = min(len(bigrams1), len(bigrams2)) * 0.5
        return overlap >= threshold

    def _extract_field(self, document: str, field: str) -> str:
        """从 document 中提取字段

        document 格式：f"{defect_type}\\n{root_cause}\\n{solution}"

        Args:
            document: Chroma 存储的文档
            field: 要提取的字段名 (defect_type/root_cause/solution)

        Returns:
            字段值（找不到返回空字符串）
        """
        parts = document.split("\n")
        idx = {"defect_type": 0, "root_cause": 1, "solution": 2}.get(field)
        if idx is not None and idx < len(parts):
            return parts[idx]
        return ""

    def save_conflict(self, conflict: ConflictRecord) -> bool:
        """持久化冲突记录到 SQLite

        Args:
            conflict: 冲突记录

        Returns:
            是否成功
        """
        try:
            self.db.execute(
                "INSERT INTO conflicts VALUES (?, ?, ?, ?, ?, ?)",
                (conflict.conflict_id, conflict.new_case_id, conflict.existing_case_id,
                 conflict.conflict_type, conflict.description, conflict.created_at),
            )
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"保存冲突记录失败: {e}")
            return False

    def list_conflicts(self, limit: int = 100) -> list[dict]:
        """列出全部冲突记录（按时间倒序）

        Args:
            limit: 最大返回数量

        Returns:
            冲突记录列表
        """
        cur = self.db.execute(
            "SELECT * FROM conflicts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ===== 遗忘机制 =====

    def cleanup_expired(self) -> int:
        """清理过期的短期记忆

        Returns:
            清理的记录数
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        cur = self.db.execute(
            "DELETE FROM episodic WHERE created_at < ? AND quality_score < 0.3",
            (cutoff,),
        )
        self.db.commit()
        deleted = cur.rowcount
        if deleted > 0:
            logger.info(f"清理过期低质记忆: {deleted} 条")
        return deleted

    def close(self):
        """关闭连接"""
        self.db.close()
