"""
MetaCraft Agent 三层记忆管理
对应 TDD 第 4.4 节

三层架构：
- 工作记忆（Working）：LangGraph State，会话级
- 短期记忆（Episodic）：SQLite，近 30 天批次与归因
- 长期记忆（Semantic）：Chroma 向量库，工艺手册 + 历史案例
"""
import json
import os
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
    M4-9: 增加 line_id 字段，支持多产线隔离。
    """
    conflict_id: str
    new_case_id: str
    existing_case_id: str
    conflict_type: str  # hard / soft / confidence
    description: str
    created_at: datetime
    line_id: str = "heat_treatment"


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
        chroma_host: str | None = None,
        chroma_port: int = 8000,
    ):
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.collection_name = chroma_collection
        self.chroma_path = Path(chroma_path) if chroma_path else DEFAULT_CHROMA_PATH
        # M3-1: 支持连接 Docker 部署的 Chroma 服务端（HTTP 模式）
        # 优先级：显式参数 > 环境变量 CHROMA_HOST > None（降级为嵌入式 PersistentClient）
        self.chroma_host = chroma_host or os.environ.get("CHROMA_HOST")
        self.chroma_port = int(os.environ.get("CHROMA_PORT", chroma_port))

        # M4-14: 初始化 SQLite（容错 + WAL 模式提升并发写入；目录创建失败也降级为内存库）
        # M4-16: 加 busy_timeout=5000ms，写锁冲突时自动等待而非立即失败
        self.db = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")  # M4-14: WAL 模式
            self.db.execute("PRAGMA busy_timeout=5000")  # M4-16: 写锁等待 5 秒
            self._init_db()
        except Exception as e:
            logger.error(f"SQLite 初始化失败，降级为内存数据库: {e}")
            self.db = sqlite3.connect(":memory:")
            self._init_db()

        # Chroma 目录创建容错
        try:
            self.chroma_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Chroma 目录创建失败（降级模式）: {e}")

        # 初始化 Chroma（延迟导入，避免无 chroma 时报错）
        self._chroma_client = None
        self._collection = None
        self._ensure_chroma()

    def _init_db(self):
        """初始化短期记忆表结构

        M4-9: 三张表均增加 line_id 列（默认 'heat_treatment'），支持多产线数据隔离。
        """
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS episodic (
                record_id TEXT PRIMARY KEY,
                batch_id TEXT,
                defect_type TEXT,
                root_cause TEXT,
                solution TEXT,
                created_at TIMESTAMP,
                quality_score REAL DEFAULT 0.5,
                line_id TEXT DEFAULT 'heat_treatment'
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
                created_at TIMESTAMP,
                line_id TEXT DEFAULT 'heat_treatment'
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
                created_at TIMESTAMP,
                line_id TEXT DEFAULT 'heat_treatment'
            )
        """)
        # M4-9: 旧表迁移（已有表无 line_id 列时 ALTER TABLE 补列）
        self._migrate_add_line_id()
        self.db.commit()

    def _migrate_add_line_id(self):
        """M4-9: 为旧版表补充 line_id 列（向后兼容）"""
        for table in ("episodic", "feedback", "conflicts"):
            cur = self.db.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cur.fetchall()]
            if "line_id" not in columns:
                self.db.execute(
                    f"ALTER TABLE {table} ADD COLUMN line_id TEXT DEFAULT 'heat_treatment'"
                )
                logger.info(f"表 {table} 已迁移：新增 line_id 列")

    def _ensure_chroma(self):
        """初始化 Chroma 客户端

        M3-1: 支持两种模式
        - HTTP 模式：chroma_host 存在时连接 Docker 部署的 Chroma 服务端
        - 嵌入式模式：否则用 PersistentClient，数据持久化到本地磁盘

        M4-16: 加 _chroma_init_attempted 标志，避免重复初始化导致锁冲突卡死。
        嵌入式 Chroma 的 PersistentClient 重复创建会因 DuckDB/SQLite 文件锁
        阻塞，首次失败后不再重试，直接走降级路径。
        """
        # M4-16: 已尝试过初始化则不再重试（避免锁冲突卡死）
        if getattr(self, "_chroma_init_attempted", False):
            return
        if self._collection is None:
            self._chroma_init_attempted = True
            try:
                import chromadb
                if self.chroma_host:
                    # M3-1: HTTP 模式连接 Docker 部署的 Chroma 服务端
                    self._chroma_client = chromadb.HttpClient(
                        host=self.chroma_host, port=self.chroma_port
                    )
                    logger.info(
                        f"Chroma HTTP 客户端已连接: {self.chroma_host}:{self.chroma_port}"
                    )
                else:
                    # 嵌入式模式
                    self._chroma_client = chromadb.PersistentClient(
                        path=str(self.chroma_path)
                    )
                    logger.info(f"Chroma 嵌入式集合已就绪: {self.collection_name} @ {self.chroma_path}")
                self._collection = self._chroma_client.get_or_create_collection(
                    self.collection_name
                )
                # 初始化成功，清除标志（允许后续重连）
                self._chroma_init_attempted = False
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
        line_id: str = "heat_treatment",
    ) -> str:
        """写入短期记忆

        Args:
            batch_id: 批次ID
            defect_type: 缺陷类型
            root_cause: 根因
            solution: 解决方案
            quality_score: 质量评分
            line_id: 产线ID（M4-9 多产线隔离）

        Returns:
            record_id
        """
        record_id = f"ep_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}"
        self.db.execute(
            "INSERT INTO episodic VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (record_id, batch_id, defect_type, root_cause, solution,
             datetime.now(), quality_score, line_id),
        )
        self.db.commit()
        logger.info(f"短期记忆已写入: {record_id} (line={line_id})")
        return record_id

    def query_episodic(
        self,
        batch_id: Optional[str] = None,
        defect_type: Optional[str] = None,
        days: int = 30,
        line_id: Optional[str] = None,
    ) -> list[dict]:
        """查询短期记忆

        Args:
            batch_id: 批次ID（可选）
            defect_type: 缺陷类型（可选）
            days: 查询天数
            line_id: 产线ID过滤（可选，M4-9 多产线隔离）

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
        if line_id:
            query += " AND line_id = ?"
            params.append(line_id)

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
        # M4-9: metadata 增加 line_id 支持多产线隔离
        metadata = {
            "defect_type": case.defect_type.value,
            "confidence": case.confidence,
            "created_at": case.created_at.isoformat(),
            "source": case.source,
            "line_id": case.line_id,
        }
        bp = case.batch_params
        if bp.temperature is not None:
            metadata["temperature"] = bp.temperature
        if bp.holding_time is not None:
            metadata["holding_time"] = bp.holding_time
        if bp.cooling_rate is not None:
            metadata["cooling_rate"] = bp.cooling_rate
        # M3-7: 标签存储为逗号分隔字符串（Chroma metadata 不支持 list）
        if case.tags:
            metadata["tags"] = ",".join(case.tags)

        self._collection.add(
            ids=[case.case_id],
            documents=[document],
            metadatas=[metadata],
        )
        logger.info(f"长期记忆已写入: {case.case_id} (line={case.line_id})")

        # M3-9: 写入后检测知识冲突（不阻止写入，仅告警 + 记录）
        # M4-9: 冲突检测在 detect_conflicts 内部按 line_id 隔离
        try:
            conflicts = self.detect_conflicts(case)
            for c in conflicts:
                self.save_conflict(c)
                logger.warning(f"知识冲突告警[{c.conflict_type}]: {c.description}")
        except Exception as e:
            logger.error(f"冲突检测失败（不影响写入）: {e}")

        return True

    def search_semantic(
        self,
        query: str,
        top_k: int = 3,
        line_id: Optional[str] = None,
    ) -> list[dict]:
        """语义检索长期记忆

        Args:
            query: 查询文本
            top_k: 返回数量
            line_id: 产线ID过滤（可选，M4-9 多产线隔离；None 则不过滤）

        Returns:
            相似案例列表
        """
        self._ensure_chroma()
        if self._collection is None:
            logger.warning("Chroma 不可用，返回空结果")
            return []

        # M4-9: 按 line_id 过滤（Chroma where 条件）
        where = {"line_id": line_id} if line_id else None
        results = self._collection.query(
            query_texts=[query], n_results=top_k, where=where
        )
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
        line_id: str = "heat_treatment",
    ) -> bool:
        """持久化用户反馈

        Args:
            feedback_id: 反馈ID
            proposal_id: 关联建议ID
            user_id: 用户ID
            action: adopted / rejected / partial
            score: 0-1 评分
            comment: 评论文本
            line_id: 产线ID（M4-9 多产线隔离）

        Returns:
            是否成功
        """
        # M4-16: 写锁重试（busy_timeout 之外的显式重试，应对高并发写竞争）
        import time as _time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.db.execute(
                    "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (feedback_id, proposal_id, user_id, action, score,
                     comment, datetime.now(), line_id),
                )
                self.db.commit()
                logger.info(f"反馈已持久化: {feedback_id}, action={action}, score={score}, line={line_id}")
                return True
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    if attempt < max_retries - 1:
                        _time.sleep(0.1 * (attempt + 1))  # 退避 100ms/200ms
                        logger.warning(f"feedback 写锁冲突，重试 {attempt+1}/{max_retries}: {e}")
                        continue
                logger.error(f"写入反馈失败（重试耗尽）: {e}")
                return False
            except Exception as e:
                logger.error(f"写入反馈失败: {e}")
                return False
        return False

    def query_feedback(
        self,
        proposal_id: Optional[str] = None,
        user_id: Optional[str] = None,
        days: int = 30,
        line_id: Optional[str] = None,
    ) -> list[dict]:
        """查询用户反馈

        Args:
            proposal_id: 关联建议ID（可选）
            user_id: 用户ID（可选）
            days: 查询天数
            line_id: 产线ID过滤（可选，M4-9 多产线隔离）

        Returns:
            反馈记录列表
        """
        since = datetime.now() - timedelta(days=days)
        query = "SELECT * FROM feedback WHERE created_at >= ?"
        params = [since]
        if proposal_id:
            query += " AND proposal_id = ?"
            params.append(proposal_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if line_id:
            query += " AND line_id = ?"
            params.append(line_id)
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

    # ===== M3-10: 反馈驱动批量权重更新 =====

    def aggregate_feedback_update(
        self,
        case_id: str,
        days: int = 90,
    ) -> dict:
        """聚合多条反馈更新案例置信度（M3-10）

        查询该案例相关的所有反馈，按时间衰减加权平均后更新 confidence。
        多条反馈比单条更可信，反馈聚合分占 50%，旧 confidence 占 50%。

        时间衰减：weight = max(0.1, 1 - age_days / days)
        （超过 days 天的反馈权重降至 0.1，不再继续下降）

        Args:
            case_id: 案例ID（用作 proposal_id 查询反馈）
            days: 衰减周期（天），默认 90

        Returns:
            {"feedback_count": N, "old_confidence": x, "new_confidence": y, "updated": bool}
        """
        # 1. 查询关联反馈（proposal_id = case_id）
        feedbacks = self.query_feedback(proposal_id=case_id, days=days * 2)
        if not feedbacks:
            return {
                "feedback_count": 0,
                "old_confidence": None,
                "new_confidence": None,
                "updated": False,
            }

        # 2. 时间衰减加权平均
        now = datetime.now()
        total_weight = 0.0
        weighted_sum = 0.0
        for fb in feedbacks:
            created = fb.get("created_at")
            if isinstance(created, datetime):
                age_days = (now - created).days
            else:
                age_days = 0
            weight = max(0.1, 1 - age_days / days)
            score = fb.get("score", 0.5)
            if score is None:
                score = 0.5
            weighted_sum += score * weight
            total_weight += weight

        aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # 3. 更新 confidence
        self._ensure_chroma()
        if self._collection is None:
            return {
                "feedback_count": len(feedbacks),
                "old_confidence": None,
                "new_confidence": None,
                "updated": False,
            }

        try:
            existing = self._collection.get(ids=[case_id], include=["metadatas"])
            if not existing["metadatas"]:
                return {
                    "feedback_count": len(feedbacks),
                    "old_confidence": None,
                    "new_confidence": None,
                    "updated": False,
                }

            metadata = existing["metadatas"][0]
            old_conf = metadata.get("confidence", 0.5)
            # 多条反馈聚合：反馈分占 50%，旧 confidence 占 50%
            new_conf = old_conf * 0.5 + aggregated_score * 0.5
            metadata["confidence"] = new_conf

            self._collection.update(ids=[case_id], metadatas=[metadata])
            logger.info(
                f"案例 {case_id} 聚合置信度更新: {old_conf:.3f} → {new_conf:.3f} "
                f"({len(feedbacks)} 条反馈, 聚合分 {aggregated_score:.3f})"
            )
            return {
                "feedback_count": len(feedbacks),
                "old_confidence": old_conf,
                "new_confidence": new_conf,
                "updated": True,
            }
        except Exception as e:
            logger.error(f"聚合反馈更新失败: {e}")
            return {
                "feedback_count": len(feedbacks),
                "old_confidence": None,
                "new_confidence": None,
                "updated": False,
            }

    def batch_update_confidence_from_feedback(self, days: int = 30) -> dict:
        """批量处理近期反馈，更新所有有反馈的案例 confidence（M3-10）

        Args:
            days: 查询近 N 天的反馈

        Returns:
            {"processed_cases": N, "total_feedback": N, "updated": N, "skipped": N}
        """
        feedbacks = self.query_feedback(days=days)
        if not feedbacks:
            return {
                "processed_cases": 0,
                "total_feedback": 0,
                "updated": 0,
                "skipped": 0,
            }

        # 按 proposal_id 分组（去重）
        case_ids = set(
            fb["proposal_id"] for fb in feedbacks
            if fb.get("proposal_id")
        )

        updated = 0
        skipped = 0
        for case_id in case_ids:
            result = self.aggregate_feedback_update(case_id, days=days * 3)
            if result["updated"]:
                updated += 1
            else:
                skipped += 1

        logger.info(
            f"批量置信度更新: {len(case_ids)} 个案例, "
            f"{updated} 成功, {skipped} 跳过"
        )
        return {
            "processed_cases": len(case_ids),
            "total_feedback": len(feedbacks),
            "updated": updated,
            "skipped": skipped,
        }

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

    # ===== M4-11: 跨产线看板统计 =====

    def get_line_stats(self, line_id: str, days: int = 30) -> dict:
        """获取单产线统计概览（M4-11 跨产线看板）

        Args:
            line_id: 产线ID
            days: 统计天数（近 N 天）

        Returns:
            {
                "line_id": str,
                "episodic_count": int,        # 近 N 天短期记忆数
                "feedback_count": int,        # 近 N 天反馈数
                "conflict_count": int,        # 冲突记录数
                "semantic_count": int,        # 长期记忆案例数
                "defect_distribution": {type: count},  # 缺陷类型分布
                "action_distribution": {action: count},  # 反馈动作分布
                "adoption_rate": float,       # 采纳率（adopted / total）
                "avg_confidence": float,      # 平均置信度
            }
        """
        # M4-14: TTL 缓存（60 秒），减少 Chroma 全量扫描开销
        import time as _time
        cache_key = (line_id, days)
        if not hasattr(self, "_stats_cache"):
            self._stats_cache = {}
        if cache_key in self._stats_cache:
            cached_stats, cached_time = self._stats_cache[cache_key]
            if _time.time() - cached_time < 60:
                return cached_stats.copy()

        since = datetime.now() - timedelta(days=days)

        # 短期记忆统计（按 line_id + 时间过滤）
        cur = self.db.execute(
            "SELECT COUNT(*) FROM episodic WHERE line_id = ? AND created_at >= ?",
            (line_id, since),
        )
        episodic_count = cur.fetchone()[0]

        # 缺陷类型分布
        cur = self.db.execute(
            "SELECT defect_type, COUNT(*) as cnt FROM episodic "
            "WHERE line_id = ? AND created_at >= ? AND defect_type IS NOT NULL "
            "GROUP BY defect_type ORDER BY cnt DESC",
            (line_id, since),
        )
        defect_dist = {row[0]: row[1] for row in cur.fetchall()}

        # 反馈统计
        cur = self.db.execute(
            "SELECT COUNT(*) FROM feedback WHERE line_id = ? AND created_at >= ?",
            (line_id, since),
        )
        feedback_count = cur.fetchone()[0]

        # 反馈动作分布
        cur = self.db.execute(
            "SELECT action, COUNT(*) as cnt FROM feedback "
            "WHERE line_id = ? AND created_at >= ? AND action IS NOT NULL "
            "GROUP BY action ORDER BY cnt DESC",
            (line_id, since),
        )
        action_dist = {row[0]: row[1] for row in cur.fetchall()}

        # 采纳率
        adopted = action_dist.get("adopted", 0)
        adoption_rate = round(adopted / feedback_count, 3) if feedback_count > 0 else 0.0

        # 冲突统计
        cur = self.db.execute(
            "SELECT COUNT(*) FROM conflicts WHERE line_id = ?",
            (line_id,),
        )
        conflict_count = cur.fetchone()[0]

        # M4-16: 长期记忆统计（Chroma 服务端 where 过滤，避免全量拉取）
        # 旧实现 list_all_semantic(500) 全量拉 documents+metadatas 再 Python 过滤，
        # 每条产线扫一遍，10 并发下 P95 达 10.7s；改用 where + 只拉 metadatas
        semantic_count = 0
        avg_confidence = 0.0
        try:
            self._ensure_chroma()
            if self._collection is not None:
                result = self._collection.get(
                    where={"line_id": line_id},
                    include=["metadatas"],
                    limit=10000,
                )
                metadatas = result.get("metadatas", [])
                semantic_count = len(metadatas)
                if metadatas:
                    confidences = [
                        m.get("confidence", 0.5)
                        for m in metadatas
                        if isinstance(m, dict)
                    ]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
        except Exception as e:
            logger.warning(f"[M4-11] 统计产线 {line_id} 长期记忆失败: {e}")

        result = {
            "line_id": line_id,
            "episodic_count": episodic_count,
            "feedback_count": feedback_count,
            "conflict_count": conflict_count,
            "semantic_count": semantic_count,
            "defect_distribution": defect_dist,
            "action_distribution": action_dist,
            "adoption_rate": adoption_rate,
            "avg_confidence": round(avg_confidence, 3),
        }
        self._stats_cache[cache_key] = (result, _time.time())
        return result

    # ===== 知识冲突检测（M3-9）=====

    def detect_conflicts(self, case: CaseRecord, top_k: int = 10) -> list[ConflictRecord]:
        """检测新案例与已有案例的知识冲突

        冲突类型：
        - hard: 相同缺陷 + 相似参数 + 不同根因（数据错误或工艺漂移）
        - soft: 相同根因 + 不同方案（工艺改进，正常但需记录）
        - confidence: 相同场景 + 置信度差异 > 0.3

        M4-9: 按 case.line_id 隔离，只在同产线内检测冲突。

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
        # M4-9: 传 line_id 过滤，只在同产线内检测冲突
        query = f"{case.defect_type.value} {case.root_cause}"
        candidates = self.search_semantic(query, top_k=top_k, line_id=case.line_id)

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
                    line_id=case.line_id,
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
                    line_id=case.line_id,
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
                    line_id=case.line_id,
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
            conflict: 冲突记录（含 line_id）

        Returns:
            是否成功
        """
        try:
            self.db.execute(
                "INSERT INTO conflicts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conflict.conflict_id, conflict.new_case_id, conflict.existing_case_id,
                 conflict.conflict_type, conflict.description, conflict.created_at,
                 conflict.line_id),
            )
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"保存冲突记录失败: {e}")
            return False

    def list_conflicts(
        self,
        limit: int = 100,
        line_id: Optional[str] = None,
    ) -> list[dict]:
        """列出全部冲突记录（按时间倒序）

        Args:
            limit: 最大返回数量
            line_id: 产线ID过滤（可选，M4-9 多产线隔离）

        Returns:
            冲突记录列表
        """
        if line_id:
            cur = self.db.execute(
                "SELECT * FROM conflicts WHERE line_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (line_id, limit),
            )
        else:
            cur = self.db.execute(
                "SELECT * FROM conflicts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ===== M3-7: 案例库 CRUD =====

    def get_semantic_case(self, case_id: str) -> Optional[dict]:
        """获取单个长期记忆案例

        Args:
            case_id: 案例 ID

        Returns:
            案例字典（id/document/metadata），不存在返回 None
        """
        self._ensure_chroma()
        if self._collection is None:
            return None
        try:
            result = self._collection.get(
                ids=[case_id], include=["metadatas", "documents"]
            )
            ids = result.get("ids", [])
            if not ids:
                return None
            return {
                "id": ids[0],
                "document": (result.get("documents") or [""])[0],
                "metadata": (result.get("metadatas") or [{}])[0],
            }
        except Exception as e:
            logger.error(f"获取案例 {case_id} 失败: {e}")
            return None

    def update_semantic(
        self,
        case_id: str,
        root_cause: Optional[str] = None,
        solution: Optional[str] = None,
        confidence: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """更新长期记忆案例的字段（部分更新）

        未传入的字段保持原值。Chroma 的 update 会覆写整个 document/metadata，
        因此先 get 旧值再合并。

        Args:
            case_id: 案例 ID
            root_cause: 新根因（None 表示不更新）
            solution: 新方案（None 表示不更新）
            confidence: 新置信度（None 表示不更新）
            tags: 新标签列表（None 表示不更新；空列表表示清空标签）

        Returns:
            是否成功（案例不存在或 Chroma 不可用返回 False）
        """
        self._ensure_chroma()
        if self._collection is None:
            logger.warning("Chroma 不可用，跳过更新")
            return False

        existing = self.get_semantic_case(case_id)
        if existing is None:
            logger.warning(f"案例 {case_id} 不存在，无法更新")
            return False

        old_meta = dict(existing.get("metadata") or {})
        old_doc = existing.get("document") or ""
        # 旧 document 形如 "{defect_type}\n{root_cause}\n{solution}"
        parts = old_doc.split("\n", 2)
        old_defect = parts[0] if len(parts) > 0 else ""
        old_root = parts[1] if len(parts) > 1 else ""
        old_solution = parts[2] if len(parts) > 2 else ""

        new_root = root_cause if root_cause is not None else old_root
        new_solution = solution if solution is not None else old_solution
        new_doc = f"{old_defect}\n{new_root}\n{new_solution}"

        new_meta = dict(old_meta)
        if confidence is not None:
            new_meta["confidence"] = float(confidence)
        if tags is not None:
            # 空列表清空标签，非空列表覆盖
            if tags:
                new_meta["tags"] = ",".join(tags)
            else:
                new_meta.pop("tags", None)

        try:
            self._collection.update(
                ids=[case_id],
                documents=[new_doc],
                metadatas=[new_meta],
            )
            logger.info(f"案例 {case_id} 已更新（root_cause={root_cause is not None}, "
                        f"solution={solution is not None}, confidence={confidence is not None}, "
                        f"tags={tags is not None}）")
            return True
        except Exception as e:
            logger.error(f"更新案例 {case_id} 失败: {e}")
            return False

    def delete_semantic(self, case_id: str) -> bool:
        """从长期记忆中删除案例

        同时清理引用该案例的冲突记录（new_case_id 或 existing_case_id 等于 case_id）。

        Args:
            case_id: 案例 ID

        Returns:
            是否成功（Chroma 不可用返回 False；案例不存在仍返回 True）
        """
        self._ensure_chroma()
        if self._collection is None:
            logger.warning("Chroma 不可用，跳过删除")
            return False
        try:
            self._collection.delete(ids=[case_id])
            logger.info(f"案例 {case_id} 已从长期记忆删除")
        except Exception as e:
            logger.error(f"删除案例 {case_id} 失败: {e}")
            return False

        # 清理引用该案例的冲突记录
        try:
            self.db.execute(
                "DELETE FROM conflicts WHERE new_case_id = ? OR existing_case_id = ?",
                (case_id, case_id),
            )
            self.db.commit()
        except Exception as e:
            logger.error(f"清理案例 {case_id} 的冲突记录失败: {e}")

        return True

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

    # ===== M3-13: 遗忘机制扩展 =====

    def cleanup_all(self, also_cleanup_feedback: bool = False) -> dict:
        """清理过期的短期记忆 + 可选清理反馈（M3-13）

        清理条件：
        - episodic: 超过 retention_days 且 quality_score < 0.3
        - feedback: 超过 retention_days * 2（可选）

        Args:
            also_cleanup_feedback: 是否同时清理过期反馈

        Returns:
            {"episodic_deleted": N, "feedback_deleted": N}
        """
        episodic_deleted = self.cleanup_expired()

        feedback_deleted = 0
        if also_cleanup_feedback:
            fb_cutoff = datetime.now() - timedelta(days=self.retention_days * 2)
            cur = self.db.execute(
                "DELETE FROM feedback WHERE created_at < ?",
                (fb_cutoff,),
            )
            self.db.commit()
            feedback_deleted = cur.rowcount

        if feedback_deleted > 0:
            logger.info(f"清理过期反馈: {feedback_deleted} 条")

        return {
            "episodic_deleted": episodic_deleted,
            "feedback_deleted": feedback_deleted,
        }

    def archive_low_quality_semantic(
        self,
        min_confidence: float = 0.3,
        min_age_days: int = 90,
        archive_path: str | Path | None = None,
    ) -> dict:
        """归档低质长期记忆（M3-13 遗忘机制）

        将置信度低于阈值且超过最小存储时间的案例从 Chroma 移出到归档文件。
        归档而非直接删除，保留可追溯性。

        多维度评估：
        1. 置信度 < min_confidence（低质）
        2. 存储时间 > min_age_days（陈旧）
        两个条件同时满足才归档。

        Args:
            min_confidence: 置信度阈值（低于此值考虑归档）
            min_age_days: 最小存储天数（超过此值才考虑归档）
            archive_path: 归档文件路径（默认 data/archived_cases.json）

        Returns:
            {"archived": N, "remaining": N, "archive_total": N}
        """
        self._ensure_chroma()
        if self._collection is None:
            return {"archived": 0, "remaining": 0, "archive_total": 0}

        # 1. 获取全部案例
        all_cases = self.list_all_semantic(limit=10000)
        if not all_cases:
            return {"archived": 0, "remaining": 0, "archive_total": 0}

        # 2. 筛选待归档案例（低质 + 陈旧）
        now = datetime.now()
        cutoff = now - timedelta(days=min_age_days)
        to_archive = []
        keep_count = 0

        for case in all_cases:
            meta = case.get("metadata", {})
            confidence = meta.get("confidence", 0.5)
            created_at_str = meta.get("created_at", "")

            try:
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else now
            except Exception:
                created_at = now

            if confidence < min_confidence and created_at < cutoff:
                to_archive.append(case)
            else:
                keep_count += 1

        if not to_archive:
            logger.info("无低质陈旧案例需归档")
            return {
                "archived": 0,
                "remaining": len(all_cases),
                "archive_total": self._count_archived(archive_path),
            }

        # 3. 写入归档文件
        if archive_path is None:
            archive_path = self.db_path.parent / "archived_cases.json"
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        existing_archive = []
        if archive_path.exists():
            try:
                existing_archive = json.loads(archive_path.read_text(encoding="utf-8"))
            except Exception:
                existing_archive = []

        for case in to_archive:
            existing_archive.append({
                "id": case["id"],
                "document": case.get("document", ""),
                "metadata": case.get("metadata", {}),
                "archived_at": now.isoformat(),
            })

        archive_path.write_text(
            json.dumps(existing_archive, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 4. 从 Chroma 删除
        archive_ids = [c["id"] for c in to_archive]
        self._collection.delete(ids=archive_ids)

        logger.info(
            f"归档低质案例: {len(to_archive)} 条, 剩余 {keep_count} 条, "
            f"归档总计 {len(existing_archive)} 条"
        )

        return {
            "archived": len(to_archive),
            "remaining": keep_count,
            "archive_total": len(existing_archive),
        }

    def _count_archived(self, archive_path: str | Path | None = None) -> int:
        """统计已归档案例数"""
        if archive_path is None:
            archive_path = self.db_path.parent / "archived_cases.json"
        archive_path = Path(archive_path)
        if not archive_path.exists():
            return 0
        try:
            return len(json.loads(archive_path.read_text(encoding="utf-8")))
        except Exception:
            return 0

    def get_archive_stats(self, archive_path: str | Path | None = None) -> dict:
        """获取归档统计（M3-13）

        Args:
            archive_path: 归档文件路径

        Returns:
            {"total": N, "by_defect_type": {type: count}, "avg_confidence": float}
        """
        if archive_path is None:
            archive_path = self.db_path.parent / "archived_cases.json"
        archive_path = Path(archive_path)
        if not archive_path.exists():
            return {"total": 0, "by_defect_type": {}, "avg_confidence": 0.0}

        try:
            archived = json.loads(archive_path.read_text(encoding="utf-8"))
        except Exception:
            return {"total": 0, "by_defect_type": {}, "avg_confidence": 0.0}

        by_defect_type: dict[str, int] = {}
        confidences = []
        for case in archived:
            meta = case.get("metadata", {})
            dtype = meta.get("defect_type", "unknown")
            by_defect_type[dtype] = by_defect_type.get(dtype, 0) + 1
            conf = meta.get("confidence", 0.5)
            if conf is not None:
                confidences.append(conf)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total": len(archived),
            "by_defect_type": by_defect_type,
            "avg_confidence": avg_conf,
        }

    def close(self):
        """关闭连接"""
        self.db.close()
