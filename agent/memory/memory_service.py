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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from models.entities import CaseRecord

# 默认数据路径
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "metacraft.db"
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent.parent / "data" / "chroma"


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
        self._collection.add(
            ids=[case.case_id],
            documents=[document],
            metadatas=[{
                "defect_type": case.defect_type.value,
                "confidence": case.confidence,
                "created_at": case.created_at.isoformat(),
                "source": case.source,
            }],
        )
        logger.info(f"长期记忆已写入: {case.case_id}")
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
