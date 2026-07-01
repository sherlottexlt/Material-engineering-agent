"""
数据库初始化脚本

用途：初始化 SQLite 短期记忆数据库
运行：python scripts/init_db.py
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.memory.memory_service import MemoryService, DEFAULT_DB_PATH  # noqa: E402
from loguru import logger  # noqa: E402


def init_database(db_path: str | Path = DEFAULT_DB_PATH):
    """初始化数据库

    创建表结构，并插入示例数据
    """
    logger.info(f"初始化数据库: {db_path}")

    memory = MemoryService(db_path=db_path)

    # 插入示例案例
    sample_cases = [
        {
            "batch_id": "B20260620-A",
            "defect_type": "hardness_low",
            "root_cause": "保温时间不足",
            "solution": "保温时间 +15 分钟",
            "quality_score": 0.9,
        },
        {
            "batch_id": "B20260615-C",
            "defect_type": "hardness_low",
            "root_cause": "冷却速率过低",
            "solution": "提高冷却速率至 10 ℃/s",
            "quality_score": 0.85,
        },
        {
            "batch_id": "B20260610-B",
            "defect_type": "deformation",
            "root_cause": "装炉方式不当导致受热不均",
            "solution": "优化装炉方式，确保间距 ≥ 50mm",
            "quality_score": 0.8,
        },
    ]

    for case in sample_cases:
        memory.write_episodic(**case)
        logger.info(f"插入示例案例: {case['batch_id']}")

    memory.close()
    logger.info(f"数据库初始化完成: {db_path}")
    logger.info(f"共插入 {len(sample_cases)} 条示例案例")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="初始化 MetaCraft Agent 数据库")
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="数据库路径",
    )
    args = parser.parse_args()

    init_database(args.db_path)
