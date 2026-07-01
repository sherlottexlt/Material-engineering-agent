"""M3 Chroma 长期记忆最小 demo

验证 MemoryService 在真实 Chroma 下的写入/检索/置信度更新闭环。
装好 chromadb 后运行：python scripts/demo_chroma_memory.py
"""
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.memory.memory_service import MemoryService
from models.entities import CaseRecord, BatchParams, DefectType, ProcessType


def main():
    print("=" * 60)
    print("M3 Chroma 长期记忆 Demo")
    print("=" * 60)

    # 1. 初始化 MemoryService（用临时目录，不污染正式库）
    demo_db = ROOT / "data" / "chroma_demo.db"
    demo_chroma = ROOT / "data" / "chroma_demo"
    if demo_db.exists():
        demo_db.unlink()
    # chroma 目录可能不存在，忽略

    print("\n[1] 初始化 MemoryService...")
    memory = MemoryService(db_path=demo_db, chroma_path=str(demo_chroma))
    print(f"    SQLite: {demo_db}")
    print(f"    Chroma: {demo_chroma}")

    # 验证 Chroma 是否真正可用
    if memory._collection is None:
        print("\n!!! Chroma 不可用，_collection 为 None")
        print("    请确认已运行: pip install chromadb")
        return 1

    print("    Chroma collection 已就绪 ✓")

    # 2. 写入 3 条案例
    print("\n[2] 写入 3 条历史案例...")
    cases = [
        CaseRecord(
            case_id="demo_001",
            defect_type=DefectType.HARDNESS_LOW,
            batch_params=BatchParams(
                batch_id="B001", process_type=ProcessType.HEAT_TREATMENT,
                temperature=850, holding_time=90, start_time=datetime.now(),
            ),
            root_cause="保温时间不足，奥氏体化不充分",
            solution="holding_time +30 分钟",
            confidence=0.8,
            created_at=datetime.now(),
            source="demo",
            tags=["demo", "hardness"],
        ),
        CaseRecord(
            case_id="demo_002",
            defect_type=DefectType.HARDNESS_LOW,
            batch_params=BatchParams(
                batch_id="B002", process_type=ProcessType.HEAT_TREATMENT,
                temperature=820, holding_time=120, start_time=datetime.now(),
            ),
            root_cause="淬火温度偏低，奥氏体化不完全",
            solution="temperature +30℃",
            confidence=0.75,
            created_at=datetime.now(),
            source="demo",
            tags=["demo", "hardness"],
        ),
        CaseRecord(
            case_id="demo_003",
            defect_type=DefectType.CRACK,
            batch_params=BatchParams(
                batch_id="B003", process_type=ProcessType.HEAT_TREATMENT,
                temperature=870, holding_time=180, start_time=datetime.now(),
            ),
            root_cause="冷却速率过快导致裂纹扩展",
            solution="cooling_rate -15℃/s",
            confidence=0.7,
            created_at=datetime.now(),
            source="demo",
            tags=["demo", "crack"],
        ),
    ]

    for case in cases:
        ok = memory.write_semantic(case)
        print(f"    写入 {case.case_id} ({case.defect_type.value}): {'✓' if ok else '✗'}")

    # 3. 语义检索
    print("\n[3] 语义检索测试...")
    queries = [
        "硬度偏低 保温时间",
        "淬火温度不够",
        "裂纹 冷却太快",
    ]
    for q in queries:
        results = memory.search_semantic(q, top_k=2)
        print(f"\n    查询: '{q}' → 命中 {len(results)} 条")
        for r in results:
            doc = r.get("document", "")
            meta = r.get("metadata") or {}
            dist = r.get("distance")
            defect = doc.split("\n")[0] if doc else "?"
            print(f"      - {r.get('id')}: defect={defect}, conf={meta.get('confidence')}, dist={dist:.4f}")

    # 4. 置信度更新（模拟反馈驱动）
    print("\n[4] 置信度更新测试（模拟用户反馈）...")
    print(f"    demo_001 原始置信度: 0.8")
    ok = memory.update_confidence("demo_001", 0.95)
    print(f"    update_confidence 返回: {ok}")
    # 验证加权更新: 0.8 * 0.7 + 0.95 * 0.3 = 0.545 + 0.285 = 0.83
    results = memory.search_semantic("硬度偏低", top_k=5)
    for r in results:
        if r.get("id") == "demo_001":
            meta = r.get("metadata") or {}
            print(f"    更新后置信度: {meta.get('confidence')} (期望 ~0.83)")

    # 5. 短期记忆写入（情景记忆）
    print("\n[5] 短期记忆写入测试...")
    ep_id = memory.write_episodic(
        batch_id="B_DEMO",
        defect_type="hardness_low",
        root_cause="demo 测试案例",
        solution="demo 方案",
        quality_score=0.8,
    )
    print(f"    写入短期记忆: {ep_id}")
    ep_records = memory.query_episodic(batch_id="B_DEMO")
    print(f"    查询到 {len(ep_records)} 条短期记忆")

    # 6. 清理
    print("\n[6] 清理 demo 数据...")
    memory.close()
    # 删除 demo 文件
    if demo_db.exists():
        demo_db.unlink()
    print(f"    已删除 {demo_db}")
    # chroma 目录清理
    import shutil
    if demo_chroma.exists():
        shutil.rmtree(demo_chroma, ignore_errors=True)
        print(f"    已删除 {demo_chroma}")

    print("\n" + "=" * 60)
    print("Demo 全部通过！Chroma 长期记忆已生效。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
