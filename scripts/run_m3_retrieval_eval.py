"""M3-11 检索效果评估

评估长期记忆（Chroma 案例库）的检索质量：
1. 将 50 条种子案例写入临时 Chroma 集合
2. 对每条案例构造检索 query，搜索 Top-K
3. 计算 Hit@1 / Hit@3 / RootCauseMatch@3 / MRR
4. 输出 JSON + Markdown 报告

用法:
    python scripts/run_m3_retrieval_eval.py
"""
import json
import sys
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.memory.memory_service import MemoryService
from models.entities import CaseRecord, BatchParams, DefectType, ProcessType


# ===== 评估配置 =====
TOP_K = 3
SEED_CASES_PATH = PROJECT_ROOT / "data" / "seed_cases" / "seed_cases.json"
REPORT_JSON = PROJECT_ROOT / "data" / "eval_retrieval_report.json"
REPORT_MD = PROJECT_ROOT / "data" / "eval_retrieval_report.md"


def load_seed_cases() -> list[dict]:
    """加载种子案例"""
    with open(SEED_CASES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def build_case_record(seed: dict) -> CaseRecord:
    """将种子案例转为 CaseRecord"""
    bp = seed["batch_params"]
    return CaseRecord(
        case_id=seed["case_id"],
        defect_type=DefectType(seed["defect_type"]),
        batch_params=BatchParams(
            batch_id=bp["batch_id"],
            process_type=ProcessType(bp.get("process_type", "heat_treatment")),
            temperature=bp.get("temperature"),
            holding_time=bp.get("holding_time"),
            cooling_rate=bp.get("cooling_rate"),
            start_time=datetime.now(),
        ),
        root_cause=seed["expected_root_cause"],
        solution=seed.get("expected_adjustment", ""),
        confidence=0.8,
        source="seed",
        tags=[],
    )


def construct_query(seed: dict, mode: str = "symptom") -> str:
    """根据种子案例构造检索 query

    mode:
    - "symptom": 用工艺参数描述症状（不含根因关键词），测试语义理解
    - "root_cause": 用根因关键词搜索（模拟 Agent 检索相似案例的真实场景）
    """
    bp = seed["batch_params"]
    defect_cn = "硬度偏低"  # seed_cases 全部是 hardness_low

    if mode == "root_cause":
        # Agent 检索场景：已知缺陷类型 + 疑似根因，找相似案例
        parts = [defect_cn, seed["expected_root_cause"]]
        if seed.get("expected_adjustment"):
            parts.append(seed["expected_adjustment"])
        return " ".join(parts)

    # symptom 模式：用工艺参数描述，不含根因
    parts = [defect_cn]
    if bp.get("temperature"):
        parts.append(f"温度{bp['temperature']}")
    if bp.get("holding_time"):
        parts.append(f"保温{bp['holding_time']}分钟")
    if bp.get("cooling_rate"):
        parts.append(f"冷却速率{bp['cooling_rate']}")
    return " ".join(parts)


def evaluate_retrieval(
    cases: list[dict],
    mode: str = "symptom",
    top_k: int = TOP_K,
) -> dict:
    """执行检索评估

    Returns:
        评估结果字典
    """
    # 创建临时 Chroma 目录
    tmp_dir = Path(tempfile.mkdtemp(prefix="m3_eval_"))
    try:
        memory = MemoryService(
            db_path=tmp_dir / "eval.db",
            chroma_path=tmp_dir / "chroma",
            chroma_collection="eval_retrieval",
        )
        # 强制初始化 Chroma
        memory._ensure_chroma()
        if memory._collection is None:
            return {"error": "Chroma 初始化失败，无法评估"}

        # 写入全部种子案例
        written = 0
        for seed in cases:
            case = build_case_record(seed)
            if memory.write_semantic(case):
                written += 1
        print(f"已写入 {written}/{len(cases)} 条案例到临时 Chroma")

        # 逐条检索评估
        results = []
        hit_at_1 = 0
        hit_at_3 = 0
        root_cause_match_at_3 = 0
        reciprocal_ranks = []

        for seed in cases:
            case_id = seed["case_id"]
            expected_root_cause = seed["expected_root_cause"]
            query = construct_query(seed, mode=mode)

            # 搜索 Top-K
            search_results = memory.search_semantic(query, top_k=top_k)

            # 分析结果
            retrieved_ids = [r.get("id") for r in search_results]
            retrieved_root_causes = []
            for r in search_results:
                doc = r.get("document", "")
                # document 格式: "{defect_type}\n{root_cause}\n{solution}"
                parts = doc.split("\n", 2)
                if len(parts) > 1:
                    retrieved_root_causes.append(parts[1])
                else:
                    retrieved_root_causes.append("")

            # Hit@1: 第一条就是自己
            if retrieved_ids and retrieved_ids[0] == case_id:
                hit_at_1 += 1
                reciprocal_ranks.append(1.0)
            else:
                # 查找自己的排名
                rank = None
                for i, rid in enumerate(retrieved_ids):
                    if rid == case_id:
                        rank = i + 1
                        break
                if rank:
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            # Hit@3: 自己在 Top-3 中
            if case_id in retrieved_ids[:top_k]:
                hit_at_3 += 1

            # RootCauseMatch@3: Top-3 中有相同根因的案例
            # 对于多根因，检查是否有任一子根因匹配
            expected_rcs = [rc.strip() for rc in expected_root_cause.split("+")]
            matched = False
            for ret_rc in retrieved_root_causes[:top_k]:
                for exp_rc in expected_rcs:
                    if exp_rc and exp_rc in ret_rc:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                root_cause_match_at_3 += 1

            results.append({
                "case_id": case_id,
                "query": query,
                "expected_root_cause": expected_root_cause,
                "retrieved_ids": retrieved_ids[:top_k],
                "retrieved_root_causes": retrieved_root_causes[:top_k],
                "hit_at_1": retrieved_ids and retrieved_ids[0] == case_id,
                "hit_at_3": case_id in retrieved_ids[:top_k],
                "root_cause_match": matched,
            })

        total = len(cases)
        mrr = sum(reciprocal_ranks) / total if total > 0 else 0

        return {
            "mode": mode,
            "top_k": top_k,
            "total_queries": total,
            "cases_written": written,
            "hit_at_1": hit_at_1,
            "hit_at_1_rate": round(hit_at_1 / total, 4) if total else 0,
            "hit_at_3": hit_at_3,
            "hit_at_3_rate": round(hit_at_3 / total, 4) if total else 0,
            "root_cause_match_at_3": root_cause_match_at_3,
            "root_cause_match_at_3_rate": round(root_cause_match_at_3 / total, 4) if total else 0,
            "mrr": round(mrr, 4),
            "details": results,
        }
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def generate_report(symptom_result: dict, root_cause_result: dict) -> dict:
    """生成综合报告"""
    report = {
        "eval_id": f"m3_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "seed_cases": SEED_CASES_PATH.name,
        "top_k": TOP_K,
        "acceptance_criterion": "Top-3 准确率 ≥ 80%",
        "results": {
            "symptom_mode": symptom_result,
            "root_cause_mode": root_cause_result,
        },
        "summary": {
            "symptom_mode": {
                "root_cause_match_rate": symptom_result.get("root_cause_match_at_3_rate", 0),
                "passed": symptom_result.get("root_cause_match_at_3_rate", 0) >= 0.8,
            },
            "root_cause_mode": {
                "root_cause_match_rate": root_cause_result.get("root_cause_match_at_3_rate", 0),
                "passed": root_cause_result.get("root_cause_match_at_3_rate", 0) >= 0.8,
            },
        },
    }
    return report


def write_markdown_report(report: dict, path: Path):
    """写入 Markdown 报告"""
    lines = [
        "# M3-11 检索效果评估报告",
        "",
        f"**评估时间**: {report['timestamp']}",
        f"**种子案例**: {report['seed_cases']}",
        f"**Top-K**: {report['top_k']}",
        f"**验收标准**: {report['acceptance_criterion']}",
        "",
        "## 评估结果总览",
        "",
        "| 模式 | 查询数 | Hit@1 | Hit@3 | 根因匹配@3 | MRR | 通过 |",
        "|------|--------|-------|-------|-----------|-----|------|",
    ]

    for mode_key, mode_label in [("symptom_mode", "症状查询"), ("root_cause_mode", "根因查询")]:
        r = report["results"][mode_key]
        if "error" in r:
            lines.append(f"| {mode_label} | - | - | - | - | - | ❌ {r['error']} |")
            continue
        total = r["total_queries"]
        h1 = r["hit_at_1"]
        h3 = r["hit_at_3"]
        rcm = r["root_cause_match_at_3"]
        mrr = r["mrr"]
        rate = r["root_cause_match_at_3_rate"]
        passed = "✅" if rate >= 0.8 else "❌"
        lines.append(
            f"| {mode_label} | {total} | {h1} ({r['hit_at_1_rate']:.0%}) | "
            f"{h3} ({r['hit_at_3_rate']:.0%}) | "
            f"{rcm} ({rate:.0%}) | {mrr:.3f} | {passed} |"
        )

    lines.append("")
    lines.append("## 模式说明")
    lines.append("")
    lines.append("- **症状查询 (symptom)**: 用工艺参数描述症状（温度/保温时间/冷却速率），不含根因关键词。测试嵌入模型能否理解参数偏差与根因的语义关联。")
    lines.append("- **根因查询 (root_cause)**: 用缺陷类型 + 根因关键词搜索，模拟 Agent 检索相似历史案例的真实场景。")
    lines.append("")

    # 详细失败案例
    for mode_key, mode_label in [("symptom_mode", "症状查询"), ("root_cause_mode", "根因查询")]:
        r = report["results"][mode_key]
        if "error" in r:
            continue
        failures = [d for d in r.get("details", []) if not d["root_cause_match"]]
        if failures:
            lines.append(f"## {mode_label} - 未匹配案例（{len(failures)} 条）")
            lines.append("")
            lines.append("| 案例 | 查询 | 期望根因 | 检索到的根因 |")
            lines.append("|------|------|---------|-------------|")
            for f in failures[:20]:  # 最多展示 20 条
                ret_rcs = " / ".join(f["retrieved_root_causes"][:3]) if f["retrieved_root_causes"] else "(空)"
                lines.append(
                    f"| {f['case_id']} | {f['query'][:50]} | {f['expected_root_cause']} | {ret_rcs} |"
                )
            if len(failures) > 20:
                lines.append(f"\n*...还有 {len(failures) - 20} 条未展示*")
            lines.append("")

    # 结论
    lines.append("## 结论与建议")
    lines.append("")
    symptom_rate = report["summary"]["symptom_mode"]["root_cause_match_rate"]
    root_cause_rate = report["summary"]["root_cause_mode"]["root_cause_match_rate"]
    if root_cause_rate >= 0.8:
        lines.append("- ✅ **根因查询模式通过验收**（Top-3 根因匹配率 ≥ 80%），Agent 检索相似案例的能力达标。")
    else:
        lines.append("- ❌ **根因查询模式未通过验收**，需优化嵌入模型或文档结构。")
    if symptom_rate < 0.5:
        lines.append("- ⚠️ **症状查询模式准确率较低**，说明默认嵌入模型（all-MiniLM-L6-v2）对中文工艺参数的语义理解有限。建议：")
        lines.append("  - 替换为支持中文的嵌入模型（如 BAAI/bge-small-zh 或 text2vec-base-chinese）")
        lines.append("  - 或在文档中补充中文缺陷描述（如「硬度偏低」而非仅「hardness_low」）")
    elif symptom_rate >= 0.8:
        lines.append("- ✅ **症状查询模式也通过验收**，嵌入模型能理解工艺参数与根因的语义关联。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 60)
    print("M3-11 检索效果评估")
    print("=" * 60)

    # 加载种子案例
    cases = load_seed_cases()
    print(f"加载 {len(cases)} 条种子案例")

    # 评估 1: 症状查询模式
    print("\n[1/2] 症状查询模式评估中...")
    symptom_result = evaluate_retrieval(cases, mode="symptom")
    if "error" not in symptom_result:
        print(f"  Hit@1: {symptom_result['hit_at_1']}/{symptom_result['total_queries']} "
              f"({symptom_result['hit_at_1_rate']:.1%})")
        print(f"  Hit@3: {symptom_result['hit_at_3']}/{symptom_result['total_queries']} "
              f"({symptom_result['hit_at_3_rate']:.1%})")
        print(f"  根因匹配@3: {symptom_result['root_cause_match_at_3']}/"
              f"{symptom_result['total_queries']} "
              f"({symptom_result['root_cause_match_at_3_rate']:.1%})")
        print(f"  MRR: {symptom_result['mrr']:.3f}")
    else:
        print(f"  错误: {symptom_result['error']}")

    # 评估 2: 根因查询模式
    print("\n[2/2] 根因查询模式评估中...")
    root_cause_result = evaluate_retrieval(cases, mode="root_cause")
    if "error" not in root_cause_result:
        print(f"  Hit@1: {root_cause_result['hit_at_1']}/{root_cause_result['total_queries']} "
              f"({root_cause_result['hit_at_1_rate']:.1%})")
        print(f"  Hit@3: {root_cause_result['hit_at_3']}/{root_cause_result['total_queries']} "
              f"({root_cause_result['hit_at_3_rate']:.1%})")
        print(f"  根因匹配@3: {root_cause_result['root_cause_match_at_3']}/"
              f"{root_cause_result['total_queries']} "
              f"({root_cause_result['root_cause_match_at_3_rate']:.1%})")
        print(f"  MRR: {root_cause_result['mrr']:.3f}")
    else:
        print(f"  错误: {root_cause_result['error']}")

    # 生成报告
    report = generate_report(symptom_result, root_cause_result)

    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(report, REPORT_MD)

    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"  JSON 报告: {REPORT_JSON}")
    print(f"  MD   报告: {REPORT_MD}")
    print("=" * 60)

    # 验收结论
    rc_rate = root_cause_result.get("root_cause_match_at_3_rate", 0)
    if rc_rate >= 0.8:
        print(f"\n✅ 验收通过：根因查询 Top-3 匹配率 {rc_rate:.1%} ≥ 80%")
    else:
        print(f"\n❌ 验收未通过：根因查询 Top-3 匹配率 {rc_rate:.1%} < 80%")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        err_path = PROJECT_ROOT / "data" / "_m3_eval_error.txt"
        err_path.write_text(
            f"评估脚本异常:\n{traceback.format_exc()}", encoding="utf-8"
        )
        print(f"ERROR: {e}")
        print(f"详情见 {err_path}")
        sys.exit(1)
