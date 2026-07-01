"""
评估报告对比脚本

对比两份 eval 报告（旧 prompt vs 新 prompt），输出：
1. 汇总指标 delta 表（loose_accuracy / strict_accuracy / review_approval_rate / avg_retry_count 等）
2. 按 root_cause 分类的指标对比
3. 按难度的指标对比
4. 逐案差异：loose_correct / strict_correct / retry_count / review_approved 翻转的用例

用法：
    python eval/compare_eval.py data/eval_report_before_prompt_opt.json data/eval_report.json
    python eval/compare_eval.py --old data/eval_report_before_prompt_opt.json --new data/eval_incremental.json
    python eval/compare_eval.py --old A.json --new B.json --output data/eval_compare.md
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run_eval import compute_summary


def load_report(path: str) -> dict:
    """加载报告，若没有 summary 字段（如 incremental 文件）则计算"""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if "summary" not in data or not data["summary"]:
        data["summary"] = compute_summary(results)
        print(f"[INFO] {path} 无 summary 字段，已重新计算（{len(results)} 条）", file=sys.stderr)
    return data


def _pct(x: float) -> str:
    """百分比格式化"""
    return f"{x:.1%}"


def _delta(old: float, new: float, higher_better: bool = True) -> str:
    """计算 delta，标注改善/恶化"""
    diff = new - old
    sign = "+" if diff >= 0 else ""
    arrow = ""
    if abs(diff) > 1e-6:
        if (diff > 0) == higher_better:
            arrow = " ↓改善"
        else:
            arrow = " ↑恶化"
    return f"{sign}{diff:.4f}{arrow}"


def _delta_pct(old: float, new: float, higher_better: bool = True) -> str:
    """百分比 delta"""
    diff = new - old
    sign = "+" if diff >= 0 else ""
    arrow = ""
    if abs(diff) > 1e-6:
        if (diff > 0) == higher_better:
            arrow = " ↓改善"
        else:
            arrow = " ↑恶化"
    return f"{sign}{_pct(diff)}{arrow}"


def compare_summaries(old_sum: dict, new_sum: dict) -> str:
    """对比汇总指标"""
    lines = []
    lines.append("## 1. 汇总指标对比\n")
    lines.append("| 指标 | 旧 prompt | 新 prompt | Delta | 判定 |")
    lines.append("|---|---|---|---|---|")

    rows = [
        ("用例总数",        old_sum.get("total_cases", 0),         new_sum.get("total_cases", 0),         "count",  None),
        ("成功用例数",      old_sum.get("success_count", 0),      new_sum.get("success_count", 0),      "count",  None),
        ("错误用例数",      old_sum.get("error_count", 0),        new_sum.get("error_count", 0),        "count",  False),
        ("宽松准确率",      old_sum.get("loose_accuracy", 0),     new_sum.get("loose_accuracy", 0),     "pct",    True),
        ("严格准确率",      old_sum.get("strict_accuracy", 0),    new_sum.get("strict_accuracy", 0),    "pct",    True),
        ("平均命中率",      old_sum.get("avg_hit_rate", 0),       new_sum.get("avg_hit_rate", 0),       "pct",    True),
        ("平均耗时(s)",     old_sum.get("avg_elapsed_seconds", 0),new_sum.get("avg_elapsed_seconds", 0),"float",  False),
        ("平均 retry 次数", old_sum.get("avg_retry_count", 0),    new_sum.get("avg_retry_count", 0),    "float",  False),
        ("LLM 来源率",      old_sum.get("llm_source_rate", 0),    new_sum.get("llm_source_rate", 0),    "pct",    None),
        ("审核通过率",      old_sum.get("review_approval_rate", 0),new_sum.get("review_approval_rate", 0),"pct",  True),
    ]

    for name, o, n, kind, higher_better in rows:
        if kind == "count":
            diff = n - o
            sign = "+" if diff >= 0 else ""
            delta_str = f"{sign}{diff}"
        elif kind == "pct":
            delta_str = _delta_pct(o, n, higher_better=True) if higher_better else _delta_pct(o, n, higher_better=False)
            o_str = _pct(o)
            n_str = _pct(n)
        elif kind == "float":
            diff = n - o
            sign = "+" if diff >= 0 else ""
            delta_str = f"{sign}{diff:.2f}"
            if higher_better is True and diff < -1e-6:
                delta_str += " ↓改善"
            elif higher_better is False and diff > 1e-6:
                delta_str += " ↑恶化"
            elif higher_better is True and diff > 1e-6:
                delta_str += " ↑恶化"
            elif higher_better is False and diff < -1e-6:
                delta_str += " ↓改善"
            o_str = f"{o:.2f}"
            n_str = f"{n:.2f}"

        if kind == "count":
            o_str = str(o)
            n_str = str(n)

        # 判定列
        if higher_better is None:
            judge = "—"
        elif kind == "count" and name in ("错误用例数",):
            judge = "改善" if diff < 0 else ("恶化" if diff > 0 else "持平")
        elif kind == "pct" or kind == "float":
            diff_val = n - o
            if abs(diff_val) < 1e-6:
                judge = "持平"
            elif (diff_val > 0) == higher_better:
                judge = "改善"
            else:
                judge = "恶化"

        lines.append(f"| {name} | {o_str} | {n_str} | {delta_str} | {judge} |")

    return "\n".join(lines) + "\n"


def compare_category(old_sum: dict, new_sum: dict) -> str:
    """按 root_cause 分类对比"""
    lines = []
    lines.append("\n## 2. 按 root_cause 分类对比\n")
    lines.append("| 根因 | 旧 total | 新 total | 旧 loose | 新 loose | 旧 strict | 新 strict | 旧 hit_rate | 新 hit_rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    old_cat = old_sum.get("category_stats", {})
    new_cat = new_sum.get("category_stats", {})
    all_cats = sorted(set(old_cat.keys()) | set(new_cat.keys()))

    for cat in all_cats:
        o = old_cat.get(cat, {"total": 0, "loose_acc": 0, "strict_acc": 0, "avg_hit_rate": 0})
        n = new_cat.get(cat, {"total": 0, "loose_acc": 0, "strict_acc": 0, "avg_hit_rate": 0})
        lines.append(
            f"| {cat} | {o['total']} | {n['total']} | "
            f"{_pct(o['loose_acc'])} | {_pct(n['loose_acc'])} | "
            f"{_pct(o['strict_acc'])} | {_pct(n['strict_acc'])} | "
            f"{_pct(o['avg_hit_rate'])} | {_pct(n['avg_hit_rate'])} |"
        )

    return "\n".join(lines) + "\n"


def compare_difficulty(old_sum: dict, new_sum: dict) -> str:
    """按难度对比"""
    lines = []
    lines.append("\n## 3. 按难度对比\n")
    lines.append("| 难度 | 旧 total | 新 total | 旧 loose_acc | 新 loose_acc | Delta |")
    lines.append("|---|---|---|---|---|---|")

    old_diff = old_sum.get("difficulty_stats", {})
    new_diff = new_sum.get("difficulty_stats", {})
    all_diffs = sorted(set(old_diff.keys()) | set(new_diff.keys()))

    for diff in all_diffs:
        o = old_diff.get(diff, {"total": 0, "loose_acc": 0})
        n = new_diff.get(diff, {"total": 0, "loose_acc": 0})
        delta = _delta_pct(o["loose_acc"], n["loose_acc"], higher_better=True)
        lines.append(
            f"| {diff} | {o['total']} | {n['total']} | "
            f"{_pct(o['loose_acc'])} | {_pct(n['loose_acc'])} | {delta} |"
        )

    return "\n".join(lines) + "\n"


def compare_per_case(old_results: list, new_results: list) -> str:
    """逐案对比：找出翻转的用例"""
    lines = []
    lines.append("\n## 4. 逐案差异（仅显示有变化的用例）\n")

    old_by_id = {r["case_id"]: r for r in old_results}
    new_by_id = {r["case_id"]: r for r in new_results}
    all_ids = sorted(set(old_by_id.keys()) | set(new_by_id.keys()))

    flipped = []
    for cid in all_ids:
        o = old_by_id.get(cid)
        n = new_by_id.get(cid)
        if not o or not n:
            continue
        diffs = []
        # loose_correct 翻转
        if o.get("loose_correct") != n.get("loose_correct"):
            diffs.append(
                f"loose: {o.get('loose_correct')} → {n.get('loose_correct')} "
                f"({'改善' if n.get('loose_correct') else '恶化'})"
            )
        # strict_correct 翻转
        if o.get("strict_correct") != n.get("strict_correct"):
            diffs.append(
                f"strict: {o.get('strict_correct')} → {n.get('strict_correct')} "
                f"({'改善' if n.get('strict_correct') else '恶化'})"
            )
        # retry_count 变化
        if o.get("retry_count", 0) != n.get("retry_count", 0):
            diffs.append(
                f"retry: {o.get('retry_count', 0)} → {n.get('retry_count', 0)} "
                f"({'改善' if n.get('retry_count', 0) < o.get('retry_count', 0) else '恶化'})"
            )
        # review_approved 翻转
        if o.get("review_approved") != n.get("review_approved"):
            diffs.append(
                f"review: {o.get('review_approved')} → {n.get('review_approved')} "
                f"({'改善' if n.get('review_approved') else '恶化'})"
            )
        # hit_rate 变化（>0.1 才显示）
        if abs(o.get("hit_rate", 0) - n.get("hit_rate", 0)) >= 0.1:
            diffs.append(
                f"hit_rate: {o.get('hit_rate', 0):.2f} → {n.get('hit_rate', 0):.2f}"
            )

        if diffs:
            flipped.append((cid, o, n, diffs))

    if not flipped:
        lines.append("（无翻转用例，所有用例指标完全一致）\n")
        return "\n".join(lines)

    lines.append(f"共 {len(flipped)} 条用例有变化：\n")
    lines.append("| case_id | 期望根因 | 变化项 |")
    lines.append("|---|---|---|")
    for cid, o, n, diffs in flipped:
        expected = o.get("expected_root_cause", "?")
        diff_str = "<br>".join(diffs)
        lines.append(f"| {cid} | {expected} | {diff_str} |")

    return "\n".join(lines) + "\n"


def compare_review_pattern(old_results: list, new_results: list) -> str:
    """对比审核通过模式"""
    lines = []
    lines.append("\n## 5. 审核通过模式对比\n")

    def _pattern(results):
        succ = [r for r in results if r.get("status") == "success"]
        approved = [r for r in succ if r.get("review_approved")]
        rejected = [r for r in succ if not r.get("review_approved")]
        return {
            "total": len(succ),
            "approved": len(approved),
            "rejected": len(rejected),
            "approval_rate": len(approved) / max(len(succ), 1),
            "approved_retry0": sum(1 for r in approved if r.get("retry_count", 0) == 0),
            "approved_retry1": sum(1 for r in approved if r.get("retry_count", 0) == 1),
            "rejected_retry0": sum(1 for r in rejected if r.get("retry_count", 0) == 0),
            "rejected_retry1": sum(1 for r in rejected if r.get("retry_count", 0) == 1),
        }

    o = _pattern(old_results)
    n = _pattern(new_results)

    lines.append("| 指标 | 旧 prompt | 新 prompt | Delta |")
    lines.append("|---|---|---|---|")
    for k, label in [
        ("total",            "成功用例数"),
        ("approved",         "审核通过数"),
        ("rejected",         "审核拒绝数"),
        ("approval_rate",    "审核通过率"),
        ("approved_retry0",  "通过且 retry=0"),
        ("approved_retry1",  "通过且 retry=1（重试后通过）"),
        ("rejected_retry0",  "拒绝且 retry=0"),
        ("rejected_retry1",  "拒绝且 retry=1（重试后仍拒绝）"),
    ]:
        ov = o[k]
        nv = n[k]
        if k == "approval_rate":
            delta = _delta_pct(ov, nv, higher_better=True)
            ov_str = _pct(ov)
            nv_str = _pct(nv)
        else:
            diff = nv - ov
            sign = "+" if diff >= 0 else ""
            delta = f"{sign}{diff}"
            ov_str = str(ov)
            nv_str = str(nv)
        lines.append(f"| {label} | {ov_str} | {nv_str} | {delta} |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="对比两份 eval 报告")
    parser.add_argument("old", nargs="?", help="旧报告路径")
    parser.add_argument("new", nargs="?", help="新报告路径")
    parser.add_argument("--old", dest="old_opt", help="旧报告路径")
    parser.add_argument("--new", dest="new_opt", help="新报告路径")
    parser.add_argument("--output", "-o", default=None, help="输出 markdown 文件路径（不指定则打印到 stdout）")
    args = parser.parse_args()

    old_path = args.old or args.old_opt
    new_path = args.new or args.new_opt
    if not old_path or not new_path:
        parser.error("必须提供 old 和 new 报告路径")

    old_report = load_report(old_path)
    new_report = load_report(new_path)

    old_sum = old_report["summary"]
    new_sum = new_report["summary"]
    old_results = old_report.get("results", [])
    new_results = new_report.get("results", [])

    md_lines = []
    md_lines.append("# 评估报告对比\n")
    md_lines.append(f"- 旧报告: `{old_path}` ({len(old_results)} 条)")
    md_lines.append(f"- 新报告: `{new_path}` ({len(new_results)} 条)")
    md_lines.append("")

    md_lines.append(compare_summaries(old_sum, new_sum))
    md_lines.append(compare_category(old_sum, new_sum))
    md_lines.append(compare_difficulty(old_sum, new_sum))
    md_lines.append(compare_review_pattern(old_results, new_results))
    md_lines.append(compare_per_case(old_results, new_results))

    md = "\n".join(md_lines)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[OK] 对比报告已保存: {out}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
