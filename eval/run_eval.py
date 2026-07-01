"""
MetaCraft Agent 跑批评估脚本（M1-24 / M2-14）

对 seed_cases.json 中的 50 条用例逐条运行 Agent，
对比预测根因与期望根因，计算准确率。

准确率指标：
- 宽松准确率：命中至少 1 个期望关键词即算正确
- 严格准确率：命中全部期望关键词才算正确
- 平均命中率：命中关键词数 / 期望关键词数

用法：
    python eval/run_eval.py                      # 跑全量 50 条（M2 默认流程）
    python eval/run_eval.py --limit 5            # 只跑前 5 条（快速验证）
    python eval/run_eval.py --ids SC-001 SC-002  # 指定 case_id
    python eval/run_eval.py --flow sequential    # 用 M1 线性流程
    python eval/run_eval.py --flow parallel      # 用 M2 并行流程（默认）
    python eval/run_eval.py --resume             # 跳过已完成的用例（基于增量结果文件）
"""
import argparse
import asyncio
import io
import json
import sys
import time
import traceback
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from agent.orchestrator import build_orchestrator
from agent.utils import setup_tracing
from models.state import AgentState


# ===== 日志配置 =====
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{message}</cyan>",
    level="WARNING",  # 评估时只看 WARNING 以上，减少噪音
)
logger.add(
    "data/eval.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
    level="DEBUG",
    rotation="10 MB",
)


SEED_CASES_PATH = PROJECT_ROOT / "data" / "seed_cases" / "seed_cases.json"

# M2-14: 报告路径根据 flow_name 区分
def _get_report_paths(flow_name: str) -> tuple[Path, Path, Path]:
    """获取报告路径（JSON / 文本 / 增量）

    M1 baseline 报告为 eval_report.json（不覆盖）。
    M2 评估报告根据 flow_name 加后缀。
    """
    if flow_name == "parallel":
        suffix = "_m2"  # M2 默认并行模式，区别于 M1 baseline
    else:
        suffix = f"_{flow_name}"
    return (
        PROJECT_ROOT / "data" / f"eval_report{suffix}.json",
        PROJECT_ROOT / "data" / f"eval_report{suffix}.txt",
        PROJECT_ROOT / "data" / f"eval_incremental{suffix}.json",
    )

# 默认路径（兼容旧代码）
REPORT_PATH, _, INCREMENTAL_PATH = _get_report_paths("parallel")


def load_seed_cases() -> list[dict]:
    """加载种子案例"""
    with open(SEED_CASES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def create_initial_state(case: dict) -> AgentState:
    """根据 case 构造 Agent 初始状态

    评估模式下 max_replan=1（最多重试 1 次），避免审核严格导致每条用例耗时过长。
    生产模式建议 max_replan=3。
    """
    trace_id = f"eval_{case['case_id']}_{uuid.uuid4().hex[:6]}"
    return {
        "user_query": case["query"],
        "batch_id": case["batch_id"],
        "defect_record": None,
        "plan": [],
        "current_step": 0,
        "observations": [],
        "data_result": None,
        "mechanism_result": None,
        "knowledge_result": None,
        "arbitration_result": None,  # M2 新增
        "decision_result": None,
        "review_result": None,
        "proposal": None,
        "final_answer": None,
        "retry_count": 0,
        "needs_replan": False,
        "max_replan": 1,  # 评估模式：最多重试 1 次
        "trace_id": trace_id,
        "session_id": trace_id,
    }


def extract_prediction_text(final_state: dict) -> str:
    """从 Agent 最终状态提取用于匹配的文本

    拼接 decision_result.proposals 的 root_cause + adjustments + expected_effect
    + evidence + final_answer，作为匹配文本。
    M2 迭代优化：加入 evidence 字段，扩展关键词匹配范围。
    """
    parts = []

    decision = final_state.get("decision_result") or {}
    for p in decision.get("proposals", []):
        parts.append(str(p.get("root_cause", "")))
        parts.append(str(p.get("expected_effect", "")))
        # M2 迭代优化：加入 evidence 字段
        evidence = p.get("evidence") or []
        if isinstance(evidence, list):
            parts.extend(str(e) for e in evidence)
        elif isinstance(evidence, str):
            parts.append(evidence)
        adjustments = p.get("adjustments") or {}
        for k, v in adjustments.items():
            parts.append(f"{k} {v}")

    answer = final_state.get("final_answer") or ""
    parts.append(answer)

    return " ".join(parts)


def match_keywords(pred_text: str, expected_keywords: list[str]) -> tuple[int, int, list[str]]:
    """关键词匹配

    Args:
        pred_text: Agent 预测文本
        expected_keywords: 期望关键词列表

    Returns:
        (命中数, 期望总数, 命中的关键词列表)
    """
    pred_lower = pred_text.lower()
    hit = []
    for kw in expected_keywords:
        if kw.lower() in pred_lower:
            hit.append(kw)
    return len(hit), len(expected_keywords), hit


def _save_incremental(results: list[dict], incremental_path: Path = None):
    """增量保存结果，每跑完一条就写一次，避免中途崩溃丢失数据"""
    path = incremental_path or INCREMENTAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2, default=str)


def _load_incremental(incremental_path: Path = None) -> dict:
    """加载增量结果（用于 --resume）"""
    path = incremental_path or INCREMENTAL_PATH
    if not path.exists():
        return {"results": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载增量结果失败: {e}")
        return {"results": []}


async def run_single_case(case: dict, idx: int, total: int, flow_name: str = None) -> dict:
    """跑单条用例

    每条用例构建全新的 orchestrator，避免单例的 MemorySaver 状态污染。

    Args:
        flow_name: 协作流程名（M2-14），None 时使用默认流程

    Returns:
        评估结果字典
    """
    case_id = case["case_id"]
    batch_id = case["batch_id"]
    expected_root_cause = case["expected_root_cause"]
    expected_keywords = case.get("expected_keywords", [])
    expected_adjustment = case.get("expected_adjustment", "")

    start = time.time()
    # 用 ASCII 标记避免 PowerShell 编码问题
    print(f"[{idx}/{total}] {case_id} ({batch_id}) start... expected: {expected_root_cause}", flush=True)

    initial_state = create_initial_state(case)
    config = {"configurable": {"thread_id": initial_state["trace_id"]}}

    result = {
        "case_id": case_id,
        "batch_id": batch_id,
        "difficulty": case.get("difficulty", "?"),
        "expected_root_cause": expected_root_cause,
        "expected_keywords": expected_keywords,
        "expected_adjustment": expected_adjustment,
        "batch_params": case.get("batch_params", {}),
    }

    try:
        # 每条用例构建全新 orchestrator，避免 MemorySaver 状态污染
        orchestrator = build_orchestrator(flow_name)
        # M2 迭代修复：添加 300s 超时，防止 LLM 调用无限卡住
        final_state = await asyncio.wait_for(
            orchestrator.ainvoke(initial_state, config),
            timeout=300,
        )
        elapsed = round(time.time() - start, 1)

        pred_text = extract_prediction_text(final_state)
        hit_count, total_kw, hit_kws = match_keywords(pred_text, expected_keywords)

        decision = final_state.get("decision_result") or {}
        review = final_state.get("review_result") or {}
        proposals = decision.get("proposals", [])

        # 提取预测的根因（取第一个方案的 root_cause 作为主预测）
        pred_root_cause = proposals[0].get("root_cause", "") if proposals else ""
        pred_confidence = proposals[0].get("confidence", 0) if proposals else 0

        # 宽松准确率：命中 ≥1 个关键词
        loose_correct = hit_count > 0
        # 严格准确率：命中全部关键词
        strict_correct = hit_count == total_kw and total_kw > 0
        # 命中率
        hit_rate = hit_count / total_kw if total_kw > 0 else 0.0

        result.update({
            "status": "success",
            "elapsed_seconds": elapsed,
            "pred_root_cause": pred_root_cause,
            "pred_confidence": pred_confidence,
            "proposals_count": len(proposals),
            "source": decision.get("source", "unknown"),
            "review_approved": review.get("approved", False),
            "retry_count": final_state.get("retry_count", 0),
            "hit_keywords": hit_kws,
            "hit_count": hit_count,
            "total_keywords": total_kw,
            "hit_rate": round(hit_rate, 2),
            "loose_correct": loose_correct,
            "strict_correct": strict_correct,
            "final_answer_preview": (final_state.get("final_answer") or "")[:200],
        })

        marker = "[OK]" if loose_correct else "[FAIL]"
        print(f"         {marker} hit {hit_count}/{total_kw} ({hit_rate:.0%}) "
              f"| pred: {pred_root_cause[:40]} | elapsed {elapsed}s", flush=True)

    except asyncio.TimeoutError:
        elapsed = round(time.time() - start, 1)
        logger.error(f"[run_single_case] {case_id} 超时 ({elapsed}s > 300s)")
        result.update({
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": f"timeout ({elapsed}s > 300s limit)",
            "loose_correct": False,
            "strict_correct": False,
            "hit_rate": 0.0,
            "hit_count": 0,
            "total_keywords": len(expected_keywords),
        })
        print(f"         [TIMEOUT] {case_id} exceeded 300s | elapsed {elapsed}s", flush=True)

    except Exception as e:
        elapsed = round(time.time() - start, 1)
        err_trace = traceback.format_exc()
        logger.error(f"[run_single_case] {case_id} 异常: {e}\n{err_trace}")
        result.update({
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(e),
            "error_traceback": err_trace,
            "loose_correct": False,
            "strict_correct": False,
            "hit_rate": 0.0,
            "hit_count": 0,
            "total_keywords": len(expected_keywords),
        })
        print(f"         [ERR] {e} | elapsed {elapsed}s", flush=True)

    return result


async def run_eval(cases: list[dict], concurrency: int = 1, resume_ids: set = None,
                   flow_name: str = None, incremental_path: Path = None,
                   prior_results: list[dict] = None) -> list[dict]:
    """跑批评估

    Args:
        cases: 用例列表
        concurrency: 并发数
        resume_ids: 需要跳过的 case_id 集合（用于 --resume）
        flow_name: 协作流程名（M2-14）
        incremental_path: 增量保存路径
        prior_results: resume 模式下已完成的旧结果（避免增量保存时覆盖）

    Returns:
        全部用例的评估结果（含 prior_results + 本次新结果）
    """
    setup_tracing()

    total = len(cases)
    inc_path = incremental_path or INCREMENTAL_PATH
    print(f"\n{'='*60}", flush=True)
    print(f"  start eval: {total} cases, concurrency={concurrency}, flow={flow_name or 'default'}", flush=True)
    if resume_ids:
        print(f"  (skip {len(resume_ids)} already-done cases)", flush=True)
    print(f"{'='*60}\n", flush=True)

    # 串行执行（评估场景下并发容易触发 SiliconFlow 限流，且便于增量保存）
    # M2 迭代修复：resume 模式下初始化 results 为 prior_results，
    # 避免增量保存时只写新结果、覆盖旧结果导致数据丢失。
    results = list(prior_results) if prior_results else []
    for idx, case in enumerate(cases, 1):
        # resume 模式：跳过已完成的用例
        if resume_ids and case["case_id"] in resume_ids:
            print(f"[{idx}/{total}] {case['case_id']} SKIP (already done)", flush=True)
            continue

        r = await run_single_case(case, idx, total, flow_name=flow_name)
        results.append(r)

        # 增量保存：每跑完一条就写一次（含 prior_results，避免覆盖）
        _save_incremental(results, inc_path)

        # 显式打印进度分隔
        done_count = len(results)
        print(f"  -> {done_count}/{total} done, saved to {inc_path.name}", flush=True)

    return results


def compute_summary(results: list[dict]) -> dict:
    """计算汇总指标"""
    total = len(results)
    if total == 0:
        return {}

    success = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") == "error"]

    loose_correct = sum(1 for r in success if r.get("loose_correct"))
    strict_correct = sum(1 for r in success if r.get("strict_correct"))
    avg_hit_rate = sum(r.get("hit_rate", 0) for r in success) / max(len(success), 1)
    avg_elapsed = sum(r.get("elapsed_seconds", 0) for r in success) / max(len(success), 1)
    avg_retries = sum(r.get("retry_count", 0) for r in success) / max(len(success), 1)
    llm_source_count = sum(1 for r in success if r.get("source") == "llm")
    approved_count = sum(1 for r in success if r.get("review_approved"))

    # 按类别统计
    category_stats = {}
    for r in success:
        cat = r.get("expected_root_cause", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "loose": 0, "strict": 0, "hit_rates": []}
        category_stats[cat]["total"] += 1
        if r.get("loose_correct"):
            category_stats[cat]["loose"] += 1
        if r.get("strict_correct"):
            category_stats[cat]["strict"] += 1
        category_stats[cat]["hit_rates"].append(r.get("hit_rate", 0))

    for cat, s in category_stats.items():
        s["loose_acc"] = round(s["loose"] / max(s["total"], 1), 2)
        s["strict_acc"] = round(s["strict"] / max(s["total"], 1), 2)
        s["avg_hit_rate"] = round(sum(s["hit_rates"]) / max(len(s["hit_rates"]), 1), 2)
        del s["hit_rates"]

    # 按难度统计
    difficulty_stats = {}
    for r in success:
        diff = r.get("difficulty", "?")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "loose": 0}
        difficulty_stats[diff]["total"] += 1
        if r.get("loose_correct"):
            difficulty_stats[diff]["loose"] += 1
    for diff, s in difficulty_stats.items():
        s["loose_acc"] = round(s["loose"] / max(s["total"], 1), 2)

    return {
        "total_cases": total,
        "success_count": len(success),
        "error_count": len(errors),
        "loose_accuracy": round(loose_correct / max(len(success), 1), 4),
        "strict_accuracy": round(strict_correct / max(len(success), 1), 4),
        "avg_hit_rate": round(avg_hit_rate, 4),
        "avg_elapsed_seconds": round(avg_elapsed, 1),
        "avg_retry_count": round(avg_retries, 2),
        "llm_source_rate": round(llm_source_count / max(len(success), 1), 4),
        "review_approval_rate": round(approved_count / max(len(success), 1), 4),
        "category_stats": category_stats,
        "difficulty_stats": difficulty_stats,
    }


def print_summary(summary: dict, results: list[dict]):
    """打印汇总报告（ASCII-only 输出，避免编码问题）"""
    print(f"\n{'='*60}")
    print(f"  EVAL SUMMARY REPORT")
    print(f"{'='*60}")

    print(f"\n[Overall]")
    print(f"  total cases:      {summary['total_cases']}")
    print(f"  success:          {summary['success_count']}")
    print(f"  errors:           {summary['error_count']}")
    print(f"  loose accuracy:   {summary['loose_accuracy']:.1%}  (hit >=1 keyword)")
    print(f"  strict accuracy:  {summary['strict_accuracy']:.1%}  (hit ALL keywords)")
    print(f"  avg hit rate:     {summary['avg_hit_rate']:.1%}  (hit/expected)")
    print(f"  avg elapsed:      {summary['avg_elapsed_seconds']}s/case")
    print(f"  avg retries:      {summary['avg_retry_count']}")
    print(f"  LLM source rate:  {summary['llm_source_rate']:.1%}")
    print(f"  review approval:  {summary['review_approval_rate']:.1%}")

    print(f"\n[By Root Cause]")
    print(f"  {'root_cause':<30s} {'total':>5s} {'loose':>6s} {'strict':>6s} {'hit_rate':>8s}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*8}")
    for cat, s in sorted(summary["category_stats"].items()):
        print(f"  {cat:<28s} {s['total']:>5d} {s['loose_acc']:>5.0%} {s['strict_acc']:>5.0%} {s['avg_hit_rate']:>7.0%}")

    print(f"\n[By Difficulty]")
    print(f"  {'difficulty':<10s} {'total':>5s} {'loose_acc':>10s}")
    print(f"  {'-'*10} {'-'*5} {'-'*10}")
    for diff, s in sorted(summary["difficulty_stats"].items()):
        print(f"  {diff:<10s} {s['total']:>5d} {s['loose_acc']:>9.0%}")

    # 错误用例
    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        print(f"\n[Error Cases]")
        for r in errors:
            print(f"  {r['case_id']} ({r['batch_id']}): {r.get('error', 'unknown')}")

    # 宽松准确率未命中的用例
    misses = [r for r in results if r.get("status") == "success" and not r.get("loose_correct")]
    if misses:
        print(f"\n[Missed All Keywords]")
        for r in misses:
            print(f"  {r['case_id']} ({r['batch_id']})")
            print(f"    expected: {r['expected_root_cause']}")
            print(f"    predicted: {r.get('pred_root_cause', '?')}")
            print(f"    expected_keywords: {r['expected_keywords']}")
            print(f"    preview: {r.get('final_answer_preview', '')[:100]}")

    print(f"\n{'='*60}")
    print(f"  EVAL DONE")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="MetaCraft Agent 跑批评估")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    parser.add_argument("--ids", nargs="*", default=[], help="指定 case_id 列表")
    parser.add_argument("--concurrency", type=int, default=1, help="并发数（默认 1）")
    parser.add_argument("--no-save", action="store_true", help="不保存报告文件")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的用例（基于增量结果文件）")
    parser.add_argument("--flow", type=str, default="parallel",
                        help="协作流程名（parallel/sequential/data_first/quick/knowledge_heavy，默认 parallel）")
    args = parser.parse_args()

    flow_name = args.flow
    report_path, text_report_path, incremental_path = _get_report_paths(flow_name)
    print(f"flow: {flow_name}", flush=True)
    print(f"report paths: {report_path.name}, {text_report_path.name}, {incremental_path.name}", flush=True)

    # 加载用例
    all_cases = load_seed_cases()
    print(f"loaded {len(all_cases)} seed cases", flush=True)

    # 筛选用例
    if args.ids:
        id_set = set(args.ids)
        cases = [c for c in all_cases if c["case_id"] in id_set]
        print(f"filter by ids: {len(cases)} cases", flush=True)
    elif args.limit > 0:
        cases = all_cases[:args.limit]
        print(f"take first {args.limit}", flush=True)
    else:
        cases = all_cases

    if not cases:
        print("no cases to run")
        return

    # resume 模式：加载已完成的结果，跳过这些用例
    resume_ids = set()
    prior_results = []
    if args.resume:
        incremental = _load_incremental(incremental_path)
        prior_results = incremental.get("results", [])
        resume_ids = {r["case_id"] for r in prior_results if r.get("status") == "success"}
        print(f"resume mode: {len(resume_ids)} cases already done, will skip", flush=True)

    # 跑批
    start_time = time.time()
    # M2 迭代修复：传入 prior_results，让 run_eval 增量保存时包含旧结果
    all_results = asyncio.run(run_eval(cases, concurrency=args.concurrency, resume_ids=resume_ids,
                                       flow_name=flow_name, incremental_path=incremental_path,
                                       prior_results=prior_results))
    total_elapsed = round(time.time() - start_time, 1)

    # 汇总
    summary = compute_summary(all_results)
    summary["total_elapsed_seconds"] = total_elapsed
    summary["flow_name"] = flow_name

    # 捕获 print_summary 输出，同时写终端和文件
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_summary(summary, all_results)
        print(f"total elapsed: {total_elapsed}s")
        print(f"flow: {flow_name}")
    report_text = buffer.getvalue()
    print(report_text, end="", flush=True)

    # 保存报告（JSON + 文本）
    if not args.no_save:
        report = {
            "summary": summary,
            "results": all_results,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"JSON report: {report_path}", flush=True)
        print(f"Text report: {text_report_path}", flush=True)

        # 清理增量文件（全量报告已保存）
        if incremental_path.exists():
            incremental_path.unlink()
            print(f"cleaned incremental file: {incremental_path.name}", flush=True)


if __name__ == "__main__":
    main()
