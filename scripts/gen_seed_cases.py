"""
生成 50 条种子评估案例

覆盖 4 类根因：
- 保温时间不足（15 条）：holding_time < 90
- 冷却速率过低（15 条）：cooling_rate < 5
- 温度偏低（10 条）：temperature < 835
- 复合问题（10 条）：多参数同时异常

每条案例包含 batch_params，与 expected_root_cause 保持一致，
确保 Agent 能基于参数做出正确归因。

运行：
    python scripts/gen_seed_cases.py
"""
import json
import random
from pathlib import Path

random.seed(42)  # 可复现

OUTPUT = Path(__file__).parent.parent / "data" / "seed_cases" / "seed_cases.json"

STANDARD = {
    "temperature": 845,      # 标准 840±10
    "holding_time": 120,     # 标准 ≥ 120
    "cooling_rate": 5.0,     # 标准 ≥ 5
    "standard_hardness": 58.0,
}


def make_case(case_id, batch_id, root_cause, params, hardness_gap, difficulty, adjust_desc, keywords):
    """构造单条用例"""
    measured = round(STANDARD["standard_hardness"] - hardness_gap, 1)
    return {
        "case_id": case_id,
        "batch_id": batch_id,
        "defect_type": "hardness_low",
        "query": f"批次 {batch_id} 硬度偏低 {hardness_gap:.0f} HRc，请分析原因并给出调整建议",
        "measured_value": measured,
        "standard_value": STANDARD["standard_hardness"],
        "batch_params": {
            "batch_id": batch_id,
            "process_type": "heat_treatment",
            **params,
            "raw_material_batch": f"RM-2026-{random.randint(1000, 9999)}",
            "start_time": f"2026-07-{random.randint(1, 28):02d}T08:00:00",
            "end_time": f"2026-07-{random.randint(1, 28):02d}T09:30:00",
        },
        "expected_root_cause": root_cause,
        "expected_adjustment": adjust_desc,
        "expected_keywords": keywords,
        "difficulty": difficulty,
    }


def gen_holding_insufficient(n=15):
    """保温时间不足：holding_time 60-89"""
    cases = []
    for i in range(n):
        idx = i + 1
        ht = random.randint(60, 89)
        gap = round(random.uniform(2.0, 6.0), 1)  # 偏低越多越严重
        difficulty = "easy" if ht < 75 else "medium"
        params = {
            "temperature": random.randint(840, 850),
            "holding_time": ht,
            "cooling_rate": round(random.uniform(5.0, 8.0), 1),
        }
        cases.append(make_case(
            f"SC-{idx:03d}",
            f"B202607{idx:02d}-A",
            "保温时间不足",
            params, gap, difficulty,
            f"holding_time +{STANDARD['holding_time'] - ht + 10}",
            ["保温", "holding_time", "时间不足", "时间不够"],
        ))
    return cases


def gen_cooling_low(n=15):
    """冷却速率过低：cooling_rate 1.0-4.9"""
    cases = []
    for i in range(n):
        idx = i + 16
        cr = round(random.uniform(1.0, 4.9), 1)
        gap = round(random.uniform(3.0, 8.0), 1)
        difficulty = "easy" if cr < 3.0 else "medium"
        params = {
            "temperature": random.randint(840, 850),
            "holding_time": random.randint(120, 150),
            "cooling_rate": cr,
        }
        cases.append(make_case(
            f"SC-{idx:03d}",
            f"B202607{idx:02d}-B",
            "冷却速率过低",
            params, gap, difficulty,
            f"cooling_rate 提升至 {STANDARD['cooling_rate'] + 3.0}",
            ["冷却", "cooling_rate", "冷却速率", "冷速"],
        ))
    return cases


def gen_temp_low(n=10):
    """温度偏低：temperature 800-834"""
    cases = []
    for i in range(n):
        idx = i + 31
        temp = random.randint(800, 834)
        gap = round(random.uniform(2.0, 5.0), 1)
        difficulty = "medium" if temp > 820 else "hard"
        params = {
            "temperature": temp,
            "holding_time": random.randint(120, 150),
            "cooling_rate": round(random.uniform(5.0, 8.0), 1),
        }
        cases.append(make_case(
            f"SC-{idx:03d}",
            f"B202607{idx:02d}-C",
            "温度偏低",
            params, gap, difficulty,
            f"temperature +{STANDARD['temperature'] - temp + 10}",
            ["温度", "temperature", "温度偏低", "淬火温度"],
        ))
    return cases


def gen_composite(n=10):
    """复合问题：多参数同时异常"""
    cases = []
    patterns = [
        ("保温时间不足+冷却速率过低", ["保温", "冷却", "holding_time", "cooling_rate"]),
        ("冷却速率过低+温度偏低", ["冷却", "温度", "cooling_rate", "temperature"]),
        ("保温时间不足+温度偏低", ["保温", "温度", "holding_time", "temperature"]),
    ]
    for i in range(n):
        idx = i + 41
        pattern_name, keywords = patterns[i % len(patterns)]
        gap = round(random.uniform(4.0, 8.0), 1)
        if "保温" in pattern_name and "冷却" in pattern_name:
            params = {
                "temperature": random.randint(840, 850),
                "holding_time": random.randint(60, 89),
                "cooling_rate": round(random.uniform(1.0, 4.5), 1),
            }
        elif "冷却" in pattern_name and "温度" in pattern_name:
            params = {
                "temperature": random.randint(800, 830),
                "holding_time": random.randint(120, 150),
                "cooling_rate": round(random.uniform(1.0, 4.5), 1),
            }
        else:  # 保温+温度
            params = {
                "temperature": random.randint(800, 832),
                "holding_time": random.randint(60, 89),
                "cooling_rate": round(random.uniform(5.0, 8.0), 1),
            }
        cases.append(make_case(
            f"SC-{idx:03d}",
            f"B202607{idx:02d}-D",
            pattern_name,
            params, gap, "hard",
            "多参数协同调整",
            keywords,
        ))
    return cases


def main():
    cases = []
    cases.extend(gen_holding_insufficient(15))
    cases.extend(gen_cooling_low(15))
    cases.extend(gen_temp_low(10))
    cases.extend(gen_composite(10))

    # 统计
    stats = {}
    for c in cases:
        stats[c["expected_root_cause"]] = stats.get(c["expected_root_cause"], 0) + 1

    output = {
        "version": "1.0",
        "description": "MetaCraft Agent 种子评估案例 - 热处理硬度偏低场景（50 条）",
        "standard_params": STANDARD,
        "total_cases": len(cases),
        "category_stats": stats,
        "cases": cases,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成 {len(cases)} 条种子案例")
    print(f"   输出: {OUTPUT}")
    print(f"   类别分布:")
    for cat, cnt in stats.items():
        print(f"     - {cat}: {cnt} 条")


if __name__ == "__main__":
    main()
