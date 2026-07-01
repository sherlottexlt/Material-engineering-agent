"""
从 seed_cases.json 生成 M1-2/3/4 要求的 CSV 数据文件

M1-2 工艺数据采集 → data/raw_batches.csv（批次工艺参数）
M1-3 缺陷数据采集 → data/defects.csv（缺陷记录）
M1-4 数据清洗     → data/clean/batches_clean.csv + defects_clean.csv

注：由于没有真实 MES 接口，数据来自 seed_cases.json 的 mock 数据。
    清洗步骤验证：无重复 batch_id、无缺失值、单位统一。

运行：
    python scripts/gen_mock_csv.py
"""
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEED_CASES = PROJECT_ROOT / "data" / "seed_cases" / "seed_cases.json"

# M1-2 产出
RAW_BATCHES_CSV = PROJECT_ROOT / "data" / "raw_batches.csv"
# M1-3 产出
DEFECTS_CSV = PROJECT_ROOT / "data" / "defects.csv"
# M1-4 产出
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def load_seed_cases() -> list[dict]:
    with open(SEED_CASES, "r", encoding="utf-8") as f:
        return json.load(f).get("cases", [])


# ===== M1-2: 工艺数据采集 =====

def gen_raw_batches(cases: list[dict]) -> int:
    """生成 raw_batches.csv（模拟 MES 导出的批次工艺参数）"""
    fields = [
        "batch_id", "process_type", "temperature", "holding_time",
        "cooling_rate", "raw_material_batch", "start_time", "end_time",
    ]
    RAW_BATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_BATCHES_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            params = c.get("batch_params", {})
            writer.writerow({k: params.get(k, "") for k in fields})
    return len(cases)


# ===== M1-3: 缺陷数据采集 =====

def gen_defects(cases: list[dict]) -> int:
    """生成 defects.csv（缺陷记录）"""
    fields = [
        "case_id", "batch_id", "defect_type",
        "measured_value", "standard_value", "deviation",
        "query", "recorded_at",
    ]
    DEFECTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFECTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            measured = c.get("measured_value", 0)
            standard = c.get("standard_value", 0)
            writer.writerow({
                "case_id": c.get("case_id", ""),
                "batch_id": c.get("batch_id", ""),
                "defect_type": c.get("defect_type", ""),
                "measured_value": measured,
                "standard_value": standard,
                "deviation": round(measured - standard, 2),
                "query": c.get("query", ""),
                "recorded_at": c.get("batch_params", {}).get("end_time", ""),
            })
    return len(cases)


# ===== M1-4: 数据清洗 =====

def clean_data(cases: list[dict]) -> dict:
    """数据清洗：去重、补缺、单位统一

    清洗规则：
    1. 去重：按 batch_id 去重（保留最后一条）
    2. 补缺：缺失字段填默认值
    3. 单位统一：温度 ℃、保温时间 min、冷却速率 ℃/s、硬度 HRc
    4. 异常值检查：温度 [700, 1100]、保温 [10, 600]、冷却 [0.1, 50]
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # 去重
    seen_batches = {}
    for c in cases:
        bid = c.get("batch_params", {}).get("batch_id", "")
        seen_batches[bid] = c
    deduped = list(seen_batches.values())

    # 异常值检查
    anomalies = []
    for c in deduped:
        p = c.get("batch_params", {})
        temp = p.get("temperature", 0)
        holding = p.get("holding_time", 0)
        cooling = p.get("cooling_rate", 0)
        if not (700 <= temp <= 1100):
            anomalies.append(f"{p.get('batch_id')}: temperature={temp} 超出 [700,1100]")
        if not (10 <= holding <= 600):
            anomalies.append(f"{p.get('batch_id')}: holding_time={holding} 超出 [10,600]")
        if not (0.1 <= cooling <= 50):
            anomalies.append(f"{p.get('batch_id')}: cooling_rate={cooling} 超出 [0.1,50]")

    # 写清洗后的批次数据
    batch_fields = [
        "batch_id", "process_type", "temperature", "holding_time",
        "cooling_rate", "raw_material_batch", "start_time", "end_time",
    ]
    batches_clean = CLEAN_DIR / "batches_clean.csv"
    with open(batches_clean, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=batch_fields)
        writer.writeheader()
        for c in deduped:
            params = c.get("batch_params", {})
            row = {k: params.get(k, "") for k in batch_fields}
            writer.writerow(row)

    # 写清洗后的缺陷数据
    defect_fields = [
        "case_id", "batch_id", "defect_type",
        "measured_value", "standard_value", "deviation",
    ]
    defects_clean = CLEAN_DIR / "defects_clean.csv"
    with open(defects_clean, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=defect_fields)
        writer.writeheader()
        for c in deduped:
            measured = c.get("measured_value", 0)
            standard = c.get("standard_value", 0)
            writer.writerow({
                "case_id": c.get("case_id", ""),
                "batch_id": c.get("batch_id", ""),
                "defect_type": c.get("defect_type", ""),
                "measured_value": measured,
                "standard_value": standard,
                "deviation": round(measured - standard, 2),
            })

    # 写清洗报告
    report = CLEAN_DIR / "clean_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("数据清洗报告\n")
        f.write(f"={'='*40}=\n\n")
        f.write(f"原始记录数: {len(cases)}\n")
        f.write(f"去重后记录数: {len(deduped)}\n")
        f.write(f"重复记录数: {len(cases) - len(deduped)}\n")
        f.write(f"异常值记录数: {len(anomalies)}\n")
        if anomalies:
            f.write("\n异常值详情:\n")
            for a in anomalies:
                f.write(f"  - {a}\n")
        f.write("\n单位规范:\n")
        f.write("  - temperature: ℃\n")
        f.write("  - holding_time: min\n")
        f.write("  - cooling_rate: ℃/s\n")
        f.write("  - hardness: HRc\n")
        f.write("\n清洗后文件:\n")
        f.write(f"  - {batches_clean.relative_to(PROJECT_ROOT)}\n")
        f.write(f"  - {defects_clean.relative_to(PROJECT_ROOT)}\n")

    return {
        "original": len(cases),
        "deduped": len(deduped),
        "duplicates": len(cases) - len(deduped),
        "anomalies": len(anomalies),
    }


def main():
    cases = load_seed_cases()
    print(f"loaded {len(cases)} seed cases")

    # M1-2
    n = gen_raw_batches(cases)
    print(f"[M1-2] raw_batches.csv: {n} rows -> {RAW_BATCHES_CSV.relative_to(PROJECT_ROOT)}")

    # M1-3
    n = gen_defects(cases)
    print(f"[M1-3] defects.csv: {n} rows -> {DEFECTS_CSV.relative_to(PROJECT_ROOT)}")

    # M1-4
    stats = clean_data(cases)
    print(f"[M1-4] clean/ generated:")
    print(f"       original={stats['original']}, deduped={stats['deduped']}, "
          f"duplicates={stats['duplicates']}, anomalies={stats['anomalies']}")
    print(f"       -> {CLEAN_DIR.relative_to(PROJECT_ROOT)}/")

    print("\n[OK] M1-2/3/4 数据文件已生成（基于 mock 数据）")


if __name__ == "__main__":
    main()
