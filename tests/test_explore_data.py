"""
数据探查脚本单测（M1-6 配套）

测试 scripts/explore_data.py 中的纯函数：
- quantile：分位数计算
- iqr_outliers：IQR 异常值检测
- stat_block：统计摘要
- fmt：数字格式化
- main：端到端生成报告（生成文件存在 + 关键字段存在）
"""
import os
import json
import subprocess
import sys

import pytest

# 把 scripts/ 加入 import 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from explore_data import quantile, iqr_outliers, stat_block, fmt


# ===== quantile =====
class TestQuantile:
    def test_median_odd(self):
        assert quantile([1, 3, 5, 7, 9], 0.5) == 5

    def test_median_even(self):
        # 4 个值：1,2,3,4 → 中位数 = 2.5
        assert quantile([1, 2, 3, 4], 0.5) == 2.5

    def test_q1(self):
        # 1..10 的 Q1 = 3.25
        assert quantile(list(range(1, 11)), 0.25) == 3.25

    def test_q3(self):
        # 1..10 的 Q3 = 7.75
        assert quantile(list(range(1, 11)), 0.75) == 7.75

    def test_min(self):
        assert quantile([5, 3, 1, 4, 2], 0.0) == 1

    def test_max(self):
        assert quantile([5, 3, 1, 4, 2], 1.0) == 5

    def test_empty(self):
        assert quantile([], 0.5) is None

    def test_single(self):
        assert quantile([42], 0.5) == 42

    def test_unsorted_input(self):
        # 不预排序也能正确算
        assert quantile([3, 1, 2], 0.5) == 2


# ===== iqr_outliers =====
class TestIqrOutliers:
    def test_no_outliers(self):
        # 1..10 没有 IQR 异常
        lower, upper, outliers = iqr_outliers(list(range(1, 11)))
        assert len(outliers) == 0
        assert lower < 1
        assert upper > 10

    def test_detects_low_outlier(self):
        # 在正态数据外加一个极小值
        data = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, -100]
        _, _, outliers = iqr_outliers(data)
        assert 10 in outliers  # -100 是第 10 个索引

    def test_detects_high_outlier(self):
        data = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1000]
        _, _, outliers = iqr_outliers(data)
        assert 10 in outliers

    def test_small_dataset_no_crash(self):
        # 少于 4 个值不报错
        lower, upper, outliers = iqr_outliers([1, 2])
        assert outliers == []

    def test_empty(self):
        lower, upper, outliers = iqr_outliers([])
        assert outliers == []
        assert lower == 0
        assert upper == 0

    def test_returns_bounds(self):
        lower, upper, _ = iqr_outliers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Q1=3.25, Q3=7.75, IQR=4.5, lower=3.25-6.75=-3.5, upper=7.75+6.75=14.5
        assert lower == pytest.approx(-3.5, abs=0.01)
        assert upper == pytest.approx(14.5, abs=0.01)


# ===== stat_block =====
class TestStatBlock:
    def test_basic_stats(self):
        s = stat_block([1, 2, 3, 4, 5])
        assert s["count"] == 5
        assert s["min"] == 1
        assert s["max"] == 5
        assert s["mean"] == 3
        assert s["median"] == 3
        assert s["stdev"] == pytest.approx(1.5811, rel=0.01)

    def test_empty(self):
        s = stat_block([])
        assert s["count"] == 0
        assert s["min"] is None
        assert s["max"] is None
        assert s["mean"] is None

    def test_single(self):
        s = stat_block([42])
        assert s["count"] == 1
        assert s["stdev"] is None  # 单个值无标准差

    def test_two_values(self):
        s = stat_block([10, 20])
        assert s["mean"] == 15
        assert s["stdev"] == pytest.approx(7.0711, rel=0.01)

    def test_quantiles(self):
        s = stat_block(list(range(1, 11)))  # 1..10
        assert s["q1"] == pytest.approx(3.25, abs=0.01)
        assert s["q3"] == pytest.approx(7.75, abs=0.01)


# ===== fmt =====
class TestFmt:
    def test_float_default_precision(self):
        assert fmt(3.14159) == "3.14"

    def test_float_custom_precision(self):
        assert fmt(3.14159, 4) == "3.1416"

    def test_int(self):
        assert fmt(42) == "42"

    def test_none(self):
        assert fmt(None) == "N/A"

    def test_zero(self):
        assert fmt(0.0) == "0.00"


# ===== main 端到端 =====
class TestMainEndToEnd:
    """端到端验证：跑 main() 后报告文件存在且包含关键章节"""

    def test_report_file_exists(self):
        # 报告应该已经被生成（CI/开发时已运行过）
        report_path = os.path.join(ROOT, "data", "data_exploration_report.md")
        assert os.path.exists(report_path), "请先运行 python scripts/explore_data.py 生成报告"

    def test_report_contains_key_sections(self):
        report_path = os.path.join(ROOT, "data", "data_exploration_report.md")
        if not os.path.exists(report_path):
            pytest.skip("报告未生成")
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 必须包含 8 个章节标题
        for section in [
            "## 1. 字段完整性",
            "## 2. 数值字段统计分布",
            "## 3. 异常值检测",
            "## 4. 根因类别分布",
            "## 5. 参数偏离方向与根因一致性",
            "## 6. 参数相对偏离幅度",
            "## 7. 工艺约束越界检查",
            "## 8. 关键发现与结论",
        ]:
            assert section in content, f"缺少章节：{section}"

    def test_report_contains_50_cases(self):
        report_path = os.path.join(ROOT, "data", "data_exploration_report.md")
        if not os.path.exists(report_path):
            pytest.skip("报告未生成")
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "案例总数**：50" in content

    def test_seed_cases_integrity(self):
        """验证 seed_cases.json 数据本身完整"""
        cases_path = os.path.join(ROOT, "data", "seed_cases", "seed_cases.json")
        with open(cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_cases"] == 50
        assert len(data["cases"]) == 50
        # 每个案例必须有核心字段
        for c in data["cases"]:
            assert "case_id" in c
            assert "batch_params" in c
            assert "expected_root_cause" in c
            bp = c["batch_params"]
            for k in ["temperature", "holding_time", "cooling_rate"]:
                assert k in bp, f"{c['case_id']} 缺 {k}"

    def test_main_rebuilds_report(self):
        """重新跑 main() 应能成功覆盖报告文件"""
        from explore_data import main
        main()
        report_path = os.path.join(ROOT, "data", "data_exploration_report.md")
        assert os.path.exists(report_path)
        # 文件应非空
        assert os.path.getsize(report_path) > 1000  # 至少 1KB
