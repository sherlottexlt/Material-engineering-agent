"""M5-2 效果归因测试运行器

sandbox stdout 被吞 + exit code 被覆盖为 0，通过写文件反馈结果。
"""
import re
import subprocess
import sys
from pathlib import Path


def main():
    test_file = "tests/test_effect_attribution.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    # 解析 summary
    summary_match = re.search(r"(\d+) passed(?:.*?(\d+) failed)?", stdout)
    passed = int(summary_match.group(1)) if summary_match else 0
    failed = int(summary_match.group(2)) if summary_match and summary_match.group(2) else 0

    # 解析失败的测试名
    failed_tests = []
    for line in stdout.split("\n"):
        if "FAILED" in line:
            failed_tests.append(line.strip())

    report = (
        f"EXIT_CODE={result.returncode}\n"
        f"PASSED={passed}\n"
        f"FAILED={failed}\n"
        f"FAILED_TESTS:\n"
        + "\n".join(failed_tests[:50])
        + f"\n\n=== STDOUT (last 4000) ===\n{stdout[-4000:]}\n"
        f"\n=== STDERR (last 2000) ===\n{stderr[-2000:]}\n"
    )

    # 写入文件（sandbox 可能限制，多路径尝试）
    for path in ["data/m52_test_result.txt", "m52_test_result.txt"]:
        try:
            Path(path).write_text(report, encoding="utf-8")
            print(f"RESULT_WRITTEN_TO={path}", file=sys.stderr)
            break
        except Exception as e:
            print(f"WRITE_FAIL {path}: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
