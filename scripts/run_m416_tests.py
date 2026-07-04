"""M4-16 验证：跑 memory + multi_tenant + degradation + sla 测试
通过 exit code 反馈结果：0=全过，1=有失败，2=有错误
"""
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent.parent

# 抑制日志
import logging
logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except Exception:
    pass

# mock langchain_openai 避免 import 超时
class _FakeChatOpenAI:
    def __init__(self, *a, **kw): pass

mod = types.ModuleType("langchain_openai")
mod.ChatOpenAI = _FakeChatOpenAI
sys.modules["langchain_openai"] = mod

test_files = [
    "tests/test_memory.py",
    "tests/test_multi_tenant.py",
    "tests/test_degradation.py",
    "tests/test_sla.py",
    "tests/test_effect_tracking.py",
]

total_failed = 0
total_errors = 0
total_tests = 0

for tf in test_files:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / tf),
         "--tb=line", "-q", "--no-header"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    # 解析 pytest stdout 拿统计
    out = proc.stdout + proc.stderr
    for line in out.splitlines():
        line = line.strip()
        # pytest 末行格式: "X passed, Y failed in Zs" 或 "X passed"
        if "passed" in line or "failed" in line or "error" in line:
            # 简单解析
            import re
            m = re.search(r"(\d+)\s+passed", line)
            if m:
                total_tests += int(m.group(1))
            m = re.search(r"(\d+)\s+failed", line)
            if m:
                total_failed += int(m.group(1))
            m = re.search(r"(\d+)\s+error", line)
            if m:
                total_errors += int(m.group(1))
            m = re.search(r"(\d+)\s+skipped", line)
            if m:
                total_tests += int(m.group(1))

if total_failed > 0:
    sys.exit(1)
if total_errors > 0:
    sys.exit(2)
sys.exit(0)
