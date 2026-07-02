"""M4-14 验证：重新运行 M4-13 故障演练，验证降级率提升

用 sys.modules mock langchain_openai 绕过 sandbox import 超时。
"""
import logging
import sys
import os
import types
import json
from pathlib import Path

logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 注入 mock langchain_openai
class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.max_retries = kwargs.get("max_retries", 3)
    def invoke(self, *a, **kw):
        return types.SimpleNamespace(content="mock")
    async def ainvoke(self, *a, **kw):
        return types.SimpleNamespace(content="mock")

_fake = types.ModuleType("langchain_openai")
_fake.ChatOpenAI = _FakeChatOpenAI
sys.modules["langchain_openai"] = _fake

from scripts.run_chaos_test import run_chaos_test, generate_markdown_report

# 运行完整演练（4 场景 x 5 端点 x 1 重复）
report = run_chaos_test(scenario_keys=None, repeats=1)

# 写报告
json_path = PROJECT_ROOT / "data" / "chaos_test_report.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

md_path = PROJECT_ROOT / "data" / "chaos_test_report.md"
md = generate_markdown_report(report)
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)

# 计算降级率
total_hit = 0
total_degraded = 0
for scenario, endpoints in report.get("results", {}).items():
    for ep_name, ep_data in endpoints.items():
        classification = ep_data.get("classification", "")
        if classification in ("degraded_ok", "failed"):
            total_hit += 1
            if classification == "degraded_ok":
                total_degraded += 1

rate = total_degraded / total_hit if total_hit > 0 else 0
print(f"Done: degradation rate = {total_degraded}/{total_hit} = {rate*100:.1f}%")
