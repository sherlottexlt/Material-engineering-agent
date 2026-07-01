"""后台运行 M2 评估，输出重定向到文件

用法:
    python scripts/run_m2_eval.py             # 全新评估
    python scripts/run_m2_eval.py --resume    # 跳过已完成的用例
"""
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# 是否 resume 模式
resume_mode = "--resume" in sys.argv

log_file = PROJECT_ROOT / "data" / "eval_m2_log.txt"

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
mode_str = "resume" if resume_mode else "fresh"
cmd_args = [sys.executable, "eval/run_eval.py", "--flow", "parallel"]
if resume_mode:
    cmd_args.append("--resume")

# resume 模式追加日志，fresh 模式覆盖
log_mode = "a" if resume_mode else "w"
with open(log_file, log_mode, encoding="utf-8") as f:
    f.write(f"\nM2 评估启动 ({mode_str}): {timestamp}\n")
    f.write(f"命令: {' '.join(cmd_args)}\n")
    f.write(f"{'='*60}\n")
    f.flush()

proc = subprocess.Popen(
    cmd_args,
    stdout=open(log_file, "a", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    cwd=str(PROJECT_ROOT),
)

with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"后台进程已启动, PID={proc.pid}\n")
    f.write(f"请定期检查 data/eval_incremental_m2.json 查看进度\n")
    f.flush()

print(f"M2 评估已启动 ({mode_str}), PID={proc.pid}")
print(f"日志: {log_file}")
print(f"增量结果: data/eval_incremental_m2.json")
