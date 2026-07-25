"""批量运行不需要游戏在环的单元测试。

本项目的测试脚本各自带 `main()` 自跑（环境里没装 pytest），于是"跑一遍全部单测"
一直没有统一入口，改完训练代码只能靠记忆逐个敲。这个运行器补上这一环。

只收 `src/tests/training/test_*.py`——这些是纯计算、不碰游戏进程的回归测试。
需要游戏在环的 `control_interaction/test_l*.py` 不在此列，它们得手动跑。

每个测试以子进程运行，避免相互污染 sys.path 与全局状态。

用法:
  python -m src.tests.run_unit_tests
  python -m src.tests.run_unit_tests -k contact     # 只跑名字含 contact 的
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNIT_DIRS = ("training",)


def discover(pattern: str | None) -> list[str]:
    """返回可用 `python -m` 调用的测试模块名。"""
    mods = []
    for sub in _UNIT_DIRS:
        for path in sorted((_REPO_ROOT / "src" / "tests" / sub).glob("test_*.py")):
            if pattern and pattern not in path.stem:
                continue
            mods.append(f"src.tests.{sub}.{path.stem}")
    return mods


def run_one(module: str) -> tuple[bool, float, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    elapsed = time.perf_counter() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, elapsed, output


def main() -> int:
    ap = argparse.ArgumentParser(description="运行离线单元测试（无需游戏在环）")
    ap.add_argument("-k", "--filter", default=None, help="只跑名字包含该子串的测试")
    ap.add_argument("-v", "--verbose", action="store_true", help="总是打印测试输出")
    args = ap.parse_args()

    modules = discover(args.filter)
    if not modules:
        print("没有匹配的测试")
        return 1

    failures = []
    for module in modules:
        ok, elapsed, output = run_one(module)
        print(f"[{'PASS' if ok else 'FAIL'}] {module}  ({elapsed:.1f}s)")
        if args.verbose or not ok:
            print("\n".join("    " + line for line in output.splitlines()))
        if not ok:
            failures.append(module)

    print(f"\n{len(modules) - len(failures)}/{len(modules)} 通过")
    if failures:
        print("失败: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
