"""从训练控制台输出反解每轮指标，还原成 `metrics.csv`。

`RunRecorder` 上线前的实验只把指标打到 stdout，曲线数据没有任何结构化副本。只要
控制台文本还在（`logs_*_run.txt` 或终端 scrollback），本模块就能把它还原成和现在
在线写出的 `metrics.csv` 完全同构的表，让历史实验和新实验能放在一起比。

对新实验也有用：训练中途崩溃、`metrics.csv` 写了一半时，可以从 `console.log` 重建。

识别三类行（均由 main_ppo 的训练循环产出）：
  - `iter=N  policy_loss=..  value_loss=..`      —— log_metrics，一轮的收尾行
  - `[SIL/MAP-Elites] cells=..  admitted=..`     —— Phase 1.6，早于收尾行
  - `[explore] pool_best_sdy=..  stall=..`       —— Phase 2.6，早于收尾行

后两类挂在紧随其后的 `iter=` 行上，保证同一轮的指标落在同一行。

用法:
  python -m src.training.run_log_parser runs/ou_me/console.log -o runs/ou_me/metrics.csv
  python -m src.training.run_log_parser logs_treefix2cont_run.txt      # 只打印摘要
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

_ITER_RE = re.compile(r"\biter=(\d+)\b")
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf))")
_SIL_RE = re.compile(r"\[SIL/MAP-Elites\]")
_EXPLORE_RE = re.compile(r"\[explore\]")

# 这些前缀行里的 key 与 log_metrics 的 key 不冲突，但仍加前缀避免将来撞名
_SIL_PREFIX = "me_"
_EXPLORE_PREFIX = "explore_"


def _parse_kv(line: str, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, raw in _KV_RE.findall(line):
        try:
            out[prefix + key] = float(raw)
        except ValueError:
            continue
    return out


def parse_console_log(path: str | Path) -> list[dict[str, Any]]:
    """解析控制台日志，返回按 iteration 排序的指标行。"""
    rows: list[dict[str, Any]] = []
    pending: dict[str, float] = {}
    seen: dict[int, dict[str, Any]] = {}

    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if _SIL_RE.search(line):
                pending.update(_parse_kv(line, _SIL_PREFIX))
                continue
            if _EXPLORE_RE.search(line):
                pending.update(_parse_kv(line, _EXPLORE_PREFIX))
                continue

            m = _ITER_RE.search(line)
            if m is None:
                continue
            # config 打印行、checkpoint 保存行等也可能含 iter=，用"是否带其它指标"区分
            kv = _parse_kv(line)
            kv.pop("iter", None)
            if not kv:
                continue

            iteration = int(m.group(1))
            row: dict[str, Any] = {"iteration": iteration}
            row.update(pending)
            row.update(kv)
            pending = {}
            # 断点续训会让同一 iteration 出现多次，后写的覆盖先写的
            if iteration in seen:
                seen[iteration].update(row)
            else:
                seen[iteration] = row
                rows.append(row)

    rows.sort(key=lambda r: r["iteration"])
    return rows


def write_metrics_csv(rows: list[dict[str, Any]], out_path: str | Path) -> int:
    """按所有行的列并集写 CSV，列顺序为首次出现顺序。"""
    if not rows:
        return 0
    columns: list[str] = ["iteration"]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="从训练控制台日志反解 metrics.csv")
    ap.add_argument("console_log", help="控制台日志文件")
    ap.add_argument("-o", "--output", default=None, help="输出 CSV（不给则只打印摘要）")
    args = ap.parse_args()

    rows = parse_console_log(args.console_log)
    if not rows:
        print(f"未从 {args.console_log} 解析到任何 iter= 指标行")
        return 1

    iters = [r["iteration"] for r in rows]
    keys = sorted({k for r in rows for k in r} - {"iteration"})
    print(f"解析到 {len(rows)} 轮: iter {min(iters)}..{max(iters)}")
    print(f"指标列 ({len(keys)}): {', '.join(keys)}")

    if args.output:
        n = write_metrics_csv(rows, args.output)
        print(f"已写入 {args.output} ({n} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
