"""归档模型评估结果到 `runs/_eval/`，并维护一张可入库的汇总表。

`test_l9_model_eval` 每次评估会写出一个几百 KB 的 JSON，里面 `records` 是每回合的
统计（这是结论），`states`/`actions` 是全量轨迹（这是原始数据，占了 99% 的体积）。
以前这些文件全落在 `logs/` 下，而 `.gitignore` 的 `[Ll]ogs/` 把整个目录吞掉了——
于是"哪个 checkpoint 评出来多少米"这种关键结论从来没进过版本库。

这里把两者分开：原始 JSON 留在 `runs/_eval/`（gitignore），逐回合统计聚合成
`runs/_eval/summary.csv`（入库）。这样即便原始文件被清掉，评估结论仍然可查、可 diff。

用法:
  python -m src.training.archive_evals --collect logs      # 归档 logs/eval_*.json
  python -m src.training.archive_evals                     # 只重建 summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.run_logging import RUNS_ROOT

EVAL_DIR = "_eval"
SUMMARY_NAME = "summary.csv"

# records 里每回合的字段，取自 test_l9_model_eval
_RECORD_KEYS = ("start_y", "final_y", "max_y", "secured_dy", "reach")


def eval_root(repo: Path | None = None) -> Path:
    return (repo or _REPO_ROOT) / RUNS_ROOT / EVAL_DIR


def summarize_eval(path: Path) -> dict[str, object] | None:
    """把一个 eval JSON 压成一行：跨回合取均值，另记最好的一回合。

    兼容两代格式：新版把逐回合统计放在 `records` 里；早期单回合版本直接把
    `start_y/final_y/max_y` 摊在顶层（没有 `secured_dy`/`reach`，汇总表里留空）。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = data.get("records") or []
    if not records:
        legacy = {k: data[k] for k in ("start_y", "final_y", "max_y") if k in data}
        if not legacy:
            return None
        records = [legacy]

    row: dict[str, object] = {
        "name": path.stem.removeprefix("eval_"),
        "checkpoint": str(data.get("checkpoint", "")).replace("\\", "/"),
        "steps": data.get("steps"),
        "episodes": len(records),
    }
    for key in _RECORD_KEYS:
        vals = [r[key] for r in records if key in r]
        if vals:
            row[f"{key}_mean"] = round(mean(vals), 4)
            row[f"{key}_best"] = round(max(vals), 4)
    row["file"] = path.name
    return row


def rebuild_summary(repo: Path | None = None) -> int:
    root = eval_root(repo)
    if not root.is_dir():
        return 0
    rows = [r for p in sorted(root.glob("eval_*.json")) if (r := summarize_eval(p))]
    if not rows:
        return 0
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (root / SUMMARY_NAME).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def collect(source: Path, repo: Path | None = None, move: bool = True) -> int:
    root = eval_root(repo)
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(source.glob("eval_*.json")):
        dest = root / path.name
        if dest.resolve() == path.resolve():
            continue
        (shutil.move if move else shutil.copy2)(str(path), str(dest))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="归档评估结果并重建汇总表")
    ap.add_argument("--collect", type=str, default=None,
                    help="从该目录搬运 eval_*.json（如 logs）")
    ap.add_argument("--copy", action="store_true", help="搬运时复制而非移动")
    args = ap.parse_args()

    if args.collect:
        moved = collect(Path(args.collect), move=not args.copy)
        print(f"归档 {moved} 个评估结果 → {eval_root()}")

    n = rebuild_summary()
    print(f"汇总表已重建: {eval_root() / SUMMARY_NAME} ({n} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
