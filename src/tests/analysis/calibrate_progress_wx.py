"""
progress_wx 标定助手（离线，只用录制轨迹 JSON，无需 torch）。

背景：奖励重设计把评分核心从单一纵向 summit 改为「攀升门控进度势」
  progress = secured_dy + progress_wx * max(0, reach)    （见 src/training/reward.py）
其中 secured_dy = dwell 驻留过滤后的净爬升，reach = 到达 secured 峰时的向右伸展。
progress_wx > 0 即可让「右侧真实路径（有 reach）」压过「爬树死路（reach≈0）」，比值主要影响
模棱两可处的取舍。本工具从你录制的「过树」轨迹几何反推一个「横向显著但不喧宾夺主」的 wx。

用法:
  # 单条：打印 secured_dy / reach 及不同占比 f 下的建议 wx
  python -m src.tests.analysis.calibrate_progress_wx src/Data/reference/reference_trajectory.json

  # 两条：过树成功轨迹 vs 爬树死路轨迹，验证在给定 wx 下前者评分更高
  python -m src.tests.analysis.calibrate_progress_wx over_tree.json dead_tree.json --wx 0.5

录制文件格式（record_reference_trajectory.py 产出）：{"states": (T+1,33), "actions": (T,2), ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.training.config import TrainConfig
from src.training.reward import base_score, progress_metric


def _load_states(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data["states"], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectories", nargs="+", help="一个或多个 reference_trajectory.json")
    ap.add_argument("--wx", type=float, default=None, help="用指定 wx 评分对比（覆盖 config 默认）")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.3, 0.5, 1.0],
                    help="横向占纵向的目标比例 f，反推 wx = f * secured_dy / reach")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.wx is not None:
        cfg.progress_wx = args.wx

    print(f"dwell_steps={cfg.dwell_steps} dwell_drop_tol={cfg.dwell_drop_tol} "
          f"progress_wx(current)={cfg.progress_wx}")
    print("-" * 72)

    infos = []
    for tpath in args.trajectories:
        p = Path(tpath)
        if not p.exists():
            print(f"[跳过] 不存在: {p}")
            continue
        states = _load_states(p)
        progress, peak_idx, secured_dy, reach = progress_metric(states, cfg)
        bs = base_score(states, cfg)
        infos.append((p.name, secured_dy, reach, progress, bs, peak_idx, len(states)))
        print(f"{p.name}")
        print(f"  T={len(states)}  secured_peak_idx={peak_idx}")
        print(f"  secured_dy={secured_dy:.3f} m   reach={reach:.3f} m   "
              f"progress={progress:.3f}   base_score={bs:.3f}")
        if reach > 1e-6 and secured_dy > 0:
            print("  建议 wx（横向占纵向比例 f → wx = f*secured_dy/reach）：")
            for f in args.fractions:
                print(f"    f={f:<4}  wx≈{f * secured_dy / reach:.3f}")
        else:
            print("  （reach≤0 或未爬升：该轨迹不产生横向加成，无法据此标定 wx）")
        print()

    if len(infos) >= 2:
        print("=" * 72)
        print("多条对比（base_score 越高越优；确认过树成功 > 爬树死路）：")
        for name, dy, rc, pg, bs, _, _ in sorted(infos, key=lambda r: -r[4]):
            print(f"  {bs:10.3f}  {name}  (dy={dy:.2f}, reach={rc:.2f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
