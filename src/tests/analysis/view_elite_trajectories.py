"""查看当前精英轨迹（SIL 池快照）。

精英池训练时仅在内存中、未落盘，故用最终模型现采集一批 rollout，按训练同口径
score_trajectory(=base_score) 排序取 top-K，即"当前策略产出的精英轨迹"。
两遍采集：pass1 建 Φ（wide 通道非零、贴近训练观测），pass2 打分+可视化。

用法:
  python -m src.tests.analysis.view_elite_trajectories --checkpoint checkpoints_treefix2/ppo_iter_0190.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "src", _REPO / "src" / "training",
          _REPO / "src" / "tests" / "control_interaction"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training.config import TrainConfig
from training.actor_critic import ActorCritic
from training.reward import (
    ClimbingEfficiencyMap, score_trajectory, progress_metric,
)
from training.rollout import RolloutWorker
from training.deploy_sampling import extract_tip_local, extract_body_local
from test_l7_surface_airdrop import extract_polygons, load_colliders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TREE = dict(x0=-30.2, x1=-26.6, y0=-4.3, y1=4.8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_treefix2/ppo_iter_0190.pt")
    ap.add_argument("--checkpoint-dir", default="checkpoints_treefix2")
    ap.add_argument("--num-agents", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="logs_treefix2/elite_trajectories.png")
    args = ap.parse_args()

    cfg = TrainConfig()
    cfg.num_agents = args.num_agents
    cfg.steps_per_rollout = args.steps
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.random_deploy = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model = ActorCritic(cfg)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device).eval()
    logger.info("模型已加载: %s (iter=%s)", args.checkpoint, ckpt.get("iteration", "?")
                if isinstance(ckpt, dict) else "?")

    game_root = Path(cfg.game_root)
    env_data, player_data = load_colliders(game_root)
    polygons = extract_polygons(env_data)
    mask_cache = str(Path(cfg.checkpoint_dir) / "cache" / "traversable_mask.npz")
    eff_map = ClimbingEfficiencyMap(
        polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations,
        diffusion_alpha=cfg.diffusion_alpha, cache_path=mask_cache,
        height_prior_weight=cfg.height_prior_weight,
    )
    logger.info("效率图就绪 (traversable=%d)", int(eff_map.traversable.sum()))

    worker = RolloutWorker(cfg)
    worker.set_deploy_obstacle_geometry(polygons, player_data)
    worker.launch_game()

    try:
        logger.info("pass1: 采集以建 Φ ...")
        _, trajs1 = worker.collect_ppo(model, cfg.steps_per_rollout, eff_map=eff_map)
        eff_map.update(trajs1, cfg, prev_eff_map=None)
        logger.info("pass2: 采集精英快照 ...")
        _, trajs = worker.collect_ppo(model, cfg.steps_per_rollout, eff_map=eff_map)
    finally:
        worker.close_game()

    scored = []
    for t in trajs:
        s = score_trajectory(t.raw_states, eff_map, cfg)
        _p, _pk, sdy, reach = progress_metric(t.raw_states, cfg)
        st = t.raw_states
        scored.append(dict(
            score=s, sdy=float(sdy), reach=float(reach),
            max_dy=float(st[:, 1].max() - st[0, 1]),
            final_dy=float(st[-1, 1] - st[0, 1]),
            start=(float(st[0, 0]), float(st[0, 1])),
            peak_y=float(st[:, 1].max()), states=st,
        ))
    scored.sort(key=lambda d: d["score"], reverse=True)
    top = scored[: args.topk]

    print("\n==== 当前精英轨迹 top-%d (按 score=base_score 排序) ====" % len(top))
    print("rank  score   secured_dy  reach   max_Δy  final_Δy  peak_y   start(x,y)")
    for i, d in enumerate(top):
        print("%3d  %7.3f  %8.2f  %6.2f  %6.2f  %7.2f  %6.1f   (%.1f,%.1f)" % (
            i + 1, d["score"], d["sdy"], d["reach"], d["max_dy"],
            d["final_dy"], d["peak_y"], d["start"][0], d["start"][1]))
    allsdy = np.array([d["sdy"] for d in scored])
    print("\n全批(%d 条) secured_dy: max=%.2f mean=%.2f  | 精英池 secured_dy: max=%.2f mean=%.2f"
          % (len(scored), allsdy.max(), allsdy.mean(),
             np.max([d["sdy"] for d in top]), np.mean([d["sdy"] for d in top])))

    # ── 可视化 ──
    fig, ax = plt.subplots(figsize=(9, 11))
    for poly in polygons:
        pp = np.vstack([poly, poly[0]])
        ax.plot(pp[:, 0], pp[:, 1], color="0.7", lw=0.3)
    ax.add_patch(plt.Rectangle((TREE["x0"], TREE["y0"]),
                               TREE["x1"] - TREE["x0"], TREE["y1"] - TREE["y0"],
                               fill=False, edgecolor="green", lw=1.5, label="Deadtree"))
    cmap = plt.cm.viridis
    for i, d in enumerate(top):
        st = d["states"]
        c = cmap(i / max(len(top) - 1, 1))
        ax.plot(st[:, 0], st[:, 1], color=c, lw=1.0, alpha=0.8)
        ax.plot(st[0, 0], st[0, 1], "o", color=c, ms=4)
        ax.plot(st[:, 0][np.argmax(st[:, 1])], st[:, 1].max(), "^", color=c, ms=5)
    xs = np.concatenate([d["states"][:, 0] for d in top])
    ys = np.concatenate([d["states"][:, 1] for d in top])
    ax.set_xlim(xs.min() - 10, xs.max() + 10)
    ax.set_ylim(ys.min() - 5, ys.max() + 15)
    ax.set_aspect("equal")
    ax.set_title("Current elite trajectories (top-%d by base_score)\n"
                 "circle=start, triangle=peak, green box=Deadtree" % len(top))
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(loc="upper right")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("\n已保存: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
