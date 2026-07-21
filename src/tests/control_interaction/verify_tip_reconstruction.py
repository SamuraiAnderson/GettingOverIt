"""
校验「近似重建 tip 世界多边形」是否有效。

背景：要判定「锤头多边形所有顶点都落在实体内」，需要 tip 的世界位姿。33D 观测里
只有 tip 中心位置（索引 23/24），没有 tip 的绝对旋转角，因此只能近似重建：
    world_poly = tip_center + scale * R(rod_dir + offset) @ tip_local
其中 rod_dir 用 pole→tip 方向（pole 与 tip 同在锤子刚体上，方向纯由锤子旋转决定；
hammerAngle=hub→tip 混入了身体位姿，作对照）。

游戏不会把 tip 世界顶点回传给 Python，无法做逐点真值比对，故本脚本用
「采样真实 settle 位姿 → 重建 → 叠加环境地图 + 已知骨架(hub–pole–tip) + tip 中心点」
做肉眼 + 统计校验，并支持 offset / scale 扫描以标定近似参数。

用法：
  python src/tests/control_interaction/verify_tip_reconstruction.py
  python src/tests/control_interaction/verify_tip_reconstruction.py --n-samples 16
  python src/tests/control_interaction/verify_tip_reconstruction.py --tip-angle-offset -90 --tip-scale 1.0
  python src/tests/control_interaction/verify_tip_reconstruction.py --sweep-offsets=-90,0,90,180
  python src/tests/control_interaction/verify_tip_reconstruction.py --angle-source hub
  python src/tests/control_interaction/verify_tip_reconstruction.py --no-launch  # 连已开游戏
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.config import TrainConfig
from training.deploy_sampling import (
    compute_drop_points,
    point_in_any_polygon,
    reconstruct_polygon_world,
)
from test_l7_surface_airdrop import (
    _get_game_root,
    compute_landable_surfaces,
    compute_player_height,
    extract_polygons,
    load_colliders,
)

logger = logging.getLogger(__name__)

# 33D 观测索引（见 PlayerState.ToFloatArray / entrypoints.md）
IDX_HUB = (5, 6)
IDX_POLE = (19, 20)
IDX_TIP = (23, 24)
IDX_HAMMER_ANGLE = 27

UNITY_MAX_DUPLICATES = 64
_REPORT_DIR = _REPO_ROOT / "src" / "Data" / "GameResults"


def _load_tip_local(player_data: dict) -> np.ndarray:
    """从 player_contour.json 取 tip 第一条 path 的局部多边形 (N, 2)。"""
    tip = player_data.get("parts", {}).get("tip")
    if not tip or not tip.get("paths"):
        raise RuntimeError("player_contour.json 缺少 tip 轮廓")
    return np.asarray(tip["paths"][0], dtype=np.float64)


def _rod_angle_deg(obs_row: np.ndarray, source: str) -> float:
    """锤杆世界朝向（度）。pole: pole→tip；hub: 直接用 hammerAngle(hub→tip)。"""
    if source == "hub":
        return float(obs_row[IDX_HAMMER_ANGLE])
    tx, ty = obs_row[IDX_TIP[0]], obs_row[IDX_TIP[1]]
    px, py = obs_row[IDX_POLE[0]], obs_row[IDX_POLE[1]]
    return math.degrees(math.atan2(ty - py, tx - px))


def _reconstruct(
    obs_row: np.ndarray,
    tip_local: np.ndarray,
    *,
    source: str,
    offset_deg: float,
    scale: float,
) -> tuple[np.ndarray, float]:
    """返回 (tip 世界多边形 (N,2), 采用的世界旋转角 deg)。"""
    center = np.array([obs_row[IDX_TIP[0]], obs_row[IDX_TIP[1]]], dtype=np.float64)
    theta = _rod_angle_deg(obs_row, source) + offset_deg
    world = reconstruct_polygon_world(center, theta, tip_local, scale)
    return world, theta


def _classify(world_poly: np.ndarray, polygons: list[np.ndarray]) -> tuple[int, bool, bool]:
    """返回 (顶点入实体数, 是否全部入实体, 中心点是否入实体)。"""
    n_in = sum(
        1 for v in world_poly if point_in_any_polygon(float(v[0]), float(v[1]), polygons)
    )
    cx = float(world_poly[:, 0].mean())
    cy = float(world_poly[:, 1].mean())
    center_in = point_in_any_polygon(cx, cy, polygons)
    return n_in, (n_in == len(world_poly)), center_in


# ── 采样 ────────────────────────────────────────────────────────

def _collect_samples(env, candidates: np.ndarray, settle_steps: int, eff_bs: int):
    """批量传送 + settle，返回 [(cand_x, cand_y, obs_row(33,)), ...]。"""
    from drop_point_physics import _deploy_chunk_settle

    rows = []
    i = 0
    while i < len(candidates):
        take = min(eff_bs, len(candidates) - i)
        chunk = candidates[i : i + take]
        i += take
        obs = _deploy_chunk_settle(env, chunk, settle_steps)
        for j in range(take):
            rows.append((float(chunk[j, 0]), float(chunk[j, 1]), obs[j + 1].copy()))
    return rows


# ── 可视化 ──────────────────────────────────────────────────────

def _draw_env_window(ax, polygons, segments, cx, cy, half, plt):
    from matplotlib.collections import PatchCollection

    patches = [plt.Polygon(p, closed=True) for p in polygons if len(p) >= 3]
    if patches:
        ax.add_collection(PatchCollection(
            patches, facecolor="#d8d8d8", edgecolor="#777", linewidth=0.3,
            alpha=0.9, zorder=1,
        ))
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="#2a9d8f", linewidth=1.2, zorder=2)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def _draw_sample(ax, obs_row, world_poly, theta, n_in, all_in, polygons, segments, plt, half):
    hub = (obs_row[IDX_HUB[0]], obs_row[IDX_HUB[1]])
    pole = (obs_row[IDX_POLE[0]], obs_row[IDX_POLE[1]])
    tip = (obs_row[IDX_TIP[0]], obs_row[IDX_TIP[1]])
    cx, cy = float(tip[0]), float(tip[1])
    _draw_env_window(ax, polygons, segments, cx, cy, half, plt)

    # 骨架 hub–pole–tip（精确，来自 state）
    ax.plot([hub[0], pole[0], tip[0]], [hub[1], pole[1], tip[1]],
            "-", color="#888", linewidth=1.0, zorder=4)
    ax.plot(hub[0], hub[1], "s", color="#666", markersize=4, zorder=5)
    ax.plot(pole[0], pole[1], "^", color="#444", markersize=5, zorder=5)
    ax.plot(tip[0], tip[1], "o", color="blue", markersize=5, zorder=6, label="tip center")

    # 重建的 tip 多边形
    color = "red" if all_in else ("orange" if n_in > 0 else "limegreen")
    closed = np.vstack([world_poly, world_poly[0:1]])
    ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.35, zorder=7)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5, zorder=8)

    ax.set_title(f"θ={theta:.0f}°  in={n_in}/{len(world_poly)}"
                 + ("  ALL-IN" if all_in else ""), fontsize=8)


def _plot_grid(rows, recon, polygons, segments, offset, scale, source, half, save_path, show):
    import matplotlib
    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt

    n = min(len(rows), 16)
    cols = 4
    rowsN = math.ceil(n / cols)
    fig, axes = plt.subplots(rowsN, cols, figsize=(cols * 4.2, rowsN * 4.0), squeeze=False)
    for k in range(rowsN * cols):
        ax = axes[k // cols][k % cols]
        if k >= n:
            ax.axis("off")
            continue
        _, _, obs_row = rows[k]
        world_poly, theta, n_in, all_in, _center_in = recon[k]
        _draw_sample(ax, obs_row, world_poly, theta, n_in, all_in,
                     polygons, segments, plt, half)
    fig.suptitle(
        f"Tip reconstruction — source={source} offset={offset:.0f}° scale={scale:.2f} "
        f"(red=all-in, orange=some, green=none)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        logger.info("图片已保存: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_sweep(rows, tip_local, polygons, segments, offsets, scale, source, half, save_path, show):
    import matplotlib
    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt

    k_samples = min(len(rows), 4)
    cols = len(offsets)
    fig, axes = plt.subplots(k_samples, cols, figsize=(cols * 4.0, k_samples * 4.0), squeeze=False)
    for r in range(k_samples):
        _, _, obs_row = rows[r]
        for cidx, off in enumerate(offsets):
            ax = axes[r][cidx]
            world_poly, theta = _reconstruct(
                obs_row, tip_local, source=source, offset_deg=off, scale=scale)
            n_in, all_in, _ = _classify(world_poly, polygons)
            _draw_sample(ax, obs_row, world_poly, theta, n_in, all_in,
                         polygons, segments, plt, half)
            if r == 0:
                ax.set_title(f"offset={off:.0f}°\n" + ax.get_title(), fontsize=8)
    fig.suptitle(
        f"Offset sweep — source={source} scale={scale:.2f} "
        f"(pick the column where tip neck points to pole & head hugs terrain)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        logger.info("扫描图已保存: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="校验 tip 世界多边形近似重建")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--drop-height", type=float, default=None)
    parser.add_argument("--angle-source", choices=["pole", "hub"], default="pole",
                        help="锤杆朝向基准：pole=pole→tip（推荐），hub=hammerAngle")
    parser.add_argument("--tip-angle-offset", type=float, default=-90.0,
                        help="世界旋转 = 锤杆朝向 + offset（几何假设默认 -90）")
    parser.add_argument("--tip-scale", type=float, default=1.0)
    parser.add_argument("--sweep-offsets", type=str, default=None,
                        help="逗号分隔的 offset 列表，进入扫描标定模式；负数打头需用等号，"
                             "如 --sweep-offsets=-90,0,90,180")
    parser.add_argument("--zoom-half", type=float, default=2.0, help="每个子图半窗口(米)")
    parser.add_argument("--settle-drop-threshold", type=float, default=None)
    parser.add_argument("--physics-settle-steps", type=int, default=None)
    parser.add_argument("--physics-warmup-steps", type=int, default=None)
    parser.add_argument("--physics-l8-batch-size", type=int, default=64)
    parser.add_argument("--game-root", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--y-max-cutoff", type=float, default=None)
    parser.add_argument("--padding", type=float, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    cfg = TrainConfig()
    port = args.port or cfg.port
    drop_height = args.drop_height if args.drop_height is not None else cfg.drop_height
    padding = args.padding if args.padding is not None else cfg.surface_padding
    y_max_cutoff = args.y_max_cutoff if args.y_max_cutoff is not None else cfg.y_max_cutoff
    settle_steps = args.physics_settle_steps or cfg.settle_steps
    warmup_steps = args.physics_warmup_steps or cfg.warmup_steps
    eff_bs = max(1, min(int(args.physics_l8_batch_size), UNITY_MAX_DUPLICATES))

    game_root = _get_game_root(args.game_root)
    logger.info("游戏根目录: %s", game_root)

    from env.goi_env import GoiEnv
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    from training.rollout import _write_num_duplicates
    from drop_point_physics import _teardown_env_and_process, _warmup_and_snapshot

    # agent 数在启动前确定（C# 初始化时读 runtime_config.json）
    num_agents = min(max(1, args.n_samples), eff_bs, UNITY_MAX_DUPLICATES) + 1
    _write_num_duplicates(game_root, num_agents)
    proc = None
    if not args.no_launch:
        GameModeController(str(game_root)).set_game_runtime_mode()
        proc = GameLauncher().launch(wait=False)
    env = GoiEnv(port=port, num_agents=num_agents, timeout=args.timeout)
    env.connect()
    try:
        env.reset()
        # 几何：环境多边形 + 可着陆线段 + tip 局部轮廓（缺失则通过 TCP 导出）
        env_data, player_data = load_colliders(game_root, env=env)
        tip_local = _load_tip_local(player_data)
        polygons = extract_polygons(env_data)
        player_height = compute_player_height(player_data)
        min_clearance = player_height + padding
        segments = compute_landable_surfaces(polygons, min_clearance, resolution=0.1,
                                             y_max_cutoff=y_max_cutoff)
        logger.info("环境多边形 %d，可着陆线段 %d，tip 局部顶点 %d",
                    len(polygons), len(segments), len(tip_local))

        candidates = compute_drop_points(segments, args.n_samples, drop_height, height_decay=0.0)
        if len(candidates) == 0:
            logger.error("无候选投放点")
            return
        logger.info("候选投放点 %d 个", len(candidates))

        _warmup_and_snapshot(env, num_agents, warmup_steps)
        logger.info("快照完成（%d agents），开始采样 %d 个位姿", num_agents, len(candidates))
        rows = _collect_samples(env, candidates, settle_steps, eff_bs)
    finally:
        _teardown_env_and_process(env, proc)
    logger.info("采样到 %d 个位姿", len(rows))

    save_dir = Path(args.save_path).parent if args.save_path else _REPORT_DIR
    show = not args.no_plot

    if args.sweep_offsets:
        offsets = [float(x) for x in args.sweep_offsets.split(",") if x.strip()]
        sweep_path = args.save_path or str(save_dir / "tip_recon_sweep.png")
        _plot_sweep(rows, tip_local, polygons, segments, offsets, args.tip_scale,
                    args.angle_source, args.zoom_half, sweep_path, show)
        logger.info("扫描模式完成：请在图中挑选 tip 沿骨架、头部贴合地形的 offset 列")
        return

    # 固定参数重建 + 分类 + 统计
    recon = []
    for _cx, _cy, obs_row in rows:
        world_poly, theta = _reconstruct(
            obs_row, tip_local, source=args.angle_source,
            offset_deg=args.tip_angle_offset, scale=args.tip_scale)
        n_in, all_in, center_in = _classify(world_poly, polygons)
        recon.append((world_poly, theta, n_in, all_in, center_in))

    n_all = sum(1 for r in recon if r[3])
    n_center = sum(1 for r in recon if r[4])
    n_some = sum(1 for r in recon if r[2] > 0)
    logger.info("=" * 60)
    logger.info("重建统计（source=%s offset=%.0f scale=%.2f）：",
                args.angle_source, args.tip_angle_offset, args.tip_scale)
    logger.info("  全顶点入实体(all-in): %d/%d", n_all, len(rows))
    logger.info("  有顶点入实体(some):   %d/%d", n_some, len(rows))
    logger.info("  中心点入实体(旧判据): %d/%d", n_center, len(rows))
    logger.info("=" * 60)

    # CSV
    csv_path = save_dir / "tip_recon_samples.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "cand_x", "cand_y", "tip_x", "tip_y", "pole_x", "pole_y",
                    "hammerAngle", "theta", "n_in", "n_verts", "all_in", "center_in"])
        for idx, ((cx, cy, obs_row), (world_poly, theta, n_in, all_in, center_in)) in enumerate(
                zip(rows, recon)):
            w.writerow([idx, f"{cx:.3f}", f"{cy:.3f}",
                        f"{obs_row[IDX_TIP[0]]:.3f}", f"{obs_row[IDX_TIP[1]]:.3f}",
                        f"{obs_row[IDX_POLE[0]]:.3f}", f"{obs_row[IDX_POLE[1]]:.3f}",
                        f"{obs_row[IDX_HAMMER_ANGLE]:.2f}", f"{theta:.2f}",
                        n_in, len(world_poly), int(all_in), int(center_in)])
    logger.info("逐样本数据: %s", csv_path)

    grid_path = args.save_path or str(save_dir / "tip_recon_grid.png")
    _plot_grid(rows, recon, polygons, segments, args.tip_angle_offset, args.tip_scale,
               args.angle_source, args.zoom_half, grid_path, show)
    logger.info("请肉眼核对：重建多边形是否沿 hub–pole–tip 骨架、尺度与地形相称、"
                "稳定落点头部贴合而非悬空/深埋")


if __name__ == "__main__":
    main()
