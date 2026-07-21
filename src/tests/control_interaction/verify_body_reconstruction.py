"""
校验「近似重建 body/锅 世界多边形」是否有效（用于 dualscale patch 的接触分支）。

背景：body/pot 轮廓在 player 共享局部系中定义（body 躯干在 +y、pot 锅在 -y），
其世界摆放需要 player 身体的世界位姿。33D 观测里有 player 中心（0/1）但没有 body 绝对
旋转角（仅 hubAngle/sliderAngle/hammerAngle），因此近似重建：
    world_poly = player_center + scale * R(player→hub 方向 + offset) @ body_local
局部 +y=躯干朝上，几何上应有 offset≈-90（与锤头同构），本脚本采样真实位姿做
肉眼 + 统计校验，并支持 offset / scale 扫描以标定该近似参数。

与 verify_tip_reconstruction 一致：游戏不回传 body 世界顶点，无法逐点真值比对，故用
「采样真实 settle 位姿 → 重建 → 叠加环境地图 + 骨架(player–hub) + player 中心」做校验。

用法：
  python src/tests/control_interaction/verify_body_reconstruction.py
  python src/tests/control_interaction/verify_body_reconstruction.py --body-angle-offset -90 --body-scale 1.0
  python src/tests/control_interaction/verify_body_reconstruction.py --sweep-offsets=-90,0,90,180
  python src/tests/control_interaction/verify_body_reconstruction.py --no-launch  # 连已开游戏
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
    extract_body_local,
    player_hub_angle_deg,
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
IDX_PLAYER = (0, 1)
IDX_HUB = (5, 6)

UNITY_MAX_DUPLICATES = 64
_REPORT_DIR = _REPO_ROOT / "src" / "Data" / "GameResults"


def _reconstruct(
    obs_row: np.ndarray,
    body_local: list[np.ndarray],
    *,
    offset_deg: float,
    scale: float,
) -> tuple[list[np.ndarray], float]:
    """返回 (body/pot 世界多边形列表, 采用的世界旋转角 deg)。"""
    center = np.array([obs_row[IDX_PLAYER[0]], obs_row[IDX_PLAYER[1]]], dtype=np.float64)
    theta = player_hub_angle_deg(obs_row) + offset_deg
    worlds = [reconstruct_polygon_world(center, theta, poly, scale) for poly in body_local]
    return worlds, theta


def _classify(worlds: list[np.ndarray], polygons: list[np.ndarray]) -> tuple[int, int]:
    """返回 (所有顶点入实体数, 总顶点数)——嵌入越多说明轮廓穿入地形越深（可能过大/偏移）。"""
    n_in = 0
    n_tot = 0
    for w in worlds:
        n_tot += len(w)
        n_in += sum(
            1 for v in w if point_in_any_polygon(float(v[0]), float(v[1]), polygons)
        )
    return n_in, n_tot


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


def _draw_sample(ax, obs_row, worlds, theta, n_in, n_tot, polygons, segments, plt, half):
    player = (obs_row[IDX_PLAYER[0]], obs_row[IDX_PLAYER[1]])
    hub = (obs_row[IDX_HUB[0]], obs_row[IDX_HUB[1]])
    cx, cy = float(player[0]), float(player[1])
    _draw_env_window(ax, polygons, segments, cx, cy, half, plt)

    # 骨架 player–hub（精确，来自 state）
    ax.plot([player[0], hub[0]], [player[1], hub[1]],
            "-", color="#888", linewidth=1.0, zorder=4)
    ax.plot(player[0], player[1], "o", color="blue", markersize=5, zorder=6, label="player")
    ax.plot(hub[0], hub[1], "s", color="#666", markersize=4, zorder=5)

    # 重建的 body/pot 多边形
    for w in worlds:
        closed = np.vstack([w, w[0:1]])
        ax.fill(closed[:, 0], closed[:, 1], color="orange", alpha=0.30, zorder=7)
        ax.plot(closed[:, 0], closed[:, 1], color="orangered", linewidth=1.3, zorder=8)

    ax.set_title(f"θ={theta:.0f}°  vin={n_in}/{n_tot}", fontsize=8)


def _plot_grid(rows, recon, polygons, segments, offset, scale, half, save_path, show):
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
        worlds, theta, n_in, n_tot = recon[k]
        _draw_sample(ax, obs_row, worlds, theta, n_in, n_tot,
                     polygons, segments, plt, half)
    fig.suptitle(
        f"Body/pot reconstruction — offset={offset:.0f}° scale={scale:.2f} "
        f"(neck should point up along player->hub, 锅贴地不深埋)",
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


def _plot_sweep(rows, body_local, polygons, segments, offsets, scale, half, save_path, show):
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
            worlds, theta = _reconstruct(obs_row, body_local, offset_deg=off, scale=scale)
            n_in, n_tot = _classify(worlds, polygons)
            _draw_sample(ax, obs_row, worlds, theta, n_in, n_tot,
                         polygons, segments, plt, half)
            if r == 0:
                ax.set_title(f"offset={off:.0f}°\n" + ax.get_title(), fontsize=8)
    fig.suptitle(
        f"Body offset sweep - scale={scale:.2f} "
        f"(pick column: torso-up, pot-on-ground, contour matches body)",
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
    parser = argparse.ArgumentParser(description="校验 body/pot 世界多边形近似重建")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--drop-height", type=float, default=None)
    parser.add_argument("--body-angle-offset", type=float, default=-90.0,
                        help="世界旋转 = player→hub 朝向 + offset（几何假设默认 -90）")
    parser.add_argument("--body-scale", type=float, default=1.0)
    parser.add_argument("--sweep-offsets", type=str, default=None,
                        help="逗号分隔 offset 列表进入扫描；负数打头用等号，"
                             "如 --sweep-offsets=-90,0,90,180")
    parser.add_argument("--zoom-half", type=float, default=2.0, help="每个子图半窗口(米)")
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
        env_data, player_data = load_colliders(game_root, env=env)
        body_local = extract_body_local(player_data)
        if not body_local:
            logger.error("player_contour.json 缺少 body/pot 轮廓")
            return
        polygons = extract_polygons(env_data)
        player_height = compute_player_height(player_data)
        min_clearance = player_height + padding
        segments = compute_landable_surfaces(polygons, min_clearance, resolution=0.1,
                                             y_max_cutoff=y_max_cutoff)
        logger.info("环境多边形 %d，可着陆线段 %d，body/pot 轮廓 %d 段（顶点 %s）",
                    len(polygons), len(segments), len(body_local),
                    [len(p) for p in body_local])

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
        sweep_path = args.save_path or str(save_dir / "body_recon_sweep.png")
        _plot_sweep(rows, body_local, polygons, segments, offsets, args.body_scale,
                    args.zoom_half, sweep_path, show)
        logger.info("扫描模式完成：请挑选躯干朝上、锅贴地、轮廓与身体相称的 offset 列")
        return

    recon = []
    for _cx, _cy, obs_row in rows:
        worlds, theta = _reconstruct(
            obs_row, body_local, offset_deg=args.body_angle_offset, scale=args.body_scale)
        n_in, n_tot = _classify(worlds, polygons)
        recon.append((worlds, theta, n_in, n_tot))

    total_in = sum(r[2] for r in recon)
    total_v = sum(r[3] for r in recon)
    logger.info("=" * 60)
    logger.info("body 重建统计（offset=%.0f scale=%.2f）：",
                args.body_angle_offset, args.body_scale)
    logger.info("  顶点入实体比例（越低越好，稳定位姿应大多贴地不深埋）: %d/%d (%.1f%%)",
                total_in, total_v, 100.0 * total_in / max(1, total_v))
    logger.info("=" * 60)

    csv_path = save_dir / "body_recon_samples.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "cand_x", "cand_y", "player_x", "player_y",
                    "hub_x", "hub_y", "theta", "n_in", "n_verts"])
        for idx, ((cx, cy, obs_row), (worlds, theta, n_in, n_tot)) in enumerate(
                zip(rows, recon)):
            w.writerow([idx, f"{cx:.3f}", f"{cy:.3f}",
                        f"{obs_row[IDX_PLAYER[0]]:.3f}", f"{obs_row[IDX_PLAYER[1]]:.3f}",
                        f"{obs_row[IDX_HUB[0]]:.3f}", f"{obs_row[IDX_HUB[1]]:.3f}",
                        f"{theta:.2f}", n_in, n_tot])
    logger.info("逐样本数据: %s", csv_path)

    grid_path = args.save_path or str(save_dir / "body_recon_grid.png")
    _plot_grid(rows, recon, polygons, segments, args.body_angle_offset, args.body_scale,
               args.zoom_half, grid_path, show)
    logger.info("请肉眼核对：body/pot 轮廓是否沿 player–hub 朝上、锅底贴地而非悬空/深埋")


if __name__ == "__main__":
    main()
