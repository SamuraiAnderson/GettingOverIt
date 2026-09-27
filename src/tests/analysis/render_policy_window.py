"""Render the dualscale patch the policy actually sees, without launching the game.

Frames match the motion-render marks: credited cap (required), arrival, prune.
Wide channel uses the corridor guidance map for that state's guidance_index.

    conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.render_policy_window --ranking runs/corridor_sil_td_pareto_20260927b/best_renders/ranking.json --output runs/corridor_sil_td_pareto_20260927b/best_renders
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.dataset import build_patch
from training.deploy_sampling import (
    extract_body_local,
    extract_tip_local,
    player_hub_angle_deg,
    reconstruct_polygon_world,
    rod_angle_deg,
)
from training.reference_anchor import scoring_context
from training.reference_corridor import ReferenceCorridor, replay_corridor_fields
from training.reward import ClimbingEfficiencyMap
from tests.analysis.render_episode_motion import corridor_score_marks
from tests.control_interaction.test_l7_surface_airdrop import extract_polygons, load_colliders

FONT = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "ch1_terrain",
    ["#cfc6b8", "#6b5344", "#2a1d16"],
)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def relative_xy(world_xy: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    return np.asarray(world_xy, np.float64) - np.asarray(center_xy, np.float64)


def apply_patch_size(cfg: TrainConfig, patch_size: int) -> TrainConfig:
    """Keep the same metric windows; only change how many cells cover them."""
    contact_m = cfg.patch_size * cfg.contact_patch_resolution
    wide_m = cfg.patch_size * cfg.wide_patch_resolution
    legacy_m = cfg.patch_size * cfg.patch_resolution
    return replace(
        cfg,
        patch_size=int(patch_size),
        contact_patch_resolution=contact_m / patch_size,
        wide_patch_resolution=wide_m / patch_size,
        patch_resolution=legacy_m / patch_size,
    )


def guidance_for_state(guidance: np.ndarray, state_index: int) -> int:
    return int(guidance[min(max(state_index, 0), len(guidance) - 1)])


def patch_at_state(
    state: np.ndarray,
    cfg: TrainConfig,
    eff_map: ClimbingEfficiencyMap,
    maps: list[np.ndarray],
    guidance_id: int,
    polygons,
    tip_local,
    body_local,
) -> np.ndarray:
    wide = maps[guidance_id] if guidance_id >= 0 else None
    return build_patch(
        state,
        eff_map.traversable,
        wide,
        cfg,
        eff_map._min_gx,
        eff_map._min_gy,
        solid_polygons=polygons,
        tip_local=tip_local,
        body_local=body_local,
    )


def _imshow_meters(ax, image, half_m, **kwargs):
    ax.imshow(
        image,
        origin="lower",
        extent=(-half_m, half_m, -half_m, half_m),
        interpolation="nearest",
        **kwargs,
    )
    ax.set_xlim(-half_m, half_m)
    ax.set_ylim(-half_m, half_m)
    ax.set_aspect("equal")


def _cross(ax, color="#24343e"):
    ax.axhline(0.0, color=color, linewidth=0.7, alpha=0.7)
    ax.axvline(0.0, color=color, linewidth=0.7, alpha=0.7)
    ax.plot(0.0, 0.0, marker="+", color=color, markersize=9, markeredgewidth=1.2, zorder=6)


def _intersects_window(rel: np.ndarray, half: float) -> bool:
    return bool(np.any((rel[:, 0] >= -half) & (rel[:, 0] <= half)
                       & (rel[:, 1] >= -half) & (rel[:, 1] <= half))) or (
        rel[:, 0].min() <= half and rel[:, 0].max() >= -half
        and rel[:, 1].min() <= half and rel[:, 1].max() >= -half
    )


def overlay_contact_geometry(ax, state: np.ndarray, cfg: TrainConfig,
                             polygons, tip_local, body_local, half: float) -> None:
    player = np.asarray(state[:2], np.float64)
    for poly in polygons:
        rel = np.asarray(poly, np.float64) - player
        if not _intersects_window(rel, half):
            continue
        ax.fill(rel[:, 0], rel[:, 1], color="#d2cbc0", edgecolor="#a59c8f",
                linewidth=0.5, alpha=0.9, zorder=2)
    if body_local:
        theta_b = player_hub_angle_deg(state) + cfg.body_angle_offset
        for i, local in enumerate(body_local):
            world = reconstruct_polygon_world(player, theta_b, local, cfg.body_scale)
            rel = np.asarray(world, np.float64) - player
            ax.fill(
                rel[:, 0], rel[:, 1],
                color=("#d9ae83" if i == 0 else "#4c6671"),
                edgecolor="#3f4a4e", linewidth=0.6, alpha=0.75, zorder=3,
            )
    if tip_local is not None and len(tip_local) > 0:
        world = reconstruct_polygon_world(
            (float(state[23]), float(state[24])),
            rod_angle_deg(state, cfg.tip_angle_source) + cfg.tip_angle_offset,
            tip_local, cfg.tip_scale,
        )
        rel = np.asarray(world, np.float64) - player
        ax.fill(rel[:, 0], rel[:, 1], color="#596169", edgecolor="#323b41",
                linewidth=0.6, alpha=0.85, zorder=4)


def _style_ax(ax, title: str, xlabel: str):
    ax.set_title(title, fontproperties=FONT, fontsize=11)
    ax.set_xlabel(xlabel, fontproperties=FONT, fontsize=9)
    ax.set_ylabel("相对 Player y（米）", fontproperties=FONT, fontsize=9)
    ax.tick_params(labelsize=8)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontproperties(FONT)


def draw_window(
    patch: np.ndarray,
    state: np.ndarray,
    cfg: TrainConfig,
    *,
    title: str,
    subtitle: str,
    caption: str,
    guidance_id: int,
    polygons,
    tip_local,
    body_local,
    output: Path,
    dpi: int = 120,
) -> Path:
    player = np.array([state[0], state[1]], np.float64)
    pot = np.array([state[33], state[34]], np.float64) if len(state) >= 35 else None
    tip = relative_xy(state[23:25], player)
    hub = relative_xy(state[5:7], player)
    cursor = relative_xy(state[29:31], player)
    pot_rel = None if pot is None else relative_xy(pot, player)

    wide_half = 0.5 * cfg.patch_size * cfg.wide_patch_resolution
    contact_half = 0.5 * cfg.patch_size * cfg.contact_patch_resolution
    contact_rgb = np.stack(
        [
            np.clip(0.20 * patch[1] + 0.95 * patch[3], 0, 1),
            np.clip(0.20 * patch[1] + 0.80 * patch[2], 0, 1),
            np.clip(0.38 * patch[1], 0, 1),
        ],
        axis=-1,
    )

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.4), facecolor="#faf9f6")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86, wspace=0.28, hspace=0.32)
    for ax in axes.ravel():
        ax.set_facecolor("#faf9f6")

    wide_ax = axes[0, 0]
    _imshow_meters(wide_ax, patch[0], wide_half, cmap="viridis")
    box = Rectangle(
        (-contact_half, -contact_half),
        2 * contact_half,
        2 * contact_half,
        fill=False,
        edgecolor="#f4f1ea",
        linewidth=1.1,
        linestyle="--",
        zorder=4,
    )
    wide_ax.add_patch(box)
    _cross(wide_ax)
    if pot_rel is not None:
        wide_ax.plot(pot_rel[0], pot_rel[1], "o", color="#f4f1ea", markersize=4, zorder=6)
    guide = "无引导" if guidance_id < 0 else f"引导段 {guidance_id}"
    wide_m = 2.0 * wide_half
    _style_ax(
        wide_ax,
        f"wide {wide_m:.0f}×{wide_m:.0f} m · {cfg.patch_size}px · {guide}",
        "相对 Player x（米）",
    )

    rgb_ax = axes[0, 1]
    _imshow_meters(rgb_ax, contact_rgb, contact_half, alpha=0.55)
    overlay_contact_geometry(rgb_ax, state, cfg, polygons, tip_local, body_local, contact_half)
    _cross(rgb_ax)
    rgb_ax.plot(hub[0], hub[1], "o", color="#8a5a2b", markersize=4, zorder=6, label="hub")
    rgb_ax.plot(tip[0], tip[1], "o", color="#24343e", markersize=4, zorder=6, label="tip")
    rgb_ax.plot(cursor[0], cursor[1], "x", color="#24343e", markersize=5, zorder=6, label="cursor")
    if pot_rel is not None:
        rgb_ax.plot(pot_rel[0], pot_rel[1], "o", mfc="none", mec="#24343e", markersize=6, zorder=6, label="锅心")
    rgb_ax.legend(prop=FONT, loc="upper right", fontsize=8, framealpha=0.85)
    _style_ax(rgb_ax, "contact 合成 + 轮廓", "相对 Player x（米）")

    axes[0, 2].axis("off")
    note = [
        f"Player 原点  ({player[0]:.2f}, {player[1]:.2f})",
    ]
    if pot is not None:
        note.append(f"锅心          ({pot[0]:.2f}, {pot[1]:.2f})")
        note.append(f"锅心相对原点  ({pot_rel[0]:+.2f}, {pot_rel[1]:+.2f}) m")
    note.append(f"tip 相对原点   ({tip[0]:+.2f}, {tip[1]:+.2f}) m")
    note.append(f"cursor 相对原点 ({cursor[0]:+.2f}, {cursor[1]:+.2f}) m")
    note.append(f"栅格          {cfg.patch_size}×{cfg.patch_size}")
    note.append(f"contact 一格  {cfg.contact_patch_resolution:.4f} m")
    note.append("十字 = 窗口中心 = Player 原点")
    note.append(f"虚线框 = contact {2.0 * contact_half:.2f}×{2.0 * contact_half:.2f} m")
    axes[0, 2].text(
        0.02, 0.95, "\n".join(note),
        va="top", ha="left", fontproperties=FONT, fontsize=10, color="#24343e",
        transform=axes[0, 2].transAxes,
    )

    for ax, channel, name, cmap, vmin in (
        (axes[1, 0], patch[1], "ch1 地形实心（CNN）", TERRAIN_CMAP, 0.0),
        (axes[1, 1], patch[2], "ch2 锅 / body（CNN）", "YlOrBr", None),
        (axes[1, 2], patch[3], "ch3 锤头（CNN）", "Blues", None),
    ):
        shown = channel if vmin is not None else np.ma.masked_less_equal(channel, 1e-4)
        kw = {"cmap": cmap}
        if vmin is not None:
            kw.update(vmin=0.0, vmax=1.0)
        _imshow_meters(ax, shown, contact_half, **kw)
        _cross(ax, color="#c0392b" if float(channel.max()) < 1e-6 else "#24343e")
        _style_ax(
            ax,
            f"{name} · {cfg.patch_size}×{cfg.patch_size} · {2.0 * contact_half:.2f} m",
            "相对 Player x（米）",
        )

    fig.suptitle(title, fontproperties=FONT, fontsize=15, y=0.97)
    fig.text(0.5, 0.905, subtitle, ha="center", fontproperties=FONT, fontsize=10, color="#24343e")
    fig.text(0.5, 0.02, caption, ha="center", fontproperties=FONT, fontsize=9, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def episode_frames(states: np.ndarray, data: dict, cfg: TrainConfig, route: ReferenceCorridor) -> list[tuple[str, int]]:
    schema, meta = scoring_context(states, config=cfg, route=route)
    _progress, guidance = replay_corridor_fields(route, states, schema, meta)
    marks = corridor_score_marks(states, cfg, route)
    frames = [("credited", int(marks["credited_state_index"]))]
    arrival = data.get("metrics", {}).get("arrival_steps")
    if arrival is not None:
        frames.append(("arrival", min(int(arrival), len(states) - 1)))
    frames.append(("prune", int(marks["prune_state_index"])))
    unique = []
    seen = set()
    for name, index in frames:
        key = (name, index)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, index, guidance_for_state(guidance, index), marks))
    return unique


def render_episode(episode: Path, output_dir: Path, cfg: TrainConfig, route: ReferenceCorridor,
                   eff_map: ClimbingEfficiencyMap, maps: list[np.ndarray],
                   polygons, tip_local, body_local, rank: int | None) -> list[dict]:
    data = json.loads(episode.read_text(encoding="utf-8"))
    states = np.asarray(data["states"], np.float32)
    if states.ndim != 2 or states.shape[1] not in (33, 35) or not np.isfinite(states).all():
        raise ValueError(f"Invalid episode states: {episode}")
    label = episode.stem
    rows = []
    for name, index, gid, marks in episode_frames(states, data, cfg, route):
        seconds = index * cfg.feature_step_seconds
        title = "策略视觉窗口" if rank is None else f"第 {rank} 名 · 策略视觉窗口"
        subtitle = (
            f"{label} · {name} · 第 {index} 步 / {seconds:.2f}s · "
            f"credited {marks['credited_progress_m']:.3f} m · "
            f"{cfg.patch_size}×{cfg.patch_size} @ {cfg.contact_patch_resolution:.4f} m/px"
        )
        caption = (
            "由 build_patch 重建，与训练/推理同一入口。wide 是当前引导段场。"
            f"合成图叠了同窗口内的真实轮廓；底下三张是 CNN 实际看到的 "
            f"{cfg.patch_size}×{cfg.patch_size} 栅格。"
        )
        path = output_dir / f"window_{name}.png"
        draw_window(
            patch_at_state(states[index], cfg, eff_map, maps, gid, polygons, tip_local, body_local),
            states[index],
            cfg,
            title=title,
            subtitle=subtitle,
            caption=caption,
            guidance_id=gid,
            polygons=polygons,
            tip_local=tip_local,
            body_local=body_local,
            output=path,
            dpi=max(120, int(round(1.2 * cfg.patch_size))),
        )
        rows.append({
            "frame": name,
            "state_index": index,
            "seconds": seconds,
            "guidance_index": gid,
            "path": str(path.resolve()),
        })
        print(path, flush=True)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--ranking", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="traversable_mask.npz；默认用该回合 checkpoint/cache")
    parser.add_argument(
        "--patch-size", type=int, default=None,
        help="覆盖格子数，米制窗口与回合配置相同（只加密栅格）",
    )
    args = parser.parse_args(argv)
    if args.ranking is None and args.episode is None:
        parser.error("需要 --episode 或 --ranking")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.ranking is not None:
        rows = json.loads(args.ranking.read_text(encoding="utf-8"))["episodes"][:args.top]
        episodes = [Path(row["path"]) for row in rows]
    else:
        episodes = [args.episode]
    first = json.loads((episodes[0].parent / "manifest.json").read_text(encoding="utf-8"))
    cfg = TrainConfig(**first["config"])
    if args.patch_size is not None:
        cfg = apply_patch_size(cfg, args.patch_size)
    env, player = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env)
    tip_local = extract_tip_local(player)
    body_local = extract_body_local(player)
    cache = args.cache
    if cache is None:
        cache = Path(cfg.checkpoint_dir) / "cache" / "traversable_mask.npz"
    eff_map = ClimbingEfficiencyMap(
        polygons,
        grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations,
        diffusion_alpha=cfg.diffusion_alpha,
        cache_path=str(cache) if cache.exists() else None,
        height_prior_weight=cfg.height_prior_weight,
    )
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    maps = route.guidance_maps(
        eff_map.traversable.shape, eff_map._min_gx, eff_map._min_gy, cfg.grid_resolution,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for rank, episode in enumerate(episodes, start=1):
        out = args.output if args.ranking is None else args.output / f"rank{rank:02d}"
        summary.append({
            "episode": str(episode.resolve()),
            "frames": render_episode(
                episode, out, cfg, route, eff_map, maps,
                polygons, tip_local, body_local,
                rank=None if args.ranking is None else rank,
            ),
        })
    if args.ranking is not None:
        (args.output / "window_index.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
