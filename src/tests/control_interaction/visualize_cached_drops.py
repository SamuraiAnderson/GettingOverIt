"""
Multi-panel hyperparameter visualization on terrain map.

Loads cached landable segments and environment polygons, then renders
a 2x3 grid of panels.  Each panel shares the same terrain base but
highlights a different group of spatially-meaningful hyperparameters.

Usage:
  python src/tests/control_interaction/visualize_cached_drops.py
  python src/tests/control_interaction/visualize_cached_drops.py --drop-height 3.0
  python src/tests/control_interaction/visualize_cached_drops.py --no-show
  python src/tests/control_interaction/visualize_cached_drops.py --physics-sample-n 20 --physics-filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.deploy_sampling import compute_drop_points, precompute_segment_arcs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_l7_surface_airdrop import (
    compute_landable_surfaces,
    compute_player_height,
    extract_polygons,
    load_colliders,
)


def load_env_data() -> dict:
    env_path = _REPO_ROOT / "checkpoints" / "environment.json"
    with open(env_path, "r", encoding="utf-8") as f:
        return json.load(f)


_EXCLUDED_COLLIDERS = {"Snake"}


def _draw_terrain_base(
    ax,
    env_data: dict,
    segments: list[np.ndarray],
    *,
    plt_mod=None,
    seg_color: str | None = None,
    seg_alpha: float = 0.6,
    seg_lw: float = 0.8,
):
    """Draw environment polygons and landable segments on *ax*."""
    from matplotlib.collections import PatchCollection

    if plt_mod is None:
        import matplotlib.pyplot as plt_mod  # noqa: N811

    env_patches = []
    for collider in env_data.get("colliders", []):
        if collider.get("name") in _EXCLUDED_COLLIDERS:
            continue
        for path in collider.get("paths", []):
            pts = np.array(path, dtype=np.float64)
            if len(pts) >= 3:
                env_patches.append(plt_mod.Polygon(pts, closed=True))
    if env_patches:
        pc = PatchCollection(
            env_patches, facecolor="#e0e0e0", edgecolor="#666666",
            linewidth=0.3, alpha=0.7, zorder=1,
        )
        ax.add_collection(pc)

    color = seg_color or "#2a9d8f"
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=seg_lw,
                alpha=seg_alpha, zorder=3)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def _pick_zoom_center(segments: list[np.ndarray], arc_lengths: np.ndarray):
    """Return (cx, cy) at the midpoint of the median-length segment."""
    median_idx = int(np.argsort(arc_lengths)[len(arc_lengths) // 2])
    seg = segments[median_idx]
    mid = seg[len(seg) // 2]
    return float(mid[0]), float(mid[1])


def _set_zoom(ax, cx: float, cy: float, half_w: float = 20.0, half_h: float = 15.0):
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)


# ── Panel renderers ─────────────────────────────────────────────


def _panel_overview(
    ax,
    env_data,
    segments,
    all_drop_pts,
    drop_height,
    total_arc,
    min_clearance,
    plt_mod,
    cmap,
    physics_sample_pts: np.ndarray | None = None,
    physics_stable_mask: np.ndarray | None = None,
    physics_probed_mask: np.ndarray | None = None,
):
    """ax[0,0] — Full overview with color-coded segments and drop points."""
    _draw_terrain_base(ax, env_data, segments, plt_mod=plt_mod, seg_lw=0.5)

    all_y = np.concatenate([s[:, 1] for s in segments])
    y_lo, y_hi = float(all_y.min()), float(all_y.max())
    y_range = y_hi - y_lo if y_hi > y_lo else 1.0
    for seg in segments:
        avg_y = seg[:, 1].mean()
        color = cmap((avg_y - y_lo) / y_range)
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.2,
                alpha=0.9, zorder=4)

    ax.scatter(
        all_drop_pts[:, 0], all_drop_pts[:, 1],
        c="red", s=3, marker=".", alpha=0.4, zorder=5,
        label=f"Drop points ({len(all_drop_pts)}, h={drop_height:.1f})",
    )

    if physics_sample_pts is not None and len(physics_sample_pts) > 0:
        n_sp = len(physics_sample_pts)
        if (
            physics_stable_mask is not None
            and len(physics_stable_mask) == n_sp
            and physics_probed_mask is not None
            and len(physics_probed_mask) == n_sp
        ):
            unprobed = ~physics_probed_mask
            if np.any(unprobed):
                up = physics_sample_pts[unprobed]
                ax.scatter(
                    up[:, 0], up[:, 1], c="silver", s=22, marker=".",
                    alpha=0.9, zorder=6,
                    label=f"not probed ({int(np.sum(unprobed))})",
                )
            probed = physics_probed_mask
            st = physics_sample_pts[probed & physics_stable_mask]
            ust = physics_sample_pts[probed & ~physics_stable_mask]
            if len(st) > 0:
                ax.scatter(
                    st[:, 0], st[:, 1], c="limegreen", s=36, marker="o",
                    edgecolors="darkgreen", linewidths=0.4, zorder=6,
                    label=f"physics stable ({len(st)})",
                )
            if len(ust) > 0:
                ax.scatter(
                    ust[:, 0], ust[:, 1], c="red", s=36, marker="x",
                    linewidths=1.2, zorder=6,
                    label=f"physics unstable ({len(ust)})",
                )
        elif physics_stable_mask is not None and len(physics_stable_mask) == n_sp:
            st = physics_sample_pts[physics_stable_mask]
            ust = physics_sample_pts[~physics_stable_mask]
            if len(st) > 0:
                ax.scatter(
                    st[:, 0], st[:, 1], c="limegreen", s=36, marker="o",
                    edgecolors="darkgreen", linewidths=0.4, zorder=6,
                    label=f"physics stable ({len(st)})",
                )
            if len(ust) > 0:
                ax.scatter(
                    ust[:, 0], ust[:, 1], c="red", s=36, marker="x",
                    linewidths=1.2, zorder=6,
                    label=f"physics unstable ({len(ust)})",
                )
        else:
            ax.scatter(
                physics_sample_pts[:, 0], physics_sample_pts[:, 1],
                c="magenta", s=28, marker="D", alpha=0.85, zorder=6,
                label=f"sample drops (n={len(physics_sample_pts)})",
            )

    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(
        f"Overview — {len(segments)} segments, "
        f"arc={total_arc:.0f}, clearance={min_clearance:.2f}",
        fontsize=10,
    )


def _panel_cutoff_lines(ax, env_data, segments, config, plt_mod):
    """ax[0,1] — y_max_cutoff and water_y_threshold horizontal lines."""
    _draw_terrain_base(ax, env_data, segments, plt_mod=plt_mod)

    all_x = np.concatenate([s[:, 0] for s in segments])
    x_lo, x_hi = float(all_x.min()), float(all_x.max())
    margin = (x_hi - x_lo) * 0.05

    ax.axhline(
        config.y_max_cutoff, color="red", linestyle="--", linewidth=1.5,
        zorder=6, label=f"y_max_cutoff = {config.y_max_cutoff}",
    )
    ax.annotate(
        f"y_max_cutoff = {config.y_max_cutoff}",
        xy=(x_hi - margin, config.y_max_cutoff),
        xytext=(0, 8), textcoords="offset points",
        fontsize=8, color="red", fontweight="bold",
        ha="right",
    )

    ax.axhline(
        config.water_y_threshold, color="dodgerblue", linestyle="--",
        linewidth=1.5, zorder=6,
        label=f"water_y_threshold = {config.water_y_threshold}",
    )
    ax.annotate(
        f"water_y_threshold = {config.water_y_threshold}",
        xy=(x_hi - margin, config.water_y_threshold),
        xytext=(0, 8), textcoords="offset points",
        fontsize=8, color="dodgerblue", fontweight="bold",
        ha="right",
    )

    info = (
        f"neg_reward_scale = {config.neg_reward_scale}\n"
        f"  (fall penalty: -scale * log(1+|dy|))"
    )
    ax.text(
        0.03, 0.03, info, transform=ax.transAxes,
        fontsize=7, fontfamily="monospace", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#aaa"),
        zorder=10,
    )

    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Y Cutoff Lines & Penalty", fontsize=10)


def _panel_drop_clearance(ax, env_data, segments, arc_lengths,
                          drop_height, min_clearance, config, plt_mod):
    """ax[0,2] — Zoomed view showing drop_height arrows and min_clearance."""
    _draw_terrain_base(ax, env_data, segments, plt_mod=plt_mod, seg_lw=1.5)

    cx, cy = _pick_zoom_center(segments, arc_lengths)
    _set_zoom(ax, cx, cy)

    median_idx = int(np.argsort(arc_lengths)[len(arc_lengths) // 2])
    seg = segments[median_idx]
    n_samples = min(5, len(seg))
    indices = np.linspace(0, len(seg) - 1, n_samples, dtype=int)

    for idx in indices:
        sx, sy = float(seg[idx, 0]), float(seg[idx, 1])
        dx, dy = sx, sy + drop_height
        ax.annotate(
            "", xy=(sx, sy), xytext=(dx, dy),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.2),
            zorder=7,
        )
        ax.plot(dx, dy, "o", color="blue", markersize=4, zorder=8)

    sample_sx, sample_sy = float(seg[indices[0], 0]), float(seg[indices[0], 1])
    ax.annotate(
        "", xy=(sample_sx + 1.0, sample_sy),
        xytext=(sample_sx + 1.0, sample_sy + min_clearance),
        arrowprops=dict(arrowstyle="<->", color="green", lw=1.5),
        zorder=7,
    )
    ax.text(
        sample_sx + 1.5, sample_sy + min_clearance / 2,
        f"min_clearance\n= {min_clearance:.2f}",
        fontsize=7, color="green", va="center", fontweight="bold",
        zorder=8,
    )

    info = (
        f"drop_height = {drop_height}\n"
        f"surface_padding = {config.surface_padding}\n"
        f"settle_drop_threshold = {config.settle_drop_threshold}"
    )
    ax.text(
        0.03, 0.97, info, transform=ax.transAxes,
        fontsize=7, fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#aaa"),
        zorder=10,
    )
    ax.set_title("Drop Height & Clearance (zoomed)", fontsize=10)


def _panel_grid_resolution(ax, env_data, segments, arc_lengths, config, plt_mod):
    """ax[1,0] — Grid overlay showing grid_resolution spacing + diffusion range."""
    _draw_terrain_base(ax, env_data, segments, plt_mod=plt_mod, seg_lw=1.5)

    cx, cy = _pick_zoom_center(segments, arc_lengths)
    hw, hh = 12.0, 9.0
    _set_zoom(ax, cx, cy, hw, hh)

    res = config.grid_resolution
    x_lo, x_hi = cx - hw, cx + hw
    y_lo, y_hi = cy - hh, cy + hh

    gx_start = np.floor(x_lo / res) * res
    gy_start = np.floor(y_lo / res) * res
    for gx in np.arange(gx_start, x_hi + res, res):
        ax.axvline(gx, color="orange", alpha=0.35, linewidth=0.5, zorder=2)
    for gy in np.arange(gy_start, y_hi + res, res):
        ax.axhline(gy, color="orange", alpha=0.35, linewidth=0.5, zorder=2)

    eff_range = np.sqrt(config.diffusion_iterations) * res
    circle = plt_mod.Circle(
        (cx, cy), eff_range,
        linewidth=1.8, edgecolor="crimson", facecolor="crimson",
        alpha=0.08, zorder=5,
    )
    ax.add_patch(circle)
    circle_border = plt_mod.Circle(
        (cx, cy), eff_range,
        linewidth=1.8, edgecolor="crimson", facecolor="none",
        linestyle="--", zorder=6,
    )
    ax.add_patch(circle_border)
    ax.plot(cx, cy, "x", color="crimson", markersize=6, markeredgewidth=1.5, zorder=7)

    ax.annotate(
        f"~{eff_range:.1f}m",
        xy=(cx + eff_range, cy), xytext=(6, 0), textcoords="offset points",
        fontsize=7, color="crimson", fontweight="bold", va="center",
        zorder=8,
    )

    info = (
        f"grid_resolution = {res} m/cell\n\n"
        f"Effective diffusion range:\n"
        f"  sqrt({config.diffusion_iterations}) * {res}\n"
        f"  = {eff_range:.1f} m"
    )
    ax.text(
        0.03, 0.97, info,
        transform=ax.transAxes, fontsize=7, fontfamily="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#aaa"),
        zorder=10,
    )
    ax.set_title("Grid Resolution & Diffusion Range (zoomed)", fontsize=10)


def _panel_patch_fov(ax, env_data, segments, arc_lengths, config, plt_mod):
    """ax[1,1] — Rectangle showing model's local patch field of view."""
    _draw_terrain_base(ax, env_data, segments, plt_mod=plt_mod, seg_lw=1.5)

    cx, cy = _pick_zoom_center(segments, arc_lengths)
    _set_zoom(ax, cx, cy)

    fov = config.patch_size * config.patch_resolution
    half = fov / 2.0
    rect = plt_mod.Rectangle(
        (cx - half, cy - half), fov, fov,
        linewidth=2, edgecolor="magenta", facecolor="magenta",
        alpha=0.12, zorder=6,
    )
    ax.add_patch(rect)
    rect_border = plt_mod.Rectangle(
        (cx - half, cy - half), fov, fov,
        linewidth=2, edgecolor="magenta", facecolor="none",
        linestyle="--", zorder=7,
    )
    ax.add_patch(rect_border)

    ax.plot(cx, cy, "+", color="magenta", markersize=10, markeredgewidth=2, zorder=8)

    ax.annotate(
        f"{fov:.0f}m",
        xy=(cx - half, cy), xytext=(-8, 0), textcoords="offset points",
        fontsize=8, color="magenta", fontweight="bold", ha="right", va="center",
        zorder=9,
    )
    ax.annotate(
        f"{fov:.0f}m",
        xy=(cx, cy - half), xytext=(0, -10), textcoords="offset points",
        fontsize=8, color="magenta", fontweight="bold", ha="center", va="top",
        zorder=9,
    )

    info = (
        f"patch_size = {config.patch_size}\n"
        f"patch_channels = {config.patch_channels}\n"
        f"patch_resolution = {config.patch_resolution} m/px\n"
        f"FOV = {fov:.0f} m x {fov:.0f} m"
    )
    ax.text(
        0.03, 0.97, info, transform=ax.transAxes,
        fontsize=7, fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#aaa"),
        zorder=10,
    )
    ax.set_title("Patch Field of View (zoomed)", fontsize=10)


def _panel_traversable_mask(ax, env_data, segments, config, plt_mod):
    """ax[1,2] — Traversable mask overlay, world bounds, diffusion & reward params."""
    _draw_terrain_base(ax, env_data, segments, plt_mod=plt_mod, seg_color="#555")

    mask_cache = Path(config.checkpoint_dir) / "cache" / "traversable_mask.npz"
    has_mask = False
    bounds_info = ""
    if mask_cache.exists():
        data = np.load(mask_cache)
        mask = data["mask"].astype(np.float32)
        min_gx, min_gy = int(data["min_gx"]), int(data["min_gy"])
        h, w = mask.shape
        res = config.grid_resolution

        x_world_lo = (min_gx + 0.5) * res
        x_world_hi = (min_gx + w - 0.5) * res
        y_world_lo = (min_gy + 0.5) * res
        y_world_hi = (min_gy + h - 0.5) * res
        extent = [x_world_lo, x_world_hi, y_world_lo, y_world_hi]

        ax.imshow(
            mask, origin="lower", extent=extent,
            cmap="Greens", alpha=0.45, zorder=2, interpolation="nearest",
        )

        bounds_rect = plt_mod.Rectangle(
            (x_world_lo, y_world_lo),
            x_world_hi - x_world_lo,
            y_world_hi - y_world_lo,
            linewidth=1.5, edgecolor="darkorange", facecolor="none",
            linestyle=":", zorder=6,
            label="Efficiency map bounds",
        )
        ax.add_patch(bounds_rect)

        bounds_info = (
            f"  grid = {w} x {h} cells\n"
            f"  X: [{x_world_lo:.0f}, {x_world_hi:.0f}]\n"
            f"  Y: [{y_world_lo:.0f}, {y_world_hi:.0f}]\n"
        )
        has_mask = True

    mask_label = "loaded" if has_mask else "NOT FOUND"
    eff_range = np.sqrt(config.diffusion_iterations) * config.grid_resolution
    info = (
        f"-- Traversable Mask ({mask_label}) --\n"
        f"grid_resolution      = {config.grid_resolution}\n"
        f"{bounds_info}\n"
        f"-- Diffusion --\n"
        f"diffusion_iterations = {config.diffusion_iterations}\n"
        f"diffusion_alpha      = {config.diffusion_alpha}\n"
        f"height_prior_weight  = {config.height_prior_weight}\n"
        f"eff. range           ~ {eff_range:.1f} m\n\n"
        f"-- Reward (updates eff. map) --\n"
        f"efficiency_weight    = {config.efficiency_weight}\n"
        f"waypoint_weight      = {config.waypoint_weight}\n"
        f"abs_height_weight    = {config.abs_height_weight}"
    )
    ax.text(
        0.03, 0.97, info, transform=ax.transAxes,
        fontsize=6.5, fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#aaa"),
        zorder=10,
    )
    if has_mask:
        ax.legend(loc="lower left", fontsize=7)
    ax.set_title("Traversable Mask, Diffusion & Reward Params", fontsize=10)


# ── Main ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-panel terrain hyperparameter visualization",
    )
    parser.add_argument("--drop-height", type=float, default=None,
                        help="Drop height override (default: config.drop_height)")
    parser.add_argument("--save", type=str, default="logs/cached_drops.png",
                        help="Output image path")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open interactive window")
    parser.add_argument(
        "--physics-sample-n",
        type=int,
        default=0,
        help="Overlay n representative drops from compute_drop_points (0=off)",
    )
    parser.add_argument(
        "--physics-filter",
        action="store_true",
        help="Requires --physics-sample-n>0; game probes each sample (stable=green, unstable=red)",
    )
    parser.add_argument(
        "--physics-no-launch",
        action="store_true",
        help="With --physics-filter: do not launch game (connect to running instance)",
    )
    parser.add_argument(
        "--settle-drop-threshold",
        type=float,
        default=None,
        help="Physics probe threshold (default TrainConfig.settle_drop_threshold)",
    )
    parser.add_argument(
        "--physics-settle-steps",
        type=int,
        default=None,
        help="Settle steps per probe (default TrainConfig.settle_steps)",
    )
    parser.add_argument(
        "--physics-warmup-steps",
        type=int,
        default=None,
        help="Warmup before snapshot (default TrainConfig.warmup_steps)",
    )
    parser.add_argument(
        "--physics-filter-max-probes",
        type=int,
        default=None,
        help="Cap probes (default: all physics-sample-n points)",
    )
    parser.add_argument(
        "--physics-sequential-probes",
        action="store_true",
        help="Use per-point 2-agent probes; default L8 batch deploy",
    )
    parser.add_argument(
        "--physics-l8-batch-size",
        type=int,
        default=64,
        help="L8 batch size 1–64 (default 64)",
    )
    args = parser.parse_args()

    config = TrainConfig()
    drop_height = args.drop_height or config.drop_height

    game_root = Path(config.game_root)
    cache_dir = Path(config.checkpoint_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "landable_segments.npz"

    if cache_path.exists():
        seg_data = np.load(cache_path, allow_pickle=True)
        segments = list(seg_data["segments"])
        min_clearance = float(seg_data["min_clearance"])
        print(f"Loaded {len(segments)} landable segments (cached, "
              f"min_clearance={min_clearance:.2f})")
    else:
        print("Cache not found, recomputing from collider data...")
        env_data_raw, player_data = load_colliders(game_root)
        polygons = extract_polygons(env_data_raw)
        player_height = compute_player_height(player_data)
        min_clearance = player_height + config.surface_padding
        segments = compute_landable_surfaces(
            polygons, min_clearance,
            resolution=0.1,
            y_max_cutoff=config.y_max_cutoff,
        )
        np.savez_compressed(
            cache_path,
            segments=np.array(segments, dtype=object),
            min_clearance=np.array(min_clearance),
        )
        print(f"Computed {len(segments)} landable segments "
              f"(min_clearance={min_clearance:.2f}, cached)")

    arc_lengths, cum_lengths = precompute_segment_arcs(segments)
    total_arc = float(arc_lengths.sum())
    print(f"Total arc length: {total_arc:.1f}")

    all_surface_pts = np.concatenate(segments, axis=0)
    all_drop_pts = all_surface_pts.copy()
    all_drop_pts[:, 1] += drop_height
    print(f"Total drop points: {len(all_drop_pts)} (drop_height={drop_height:.1f})")

    physics_sample_pts: np.ndarray | None = None
    physics_stable_mask: np.ndarray | None = None
    physics_probed_mask: np.ndarray | None = None
    if args.physics_filter and args.physics_sample_n <= 0:
        print("Error: --physics-filter requires --physics-sample-n > 0", file=sys.stderr)
        sys.exit(1)
    if args.physics_sample_n > 0:
        physics_sample_pts = compute_drop_points(
            segments, args.physics_sample_n, drop_height, height_decay=0.0,
        )
        print(f"Physics sample drops: {len(physics_sample_pts)}")
        if args.physics_filter:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from drop_point_physics import probe_drop_points_stability

            tc = TrainConfig()
            thr = (
                args.settle_drop_threshold
                if args.settle_drop_threshold is not None
                else tc.settle_drop_threshold
            )
            p_settle = (
                args.physics_settle_steps
                if args.physics_settle_steps is not None
                else tc.settle_steps
            )
            p_warm = (
                args.physics_warmup_steps
                if args.physics_warmup_steps is not None
                else tc.warmup_steps
            )
            game_root = Path(config.game_root)
            n_sp = len(physics_sample_pts)
            m = (
                n_sp
                if args.physics_filter_max_probes is None
                else min(n_sp, args.physics_filter_max_probes)
            )
            l8_bs = max(1, min(int(args.physics_l8_batch_size), 64))
            mask_m, _drops, _meta = probe_drop_points_stability(
                physics_sample_pts[:m],
                settle_drop_threshold=thr,
                settle_steps=p_settle,
                warmup_steps=p_warm,
                port=config.port,
                game_root=game_root,
                timeout=120.0,
                no_launch=args.physics_no_launch,
                max_probes=None,
                use_l8_batch_deploy=not args.physics_sequential_probes,
                l8_batch_size=l8_bs,
            )
            physics_probed_mask = np.zeros(n_sp, dtype=bool)
            physics_probed_mask[:m] = True
            physics_stable_mask = np.zeros(n_sp, dtype=bool)
            physics_stable_mask[:m] = mask_m
            n_stable = int(np.sum(mask_m))
            print(f"Physics probe: {n_stable}/{m} stable (sampled {n_sp}, probed {m})")

    env_data = load_env_data()

    import matplotlib
    matplotlib.use("TkAgg" if not args.no_show else "Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(36, 20))
    cmap = matplotlib.colormaps["viridis"]

    _panel_overview(
        axes[0, 0], env_data, segments, all_drop_pts,
        drop_height, total_arc, min_clearance, plt, cmap,
        physics_sample_pts=physics_sample_pts,
        physics_stable_mask=physics_stable_mask,
        physics_probed_mask=physics_probed_mask,
    )
    _panel_cutoff_lines(axes[0, 1], env_data, segments, config, plt)
    _panel_drop_clearance(
        axes[0, 2], env_data, segments, arc_lengths,
        drop_height, min_clearance, config, plt,
    )
    _panel_grid_resolution(
        axes[1, 0], env_data, segments, arc_lengths, config, plt,
    )
    _panel_patch_fov(
        axes[1, 1], env_data, segments, arc_lengths, config, plt,
    )
    _panel_traversable_mask(axes[1, 2], env_data, segments, config, plt)

    fig.suptitle(
        "Terrain & Efficiency-Map Hyperparameters",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        print(f"Image saved: {out}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
