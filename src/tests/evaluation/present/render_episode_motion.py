"""Render one archived physical trajectory, without launching or simulating the game."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"src"))
from training.config import TrainConfig
from training.dataset import Trajectory, prune_trajectory
from training.deploy_sampling import extract_body_local, extract_tip_local, reconstruct_polygon_world, player_hub_angle_deg, rod_angle_deg
from training.reference_anchor import scoring_context
from training.reference_corridor import (
    PROGRESS_MODE_V2, ReferenceCorridor, corridor_cap_index, progress_mode_of, replay_corridor_result,
)
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons


def physical_states(states: np.ndarray) -> np.ndarray:
    if states.ndim != 2 or states.shape[1] not in (33, 35):
        raise ValueError("Episode states must be 33D or 35D")
    return np.asarray(states[:, :33], np.float32)


def path_xy(states: np.ndarray) -> np.ndarray:
    """Pot-center trail when 35D is present; otherwise Player origin."""
    if states.shape[1] == 35:
        return np.asarray(states[:, 33:35], np.float64)
    return np.asarray(states[:, :2], np.float64)


def corridor_score_marks(states: np.ndarray, cfg: TrainConfig, route: ReferenceCorridor) -> dict:
    """封顶状态索引，以及剪枝后保留的最后一个状态索引。v2 的封顶是最后一次确认事件。"""
    schema, meta = scoring_context(states, config=cfg, route=route)
    played = replay_corridor_result(route, states, schema, meta, reject_invalid=False)
    traj = Trajectory(
        raw_states=states,
        actions=np.zeros((len(states) - 1, 2), np.float32),
        corridor_progress=np.asarray(played["legacy_credit"], np.float32),
        guidance_index=played["guidance_index"],
        corridor_credit_events=np.asarray(played["credit_events"], bool),
        corridor_confirmed_progress=np.asarray(played["confirmed_progress"], np.float64),
        progress_mode=played["mode"],
        trajectory_valid=bool(played["valid"]),
    )
    if progress_mode_of(cfg) == PROGRESS_MODE_V2 and played["mode"] == PROGRESS_MODE_V2:
        cap = int(played["cap"])
        distance = float(played["distance"])
        state_index = 0 if cap < 0 else min(cap + 1, len(states) - 1)
    else:
        progress = np.asarray(played["legacy_credit"], np.float64)
        cap = corridor_cap_index(progress)
        distance = float(progress.sum())
        state_index = min(int(cap) + 1, len(states) - 1)
    return {
        "credited_state_index": state_index,
        "prune_state_index": len(prune_trajectory(traj, cfg).raw_states) - 1,
        "credited_progress_m": distance,
    }


def render_episode(episode: Path, output: Path, terrain, bodies, tip, route: ReferenceCorridor,
                   rank: int | None = None, mark_prune: bool = False,
                   mark_cap: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(episode.read_text(encoding="utf-8"))
    manifest = json.loads((episode.parent/"manifest.json").read_text(encoding="utf-8"))
    cfg = TrainConfig(**manifest["config"])
    states = np.asarray(data["states"], np.float32)
    if (states.ndim != 2 or states.shape[0] != len(data["trace"]) + 1
            or states.shape[1] not in (33, 35) or not np.isfinite(states).all()):
        raise ValueError(f"Invalid raw physical-state trajectory: {episode}")
    phys = physical_states(states)
    trail = path_xy(states)
    marks = corridor_score_marks(states, cfg, route) if mark_prune or mark_cap else None
    peak = None if marks is None else marks["credited_state_index"]
    cut = marks["prune_state_index"] if mark_prune else None
    bounds = np.concatenate([trail, phys[:, [0, 1]], phys[:, [23, 24]]])
    low, high = bounds.min(axis=0)-1.6, bounds.max(axis=0)+1.6
    fig, ax = plt.subplots(figsize=(9.6, 6), facecolor="#faf9f6")
    ax.set_facecolor("#faf9f6")
    fig.subplots_adjust(left=.035, right=.985, bottom=.075, top=.87)
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    title = "原始状态回放" if rank is None else f"第 {rank} 名 · 原始状态回放"
    fig.suptitle(title, fontproperties=font, fontsize=17, y=.97)
    progress_name = "确认终点 D" if progress_mode_of(cfg) == PROGRESS_MODE_V2 else "累计有效进度"
    progress = (data["metrics"].get("cap_progress_m")
                if progress_mode_of(cfg) == PROGRESS_MODE_V2 else data["metrics"]["credited_progress_m"])
    if progress is None:
        progress = (marks or corridor_score_marks(states, cfg, route))["credited_progress_m"]
    label=f"{progress_name} {progress:.2f} 米 · {episode.parent.name}"
    if peak is not None:
        label += f" · 封顶于 {peak * cfg.feature_step_seconds:.2f}s / 第 {peak} 步"
    if cut is not None:
        label += f" · 裁剪于 {cut * cfg.feature_step_seconds:.2f}s / 第 {cut} 步"
    fig.text(.5, .91, label, ha="center", fontproperties=font, fontsize=10)
    caption = "根据保存状态重建的运动示意，非游戏录像；绿色线为参考路线。"
    if states.shape[1] == 35:
        caption += "蓝线是 TransformPoint(b) 锅心轨迹。"
    else:
        caption += "蓝线是 Player 原点轨迹。"
    if peak is not None:
        star = "金点为确认终点" if progress_mode_of(cfg) == PROGRESS_MODE_V2 else "金点为 credited 最高分（封顶）"
        caption += f"{star}。"
    if cut is not None:
        caption += "红点为裁剪点，红线为其后被剪掉的段。"
    fig.text(.5, .025, caption, ha="center", fontproperties=font, fontsize=9, color="#555555")
    for poly in terrain:
        if np.all(poly.max(axis=0) >= low) and np.all(poly.min(axis=0) <= high):
            ax.fill(poly[:, 0], poly[:, 1], color="#d2cbc0", edgecolor="#a59c8f", linewidth=.6, zorder=0)
    for segment in route.segments:
        ax.plot(segment.points[:, 0], segment.points[:, 1], color="#689573", linewidth=1.3, alpha=.8)
    patches=[]
    for i, local in enumerate(bodies or []):
        patch=Polygon(np.zeros_like(local), closed=True, facecolor=("#d9ae83" if i==0 else "#4c6671"), edgecolor="#3f4a4e", linewidth=1, zorder=4)
        ax.add_patch(patch); patches.append(patch)
    tip_patch=Polygon(np.zeros_like(tip), closed=True, facecolor="#596169", edgecolor="#323b41", zorder=5) if tip is not None else None
    if tip_patch is not None: ax.add_patch(tip_patch)
    kept,=ax.plot([],[],color="#32789d",linewidth=1.6,alpha=.85,zorder=2)
    dropped,=ax.plot([],[],color="#c0392b",linewidth=1.3,alpha=.75,zorder=2)
    peak_marker,=ax.plot([],[],marker="*",color="#d4a017",markersize=13,markeredgecolor="#fff7f0",markeredgewidth=0.8,zorder=7)
    marker,=ax.plot([],[],"o",color="#c0392b",markersize=7,markeredgecolor="#fff7f0",markeredgewidth=1.2,zorder=6)
    shaft,=ax.plot([],[],color="#946640",linewidth=3,zorder=3)
    hand,=ax.plot([],[],"o",color="#d9ae83",markersize=4,zorder=5)
    timer=ax.text(.02,.97,"",transform=ax.transAxes,va="top",fontsize=12,color="#24343e",fontproperties=font)
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]))
    ax.set_aspect("equal"); ax.set_axis_off()
    def draw(i):
        row=phys[i]
        split = len(states)-1 if cut is None else cut
        kept.set_data(trail[:min(i, split)+1,0], trail[:min(i, split)+1,1])
        if peak is not None and i >= peak:
            peak_marker.set_data([trail[peak,0]], [trail[peak,1]])
        else:
            peak_marker.set_data([], [])
        if cut is not None and i >= cut:
            marker.set_data([trail[cut,0]], [trail[cut,1]])
            if i > cut:
                dropped.set_data(trail[cut:i+1,0], trail[cut:i+1,1])
            else:
                dropped.set_data([], [])
        else:
            dropped.set_data([], [])
            marker.set_data([], [])
        shaft.set_data([row[15],row[23]],[row[16],row[24]])
        hand.set_data([row[5]],[row[6]])
        angle=player_hub_angle_deg(row)+cfg.body_angle_offset
        for local, patch in zip(bodies or [],patches):
            patch.set_xy(reconstruct_polygon_world(row[:2],angle,local,cfg.body_scale))
        if tip_patch is not None:
            tip_patch.set_xy(reconstruct_polygon_world(row[23:25],rod_angle_deg(row,cfg.tip_angle_source)+cfg.tip_angle_offset,tip,cfg.tip_scale))
        note = f"{i*cfg.feature_step_seconds:4.2f} / {(len(states)-1)*cfg.feature_step_seconds:.2f} s"
        if peak is not None and i >= peak:
            note += "\ncredited 封顶之后"
        if cut is not None and i >= cut:
            note += "\n裁剪点之后"
        timer.set_text(note)
    frames=sorted(set([*range(0,len(states),4),len(states)-1]))
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg: raise RuntimeError("ffmpeg is required for the single-episode video")
    matplotlib.rcParams['animation.ffmpeg_path']=ffmpeg
    writer=FFMpegWriter(fps=30,codec="libx264",bitrate=1600,extra_args=["-pix_fmt","yuv420p"])
    with writer.saving(fig,str(output/"best-motion.mp4"),dpi=100):
        for i in frames:
            draw(i); writer.grab_frame()
    for i,name in [(0,"first"),(len(states)//2,"middle"),(len(states)-1,"last")]:
        draw(i); fig.savefig(output/f"{name}.png",dpi=100)
    if peak is not None:
        draw(peak)
        fig.savefig(output/"credited.png",dpi=120)
    arrival = data["metrics"].get("arrival_steps")
    if arrival is not None:
        draw(min(int(arrival), len(states) - 1))
        fig.savefig(output/"arrival.png",dpi=120)
    if cut is not None:
        draw(len(states)-1)
        fig.savefig(output/"prune.png",dpi=120)
    plt.close(fig)
    meta = {"episode":str(episode.resolve()),"physical_steps":len(states)-1,
            "physical_seconds":(len(states)-1)*cfg.feature_step_seconds,
            "rendered_frames":len(frames),"fps":30,
            "type":"archived_state_reconstruction_not_game_recording",
            "trail": "pot_transform_point_b" if states.shape[1] == 35 else "player_origin",
            "credited_progress_m": data["metrics"]["credited_progress_m"]}
    meta["displayed_progress_m"] = progress
    meta["displayed_progress_name"] = progress_name
    if peak is not None:
        meta["credited_state_index"] = int(peak)
        meta["credited_seconds"] = peak * cfg.feature_step_seconds
    if cut is not None:
        meta["prune_state_index"] = int(cut)
        meta["prune_seconds"] = cut * cfg.feature_step_seconds
    (output/"source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(output/"best-motion.mp4", flush=True)
    return meta


def sampling_path_probabilities(data, trajectories):
    """Map action t's inclusion probability to the physical segment state[t:t+2]."""
    from training.reference_corridor import trajectory_endpoint
    if data.get("kind") != "offline_sampling_preview_before_advantage_and_KL":
        raise ValueError("Expected an offline SIL sampling report")
    result = {}
    for mode in data.get("comparison_modes", ("trajectory_uniform", "phase_mixture_v1")):
        arrays = [np.zeros(len(t.actions), np.float64) for t in trajectories]
        covered = [np.zeros(len(t.actions), bool) for t in trajectories]
        total = 0.
        for row in data["modes"][mode]["chunks"]:
            i, start, end, cap = (int(row[k]) for k in ("trajectory", "start", "end", "cap"))
            p = float(row["probability"])
            if (not 0 <= i < len(trajectories) or not np.isfinite(p) or not 0 <= p <= 1
                    or not 0 <= start < end <= cap <= len(trajectories[i].actions)
                    or cap != trajectory_endpoint(trajectories[i]).t_cap):
                raise ValueError("Sampling chunk does not match the physical trajectory")
            if covered[i][start:end].any():
                raise ValueError("Overlapping sampling chunks")
            arrays[i][start:end] = p
            covered[i][start:end] = True
            total += p
        for i, traj in enumerate(trajectories):
            cap = trajectory_endpoint(traj).t_cap
            if not covered[i][:cap].all():
                raise ValueError("Sampling report is missing a trajectory prefix")
        if not np.isclose(total, 1., rtol=0, atol=1e-10):
            raise ValueError("Sampling probabilities must sum to one across chunks")
        result[mode] = arrays
    return result


def render_sampling_paths(pool, sampling_report, output, trajectories, order, terrain, route, low, high, focus=None):
    """Put the existing sampling audit directly on archived pot-center trajectories."""
    from training.reference_corridor import trajectory_endpoint
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from training.climb_experiment import identity
    data = json.loads(sampling_report.read_text(encoding="utf-8"))
    if data["source_pool"]["sha256"] != hashlib.sha256(pool.read_bytes()).hexdigest():
        raise ValueError("Sampling report was produced from a different elite pool")
    probabilities = sampling_path_probabilities(data, trajectories)
    if focus is not None and (not 0 <= focus[0] < len(trajectories)
                             or not 0 <= focus[1] < trajectory_endpoint(trajectories[focus[0]]).t_cap):
        raise ValueError("Sampling focus must be an eligible pool index and pre-cap action")
    modes = data.get("comparison_modes", ("trajectory_uniform", "phase_mixture_v1"))
    from .plot_sil_td_envelope import SAMPLING_NAMES
    names = SAMPLING_NAMES
    vmax = max(float(p.max())*100 for arrays in probabilities.values() for p in arrays)
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("YlGnBu")
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    legends = [Line2D([], [], color="#a5a5a5", ls="--", label="封顶后尾段：不采样"),
               Line2D([], [], color="#6b9576", ls=":", label="参考路线"),
               Line2D([], [], marker="o", mfc="white", mec="#333333", ls="", label="出生点"),
               Line2D([], [], marker="*", color="#bd8610", ls="", label="封顶点"),
               Line2D([], [], marker="s", color="#333333", ls="", label="完整轨迹末端")]

    def draw(ax, index, rank, mode, *, annotate=False):
        trail = path_xy(trajectories[index].raw_states)
        endpoint = trajectory_endpoint(trajectories[index]); cap = endpoint.t_cap
        for poly in terrain:
            if np.all(poly.max(axis=0) >= low) and np.all(poly.min(axis=0) <= high):
                ax.fill(poly[:, 0], poly[:, 1], color="#ebe7df", edgecolor="#c9c3b9", lw=.4, zorder=0)
        for segment in route.segments:
            ax.plot(*segment.points.T, color="#6b9576", lw=.9, ls=":", alpha=.7, zorder=1)
        ax.plot(*trail[cap:].T, color="#999999", lw=1., ls="--", alpha=.65, zorder=2)
        # A thin dark outline keeps low-probability yellow segments visible against terrain.
        ax.plot(*trail[:cap+1].T, color="#555555", lw=3.5, alpha=.75, zorder=3)
        segments = np.stack((trail[:cap], trail[1:cap+1]), axis=1)
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.6, zorder=4)
        collection.set_array(probabilities[mode][index][:cap]*100)
        ax.add_collection(collection)
        ax.plot(*trail[0], marker="o", mfc="white", mec="#333333", ms=4, zorder=6)
        ax.plot(*trail[cap], marker="*", color="#bd8610", ms=11, mec="white", mew=.5, zorder=6)
        ax.plot(*trail[-1], marker="s", color="#333333", ms=4, zorder=6)
        if annotate:
            chunks = [c for c in data["modes"][modes[1]]["chunks"] if c["trajectory"] == index]
            # Label the same two high-probability chunks in both views for direct comparison.
            chosen = sorted(chunks, key=lambda c: (-c["probability"], c["start"]))[:2]
            if focus is not None and index == focus[0]:
                focal = next(c for c in chunks if c["start"] <= focus[1] < c["end"])
                chosen = [focal, *[c for c in chosen if c is not focal]][:2]
            for j, row in enumerate(chosen):
                start, end = row["start"], row["end"]
                focused = focus is not None and index == focus[0] and start <= focus[1] < end
                xy = trail[focus[1] if focused else (start+end)//2]
                if focused: ax.plot(*xy, "o", ms=5, mfc="white", mec="#b53f37", zorder=8)
                ax.annotate(f"{start}–{end-1} 步\n{probabilities[mode][index][start]*100:.2f}%",
                            xy, xytext=((-42, 26) if j == 0 else (12, -35)), textcoords="offset points",
                            fontproperties=font, fontsize=8,
                            bbox=dict(boxstyle="round,pad=.2", fc="white", ec="#a0a0a0", alpha=.93),
                            arrowprops=dict(arrowstyle="-", color="#555555", lw=.7), zorder=8)
        ax.set_title(f"{rank:02d} · 池 {index:02d} · D={endpoint.distance:.2f} m · {names[mode]}\n"
                     f"封顶 t={cap} 步；第 {trajectories[index].iteration} 轮精英", fontproperties=font, fontsize=10)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="World x (m)", ylabel="World y (m)")
        ax.set_aspect("equal"); ax.tick_params(labelsize=8); ax.grid(alpha=.1)
        return collection

    def decorate(fig, axes, title):
        fig.suptitle(title, fontproperties=font, fontsize=17)
        scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        bar = fig.colorbar(scalar, ax=np.asarray(axes).ravel().tolist(), shrink=.8, pad=.02)
        bar.set_label("单次抽片段包含该动作的概率（%，越深越常采）", fontproperties=font)
        fig.supxlabel("锅心真实空间轨迹；每段颜色对应动作 t 导致的状态 t → t+1，所有小图同坐标和色标。\n"
                      "灰虚线仍完整保存；颜色为筛选前的采样机会，非已接受更新或策略成功率。\n"
                      f"来源：{pool.name}；完整池 {len(trajectories)} 条；几何与参考路线同此前轨迹图。",
                      fontproperties=font, fontsize=9)

    rows = int(np.ceil(len(order)/4))
    fig, axes = plt.subplots(rows, 4, figsize=(18, rows*2.7+1.2), squeeze=False, layout="constrained", facecolor="#faf9f6")
    for rank, index in enumerate(order, 1):
        draw(axes.flat[rank-1], index, rank, modes[1])
    for ax in list(axes.flat)[len(order):]: ax.set_visible(False)
    decorate(fig, axes, f"精英池空间轨迹上的 SIL 采样机会 · {names[modes[1]]}")
    fig.legend(handles=legends, loc="outside upper center", bbox_to_anchor=(.48,.971), ncol=5, prop=font, frameon=False)
    fig.get_layout_engine().set(rect=(0,0,1,.95))
    gallery = output/"all_elite_sampling.png"
    fig.savefig(gallery, dpi=150); plt.close(fig)

    # Paired detail views retain the exact same physical path, terrain and color normalization.
    entries = []
    for rank, index in enumerate(order, 1):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), layout="constrained", facecolor="#faf9f6")
        for ax, mode in zip(axes, modes): draw(ax, index, rank, mode, annotate=True)
        decorate(fig, axes, f"第 {rank} 条精英：只改变采样机会，物理轨迹相同")
        fig.legend(handles=legends, loc="outside upper center", bbox_to_anchor=(.47,.936), ncol=3, prop=font, frameon=False)
        fig.get_layout_engine().set(rect=(0,0,1,.85))
        path = output/f"elite_{index:02d}_sampling.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        entries.append({"rank":rank, "pool_index":index, "image":path.name,
                        "cap_step":trajectory_endpoint(trajectories[index]).t_cap})
    meta = {"source_pool":identity(pool), "sampling_report":identity(sampling_report), "renderer":identity(Path(__file__)),
            "kind":"archived_spatial_trajectory_with_expected_sampling_probability",
            "comparison_modes": modes,
            "focus_pool_index_action": focus,
            "probability_units":"percent inclusion per chunk draw before advantage/KL filters",
            "segment_alignment":"action t -> states[t] to states[t+1]", "vmin":0, "vmax":vmax,
            "shared_bounds":[low.tolist(),high.tolist()], "reference_sha256":route.metadata["sha256"],
            "gallery":gallery.name, "entries":entries}
    (output/"sampling_trajectories.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(gallery, flush=True)
    print(output/entries[0]["image"], flush=True)


def render_elite_pool(pool: Path, manifest_path: Path, output: Path, *, video=False, sampling_report=None,
                      sampling_focus=None):
    """Plot every complete elite on common world axes, retaining its entire tail."""
    from training.elite_archive import EliteTrajectoryArchive
    from training.reference_fingerprint import reference_fingerprint
    from training.reference_corridor import trajectory_endpoint
    from training.state_schema import schema_from_raw_width
    from matplotlib.lines import Line2D

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    cfg = TrainConfig(**manifest["config"])
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    if route.metadata["sha256"] != manifest["reference"]["sha256"]:
        raise ValueError("Reference file differs from the archived manifest")
    with np.load(pool) as data:
        width = data["t0_states"].shape[1]
    fingerprint = reference_fingerprint(cfg, route_meta=route.metadata,
                                        schema_id=schema_from_raw_width(width).schema_id, raw_dim=width)
    archive = EliteTrajectoryArchive(cfg, pool.parent, fingerprint)
    archive.restore(pool)
    trajectories = archive.trajectories
    if not trajectories:
        raise ValueError("Elite pool is empty")
    order = sorted(range(len(trajectories)), key=lambda i: (
        -trajectory_endpoint(trajectories[i]).distance, trajectory_endpoint(trajectories[i]).t_cap, i))
    env, _ = load_colliders(Path(cfg.game_root))
    terrain = extract_polygons(env)
    trails = [path_xy(t.raw_states) for t in trajectories]
    points = np.concatenate(trails + [t.raw_states[:, 23:25] for t in trajectories])
    low, high = points.min(axis=0) - 1.5, points.max(axis=0) + 1.5
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    output.mkdir(parents=True, exist_ok=True)
    if sampling_report is not None:
        if video:
            raise ValueError("Sampling overlay is a static view; omit --pool-video")
        return render_sampling_paths(pool, sampling_report, output, trajectories, order, terrain, route, low, high,
                                     focus=sampling_focus)
    colors = {"prefix": "#226a91", "tail": "#a75e42", "route": "#6b9576"}
    legends = [Line2D([], [], color=colors["prefix"], label="最远确认前"),
               Line2D([], [], color=colors["tail"], label="最远确认后（完整保留）"),
               Line2D([], [], color=colors["route"], label="参考路线"),
               Line2D([], [], marker="*", color="#c69716", ls="", label="最远确认点"),
               Line2D([], [], marker="s", color="#303a43", ls="", label="轨迹末端")]

    def panel(ax, index, rank):
        traj, trail = trajectories[index], trails[index]
        endpoint = trajectory_endpoint(traj)
        cap = endpoint.t_cap
        for poly in terrain:
            if np.all(poly.max(axis=0) >= low) and np.all(poly.min(axis=0) <= high):
                ax.fill(poly[:, 0], poly[:, 1], color="#e4e0d9", edgecolor="#c4bdb3", lw=.4)
        for segment in route.segments:
            ax.plot(segment.points[:, 0], segment.points[:, 1], color=colors["route"], lw=.8, alpha=.8)
        ax.plot(trail[:cap + 1, 0], trail[:cap + 1, 1], color=colors["prefix"], lw=1.3)
        ax.plot(trail[cap:, 0], trail[cap:, 1], color=colors["tail"], lw=1., alpha=.75)
        ax.plot(*trail[0], marker="o", color=colors["prefix"], ms=3)
        ax.plot(*trail[cap], marker="*", color="#c69716", ms=9, mec="white", mew=.4, zorder=5)
        ax.plot(*trail[-1], marker="s", color="#303a43", ms=3, zorder=5)
        role = "前沿" if traj.archive_role == "pareto" else "多样性"
        ax.set_title(f"{rank:02d} · 池索引 {index} · 第 {traj.iteration} 轮 · {role}\n"
                     f"D={endpoint.distance:.2f} m · t={cap} 步 / {cap*cfg.feature_step_seconds:.2f} s",
                     fontproperties=font, fontsize=9)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="World x (m)", ylabel="World y (m)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.12)
        return endpoint

    cols = 4
    rows = int(np.ceil(len(order)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows*3.6), squeeze=False, facecolor="#faf9f6")
    fig.subplots_adjust(top=.90, bottom=.055, left=.045, right=.99, hspace=.48, wspace=.30)
    title = fig.suptitle(f"完整精英池 · {len(order)} 条运动轨迹", fontproperties=font, fontsize=20, y=.986)
    fig.text(.5, .959, f"{pool.name} · 按确认距离 D 降序 · 所有小图同尺度 · D 为路线进度，不是高度",
             ha="center", fontproperties=font, fontsize=10)
    fig.legend(handles=legends, loc="upper center", bbox_to_anchor=(.5, .948), ncol=5, prop=font, frameon=False)
    fig.text(.5, .018, "保存状态重建；线条为锅心轨迹。每条完整 1200 步 = 10 秒；与当前策略评估能力分开解读。"
             if all(len(t.actions)==1200 for t in trajectories) else "保存状态重建；线条为锅心轨迹，末端停留到共同时间轴结束。",
             ha="center", fontproperties=font, fontsize=9)
    entries, moving = [], []
    for rank, index in enumerate(order, 1):
        ax = axes.flat[rank-1]
        endpoint = panel(ax, index, rank)
        traj = trajectories[index]
        entries.append({"rank":rank, "pool_index":index, "source_iteration":int(traj.iteration),
                        "role":traj.archive_role, "distance_m":endpoint.distance, "cap_step":endpoint.t_cap,
                        "steps":len(traj.actions), "image":f"elite_{index:02d}.png"})
        single, single_ax = plt.subplots(figsize=(8, 7))
        panel(single_ax, index, rank)
        single.legend(handles=legends, loc="lower center", ncol=3, prop=font, frameon=False)
        single.subplots_adjust(bottom=.16, top=.89)
        single.savefig(output/f"elite_{index:02d}.png", dpi=150)
        plt.close(single)
        dot, = ax.plot([], [], "o", color="#122b3b", ms=4, zorder=7)
        shaft, = ax.plot([], [], color="#122b3b", lw=1.7, zorder=6)
        moving.append((dot, shaft, traj))
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    fig.savefig(output/"all_elite_trajectories.png", dpi=170)
    if video:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for pool video")
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
        end = max(len(t.actions) for t in trajectories)
        stride = max(1, round(1/(30*cfg.feature_step_seconds)))
        frames = sorted(set([*range(0, end+1, stride), end]))
        writer = FFMpegWriter(fps=30, codec="libx264", bitrate=2800, extra_args=["-pix_fmt", "yuv420p"])
        with writer.saving(fig, str(output/"all_elite_motion.mp4"), dpi=80):
            for step in frames:
                for dot, shaft, traj in moving:
                    row = traj.raw_states[min(step, len(traj.actions))]
                    xy = row[33:35] if len(row)==35 else row[:2]
                    dot.set_data([xy[0]], [xy[1]])
                    shaft.set_data([row[15], row[23]], [row[16], row[24]])
                title.set_text(f"完整精英池 · {len(order)} 条运动轨迹 · {step*cfg.feature_step_seconds:.2f} / {end*cfg.feature_step_seconds:.2f} 秒")
                writer.grab_frame()
    plt.close(fig)
    metadata = {"source_pool":str(pool.resolve()), "source_sha256":hashlib.sha256(pool.read_bytes()).hexdigest(),
                "manifest":str(manifest_path.resolve()), "type":"archived_state_reconstruction_not_game_recording",
                "trail":"pot_transform_point_b" if width==35 else "player_origin", "unpruned":True,
                "reference_sha256":route.metadata["sha256"], "shared_bounds":[low.tolist(),high.tolist()], "entries":entries}
    (output/"index.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output/"all_elite_trajectories.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, help="评估排名 JSON；与 --top 一起渲染前若干条")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--mark-prune", action="store_true", help="按走廊 credited 封顶口径标出裁剪点")
    parser.add_argument("--mark-cap", action="store_true", help="标出最远确认点，保留完整轨迹显示")
    parser.add_argument("--elite-pool", type=Path, help="绘制完整精英池的全部运动轨迹")
    parser.add_argument("--manifest", type=Path, help="精英池模式所需的 config/reference manifest")
    parser.add_argument("--pool-video", action="store_true", help="额外生成所有精英同步运动的视频")
    parser.add_argument("--sampling-report", type=Path, help="将 sampling.json 的采样概率标在完整精英空间轨迹上")
    parser.add_argument("--sampling-focus", nargs=2, type=int, metavar=("POOL_INDEX", "ACTION"),
                        help="在并排图上额外标注指定池索引和动作所在的采样片段")
    args = parser.parse_args()
    if args.sampling_focus and not args.sampling_report:
        parser.error("--sampling-focus 需要 --sampling-report")
    if args.elite_pool:
        if args.manifest is None or args.episode or args.ranking:
            parser.error("--elite-pool 需要 --manifest，且不能与 --episode/--ranking 同用")
        render_elite_pool(args.elite_pool, args.manifest, args.output, video=args.pool_video,
                          sampling_report=args.sampling_report, sampling_focus=args.sampling_focus)
        return
    if args.sampling_report:
        parser.error("--sampling-report 需要 --elite-pool")
    if args.ranking is None and args.episode is None:
        parser.error("需要 --episode 或 --ranking")
    if args.ranking is not None:
        rows = json.loads(args.ranking.read_text(encoding="utf-8"))["episodes"][:args.top]
        episodes = [Path(row["path"]) for row in rows]
    else:
        episodes = [args.episode]
    first = json.loads((episodes[0].parent/"manifest.json").read_text(encoding="utf-8"))
    cfg = TrainConfig(**first["config"])
    env, player = load_colliders(Path(cfg.game_root))
    terrain, bodies, tip = extract_polygons(env), extract_body_local(player), extract_tip_local(player)
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for rank, episode in enumerate(episodes, start=1):
        out = args.output if args.ranking is None else args.output / f"rank{rank:02d}"
        summary.append(render_episode(episode, out, terrain, bodies, tip, route,
                                       rank=None if args.ranking is None else rank,
                                       mark_prune=args.mark_prune, mark_cap=args.mark_cap))
    if args.ranking is not None:
        (args.output/"index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__=="__main__": main()
