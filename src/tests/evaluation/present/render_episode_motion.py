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
    handoff = data["metrics"].get("prefix_handoff")
    noise_window = data["metrics"].get("noise_intervention")
    return_segment = data["metrics"].get("return_segment")
    handoff_step = None if handoff is None else int(handoff["prefix_steps"])
    if handoff is not None:
        title = ("纯策略回放（对照）" if handoff_step == 0 else "完整示范回放（对照）"
                 if handoff_step >= len(states)-1 else f"示范前 {handoff_step} 步 → 冻结策略接管")
    if noise_window is not None:
        title = f"冻结策略 · 噪声窗口 [{noise_window['start']}, {noise_window['stop']}) 步"
    if return_segment is not None:
        title = f"离开—返回片段 {return_segment['display_rank']:02d} · 原始状态回放"
    fig.suptitle(title, fontproperties=font, fontsize=17, y=.97)
    progress_name = "确认终点 D" if progress_mode_of(cfg) == PROGRESS_MODE_V2 else "累计有效进度"
    progress = (return_segment["delta_s_m"] if return_segment is not None else data["metrics"].get("cap_progress_m")
                if progress_mode_of(cfg) == PROGRESS_MODE_V2 else data["metrics"]["credited_progress_m"])
    if progress is None:
        progress = (marks or corridor_score_marks(states, cfg, route))["credited_progress_m"]
    if return_segment is not None:
        progress_name, progress = "正向投影跨度", return_segment["delta_s_m"]
    label=f"{progress_name} {progress:.2f} 米 · {episode.parent.name}"
    if handoff is not None:
        source = manifest["prefix_handoff"]["source"]["iteration"]
        checkpoint = Path(manifest["checkpoint"]).stem
        label = f"{progress_name} {progress:.2f} 米 · 示范 {source} · {checkpoint}"
    if peak is not None:
        label += f" · 封顶于 {peak * cfg.feature_step_seconds:.2f}s / 第 {peak} 步"
    if cut is not None:
        label += f" · 裁剪于 {cut * cfg.feature_step_seconds:.2f}s / 第 {cut} 步"
    if return_segment is not None:
        label = (f"s 起点={return_segment['s_start_m']:.3f} m · Δs={progress:.3f} m · "
                 f"T={return_segment['T_steps']} 步 / {return_segment['T_seconds']:.3f} s\n"
                 f"原回合第 {return_segment['start_step']} → {return_segment['end_step']} 步 · "
                 f"{return_segment['source_label']}")
    if return_segment is not None:
        fig.subplots_adjust(top=.80)
    fig.text(.5, .89 if return_segment is not None else .91, label, ha="center",
             va="center" if return_segment is not None else "baseline", fontproperties=font, fontsize=10)
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
    if handoff is not None:
        caption = "保存状态重建，非游戏录像；灰虚线为示范前缀，蓝线为策略接管后；黑方块为接管点，绿色为参考路线。"
    if noise_window is not None:
        caption = "保存状态重建，非游戏录像；灰虚线为加噪前，橙线为噪声窗口，蓝线为撤噪后；黑方块为撤噪点，绿色为参考路线。"
        if noise_window["stop"] >= len(states)-1:
            caption = "保存状态重建，非游戏录像；橙线为全程探索，绿色为参考路线；本条没有撤噪后的观测。"
    if return_segment is not None:
        caption = "保存状态重建，非游戏录像；橙线为入选片段，灰线为前后上下文；蓝圆为离开前，紫方块为首次返回，绿色为参考路线。"
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
    prefix_line,=ax.plot([],[],color="#777777",linewidth=1.6,ls="--",zorder=2)
    noise_line,=ax.plot([],[],color="#df861d",linewidth=2,zorder=3)
    handoff_marker,=ax.plot([],[],marker="s",color="#333333",markersize=5,zorder=7)
    return_start_marker,=ax.plot([],[],marker="o",color="#246b9b",markersize=6,zorder=7)
    return_end_marker,=ax.plot([],[],marker="s",color="#824da1",markersize=6,zorder=7)
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
        if handoff_step is not None:
            end = min(i, handoff_step)
            prefix_line.set_data(trail[:end+1,0], trail[:end+1,1])
            kept.set_data(trail[handoff_step:i+1,0], trail[handoff_step:i+1,1])
            if handoff_step < len(states)-1 and i >= handoff_step:
                handoff_marker.set_data([trail[handoff_step,0]], [trail[handoff_step,1]])
            else:
                handoff_marker.set_data([], [])
        if noise_window is not None:
            noise_start, noise_stop = noise_window["start"], noise_window["stop"]
            end = min(i, noise_start)
            prefix_line.set_data(trail[:end+1,0], trail[:end+1,1])
            noise_line.set_data(trail[noise_start:min(i,noise_stop)+1,0], trail[noise_start:min(i,noise_stop)+1,1])
            kept.set_data(trail[noise_stop:i+1,0], trail[noise_stop:i+1,1])
            if i >= noise_stop and noise_stop < len(states)-1:
                handoff_marker.set_data([trail[noise_stop,0]], [trail[noise_stop,1]])
            else:
                handoff_marker.set_data([], [])
        if return_segment is not None:
            a = return_segment["start_step"] - return_segment["render_start_step"]
            b = return_segment["end_step"] - return_segment["render_start_step"]
            prefix_line.set_data(trail[:min(i, a)+1, 0], trail[:min(i, a)+1, 1])
            noise_line.set_data(trail[a:min(i, b)+1, 0], trail[a:min(i, b)+1, 1])
            kept.set_color("#999999")
            kept.set_data(trail[b:i+1, 0], trail[b:i+1, 1])
            if i >= a:
                return_start_marker.set_data([trail[a, 0]], [trail[a, 1]])
            else:
                return_start_marker.set_data([], [])
            if i >= b:
                return_end_marker.set_data([trail[b, 0]], [trail[b, 1]])
            else:
                return_end_marker.set_data([], [])
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
        if handoff_step is not None:
            note += "\n示范动作" if i < handoff_step else "\n策略动作（权重冻结）"
        if noise_window is not None:
            decision = min(i, len(states)-2)
            note += "\n噪声开启" if noise_window["start"] <= decision < noise_window["stop"] else "\n零噪声（权重冻结）"
        if peak is not None and i >= peak:
            note += "\ncredited 封顶之后"
        if cut is not None and i >= cut:
            note += "\n裁剪点之后"
        if return_segment is not None:
            note += f"\n原回合第 {i + return_segment['render_start_step']} 步"
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
    if return_segment is not None:
        for field, name in [("start_step", "departure"), ("end_step", "return")]:
            draw(return_segment[field] - return_segment["render_start_step"])
            fig.savefig(output/f"{name}.png", dpi=120)
    plt.close(fig)
    meta = {"episode":str(episode.resolve()),"physical_steps":len(states)-1,
            "physical_seconds":(len(states)-1)*cfg.feature_step_seconds,
            "rendered_frames":len(frames),"fps":30,
            "type":"archived_state_reconstruction_not_game_recording",
            "trail": "pot_transform_point_b" if states.shape[1] == 35 else "player_origin",
            "credited_progress_m": data["metrics"].get("credited_progress_m")}
    meta["displayed_progress_m"] = progress
    meta["displayed_progress_name"] = progress_name
    if return_segment is not None:
        meta["return_segment"] = return_segment
        meta.pop("credited_progress_m", None)
    if handoff is not None:
        meta["prefix_handoff"] = handoff
    if noise_window is not None:
        meta["noise_intervention"] = noise_window
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
        pool_metadata = json.loads(str(data["archive_metadata_json"])) if "archive_metadata_json" in data else {}
        pool_size = int(data["n_trajectories"])
    offline_layer = pool_metadata.get("selection") == "offline_saved_history_pareto_layer"
    if offline_layer:
        # A visualization must not silently compress the supplied offline layer.
        from dataclasses import replace
        cfg = replace(cfg, elite_archive_capacity=max(cfg.elite_archive_capacity, pool_size))
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
        role = pool_metadata["display_role"] if offline_layer else ("前沿" if traj.archive_role == "pareto" else "多样性")
        ax.set_title(f"{rank:02d} · 池索引 {index} · 第 {traj.iteration} 轮 · {role}\n"
                     f"D={endpoint.distance:.2f} m · t={cap} 步 / {cap*cfg.feature_step_seconds:.2f} s",
                     fontproperties=font, fontsize=9)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="World x (m)", ylabel="World y (m)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.12)
        return endpoint

    cols = (2 if len(order) <= 6 else 3) if offline_layer else 4
    rows = int(np.ceil(len(order)/cols))
    height = rows*3.6 + (1.8 if offline_layer else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(16, height), squeeze=False, facecolor="#faf9f6")
    fig.subplots_adjust(top=1-1.8/height if offline_layer else .90, bottom=.065, left=.045, right=.99, hspace=.48, wspace=.30)
    pool_title = pool_metadata["display_title"] if offline_layer else "完整精英池"
    title = fig.suptitle(f"{pool_title} · {len(order)} 条运动轨迹", fontproperties=font, fontsize=20,
                         y=1-.2/height if offline_layer else .986)
    fig.text(.5, 1-.8/height if offline_layer else .959, f"{pool.name} · 按确认距离 D 降序 · 所有小图同尺度 · D 为路线进度，不是高度",
             ha="center", fontproperties=font, fontsize=10)
    fig.legend(handles=legends, loc="upper center", bbox_to_anchor=(.5, 1-1.05/height if offline_layer else .948), ncol=5, prop=font, frameon=False)
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
                title.set_text(f"{pool_title} · {len(order)} 条运动轨迹 · {step*cfg.feature_step_seconds:.2f} / {end*cfg.feature_step_seconds:.2f} 秒")
                writer.grab_frame()
    plt.close(fig)
    metadata = {"source_pool":str(pool.resolve()), "source_sha256":hashlib.sha256(pool.read_bytes()).hexdigest(),
                "archive_metadata":pool_metadata,
                "manifest":str(manifest_path.resolve()), "type":"archived_state_reconstruction_not_game_recording",
                "trail":"pot_transform_point_b" if width==35 else "player_origin", "unpruned":True,
                "reference_sha256":route.metadata["sha256"], "shared_bounds":[low.tolist(),high.tolist()], "entries":entries}
    (output/"index.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output/"all_elite_trajectories.png", flush=True)


def render_return_segments(report_path: Path, output: Path, video_ranks=()):
    """Show selected excursions from screening reports or retained pool snapshots."""
    from matplotlib.lines import Line2D

    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = TrainConfig(**report["config"])
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    if route.metadata["sha256"] != report["reference"]["sha256"]:
        raise ValueError("Reference differs from the screening report")
    selected = report["selected"]
    if not selected:
        raise ValueError("No selected return segments to render")
    output.mkdir(parents=True, exist_ok=True)
    env, player = load_colliders(Path(cfg.game_root))
    terrain, bodies, tip = extract_polygons(env), extract_body_local(player), extract_tip_local(player)
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    offsets = np.r_[0., np.cumsum([s.length for s in route.segments])]
    source_data = {}
    with np.load(report_path.parent / report["data_archive"], allow_pickle=False) as archive:
        for row in report["candidates"]:
            index = row["episode_index"]
            if index not in source_data:
                source_data[index] = (archive[f"e{index}_states"], archive[f"e{index}_actions"])

    def source_label(row):
        source = report["episodes"][row["episode_index"]]["sources"][0]
        if source["kind"] == "full_episode_archive":
            return f"历史完整轨迹 · 采集第 {source['iteration']} 轮"
        return f"冻结测试 · {Path(source['path']).stem}"

    def on_route(row, value):
        segment = route.segments[row["segment_index"]]
        s = value - offsets[row["segment_index"]]
        k = min(len(segment.lengths)-1, max(0, int(np.searchsorted(segment.starts, s, side="right")-1)))
        t = np.clip((s-segment.starts[k])/segment.lengths[k], 0., 1.)
        return segment.points[k] + t * segment.vectors[k]

    def panel(ax, row, title=None):
        states, _ = source_data[row["episode_index"]]
        a, b = row["start_step"], row["end_step"]
        left, right = max(0, a-24), min(len(states)-1, b+24)
        xy = path_xy(states)
        projected = np.stack([on_route(row, row[k]) for k in ["s_start_m", "s_end_m"]])
        bounds = np.concatenate([xy[left:right+1], projected])
        center = (bounds.min(axis=0)+bounds.max(axis=0))/2
        span = np.maximum(bounds.max(axis=0)-bounds.min(axis=0)+2., [7., 5.])
        low, high = center-span/2, center+span/2
        for poly in terrain:
            if np.all(poly.max(axis=0) >= low) and np.all(poly.min(axis=0) <= high):
                ax.fill(poly[:, 0], poly[:, 1], color="#e5e1d9", ec="#c0b9ac", lw=.5, zorder=0)
        for segment in route.segments:
            ax.plot(segment.points[:, 0], segment.points[:, 1], color="#648d69", lw=1., zorder=1)
        ax.plot(xy[left:right+1, 0], xy[left:right+1, 1], color="#a4a4a4", lw=1., ls="--", zorder=2)
        ax.plot(xy[a:b+1, 0], xy[a:b+1, 1], color="#cc6b27", lw=1.8, zorder=3)
        ax.plot(*xy[a], "o", color="#246b9b", ms=6, mec="white", mew=.6, zorder=5)
        ax.plot(*xy[b], "s", color="#824da1", ms=6, mec="white", mew=.6, zorder=5)
        for p, q, color in zip([xy[a], xy[b]], projected, ["#246b9b", "#824da1"]):
            ax.plot([p[0], q[0]], [p[1], q[1]], ":", color=color, lw=.8)
            ax.plot(*q, ".", color=color, ms=4)
        cell = row["decision"].get("cell")
        prior = f" · 原 {row['previous_display_rank']:02d}" if "previous_display_rank" in row else ""
        head = title or f"{row['display_rank']:02d}{prior} · 分桶 ({cell[1]}, {cell[2]})"
        boundary_note = ""
        if "return_projection_speed_mps" in row:
            boundary_note = f"返回投影速度 {row['return_projection_speed_mps']:+.3f} m/s\n"
        duration_note = f" ({row['T_seconds']:.3f} s)" if "T_seconds" in row else ""
        ax.set_title(f"{head}\ns_start={row['s_start_m']:.3f} m · Δs={row['delta_s_m']:.3f} m · T={row['T_steps']} 步{duration_note}\n"
                     f"{boundary_note}{source_label(row)} · {a} → {b} 步", fontproperties=font, fontsize=9)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="World x (m)", ylabel="World y (m)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.12)

    legends = [Line2D([], [], color="#cc6b27", label="离开—返回片段"),
               Line2D([], [], color="#a4a4a4", ls="--", label="前后各 24 步上下文"),
               Line2D([], [], color="#648d69", label="参考路线"),
               Line2D([], [], marker="o", color="#246b9b", ls="", label="离开前最后内点"),
               Line2D([], [], marker="s", color="#824da1", ls="", label="首次返回内点")]
    cols = 2 if len(selected) <= 6 else 3
    rows = int(np.ceil(len(selected)/cols))
    height = rows*4.2+2.3
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, height), squeeze=False, facecolor="#faf9f6")
    fig.subplots_adjust(top=1-2.3/height, bottom=.7/height, left=.055, right=.985, hspace=.7, wspace=.30)
    rule = report["rule"]
    fig.suptitle(report.get("display_title", f"离开—返回片段 · {len(selected)} 个入选代表"),
                 fontproperties=font, fontsize=20, y=1-.08/height)
    fig.text(.5, 1-.48/height, f"Δs ≥ {rule['min_forward_m']:g} m · s_start 分桶 {rule['start_bin_m']:g} m · Δs 分桶 {rule['delta_bin_m']:g} m · 同桶取最短 T",
             ha="center", fontproperties=font, fontsize=11)
    filters = []
    if rule.get("reject_upper_boundary"):
        filters.append("排除从上方离开或返回（相对投影点的世界 y）")
    if rule.get("reject_negative_return_speed"):
        filters.append("返回投影速度 ≥ 0")
    if filters:
        fig.text(.5, 1-.77/height, " · ".join(filters), ha="center", fontproperties=font, fontsize=10)
    fig.legend(handles=legends, loc="upper center", bbox_to_anchor=(.5,1-.94/height),
               ncol=3 if cols==2 else 5, prop=font, frameon=False)
    fig.text(.5, .014, report.get("display_note", "各图局部放大，坐标单位为米；s_start 为参考路径绝对弧长，Δs 为净正向跨度；1 步 = 1/120 秒。仅离线筛选，未接入训练。"),
             ha="center", fontproperties=font, fontsize=9)
    entries = []
    for row, ax in zip(selected, axes.flat):
        panel(ax, row)
        rank = row["display_rank"]
        single, single_ax = plt.subplots(figsize=(9, 7), facecolor="#faf9f6")
        single.subplots_adjust(top=.84, bottom=.16)
        panel(single_ax, row)
        single.legend(handles=legends, loc="lower center", ncol=3, prop=font, frameon=False)
        single.savefig(output / f"segment_{rank:02d}.png", dpi=150)
        plt.close(single)
        states, actions = source_data[row["episode_index"]]
        a, b = row["start_step"], row["end_step"]
        left, right = max(0, a-24), min(len(actions), b+24)
        detail = dict(row, render_start_step=left, source_label=source_label(row))
        episode_dir = output / f"segment_{rank:02d}"
        episode_dir.mkdir(exist_ok=True)
        (episode_dir / "manifest.json").write_text(json.dumps({"config":report["config"], "reference":report["reference"]}, ensure_ascii=False), encoding="utf-8")
        episode_path = episode_dir / "episode.json"
        episode_path.write_text(json.dumps({"metrics":{"return_segment":detail},
            "states":states[left:right+1].tolist(),
            "trace":[{"step":t+1, "action":actions[t].tolist()} for t in range(left, right)]}, ensure_ascii=False), encoding="utf-8")
        if rank in video_ranks:
            render_episode(episode_path, episode_dir, terrain, bodies, tip, route)
        entries.append(dict(rank=rank, candidate_id=row["candidate_id"], image=f"segment_{rank:02d}.png",
                            video=f"segment_{rank:02d}/best-motion.mp4" if rank in video_ranks else None))
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    fig.savefig(output / "all_selected_segments.png", dpi=160)
    plt.close(fig)
    by_id = {r["candidate_id"]:r for r in report["candidates"]}
    replacements = [(by_id[r["decision"]["winner_id"]], r) for r in report["candidates"]
                    if r["decision"]["reason"] == "same_cell_replaced"
                    and r["content_sha256"] != by_id[r["decision"]["winner_id"]]["content_sha256"]]
    if replacements:
        compare, axes = plt.subplots(len(replacements), 2, figsize=(12, 4.5*len(replacements)+.6), squeeze=False)
        compare.subplots_adjust(top=.84, bottom=.14, hspace=.6, wspace=.28)
        compare.suptitle("同桶替换实例 · 只按三个标识比较", fontproperties=font, fontsize=17)
        for (winner, removed), pair in zip(replacements, axes):
            panel(pair[0], winner, "保留 · T 更短")
            panel(pair[1], removed, "淘汰 · 同桶内 T 更长")
        compare.legend(handles=legends, loc="lower center", ncol=3, prop=font, frameon=False)
        compare.savefig(output / "replacement_examples.png", dpi=150)
        plt.close(compare)
    (output / "index.json").write_text(json.dumps({"report":str(report_path.resolve()), "entries":entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output / "all_selected_segments.png", flush=True)


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
    parser.add_argument("--return-segments", type=Path, help="绘制离线筛选报告中全部入选离开—返回片段")
    parser.add_argument("--return-video", nargs="*", type=int, default=[], help="额外渲染指定展示序号的片段视频")
    parser.add_argument("--sampling-report", type=Path, help="将 sampling.json 的采样概率标在完整精英空间轨迹上")
    parser.add_argument("--sampling-focus", nargs=2, type=int, metavar=("POOL_INDEX", "ACTION"),
                        help="在并排图上额外标注指定池索引和动作所在的采样片段")
    args = parser.parse_args()
    if args.return_segments:
        if args.episode or args.ranking or args.elite_pool or args.sampling_report:
            parser.error("--return-segments 不能与其他输入模式同用")
        render_return_segments(args.return_segments, args.output, args.return_video)
        return
    if args.return_video:
        parser.error("--return-video 需要 --return-segments")
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
