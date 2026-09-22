"""Render one archived physical trajectory, without launching or simulating the game."""
import argparse
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from training.config import TrainConfig
from training.dataset import Trajectory, prune_trajectory
from training.deploy_sampling import extract_body_local, extract_tip_local, reconstruct_polygon_world, player_hub_angle_deg, rod_angle_deg
from training.reference_corridor import ReferenceCorridor, replay_corridor_fields
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons


def corridor_prune_index(states: np.ndarray, cfg: TrainConfig, route: ReferenceCorridor) -> int:
    """走廊口径剪枝后保留的最后一个状态索引（与 prune_trajectory 同一函数）。"""
    progress, guidance = replay_corridor_fields(route, states)
    traj = Trajectory(
        raw_states=states,
        actions=np.zeros((len(states) - 1, 2), np.float32),
        corridor_progress=progress,
        guidance_index=guidance,
    )
    return len(prune_trajectory(traj, cfg).raw_states) - 1


def render_episode(episode: Path, output: Path, terrain, bodies, tip, route: ReferenceCorridor,
                   rank: int | None = None, mark_prune: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(episode.read_text(encoding="utf-8"))
    manifest = json.loads((episode.parent/"manifest.json").read_text(encoding="utf-8"))
    cfg = TrainConfig(**manifest["config"])
    states = np.asarray(data["states"], np.float32)
    if states.shape != (len(data["trace"])+1, 33) or not np.isfinite(states).all():
        raise ValueError(f"Invalid raw physical-state trajectory: {episode}")
    cut = corridor_prune_index(states, cfg, route) if mark_prune else None
    bounds = np.concatenate([states[:, [0, 1]], states[:, [23, 24]]])
    low, high = bounds.min(axis=0)-1.6, bounds.max(axis=0)+1.6
    fig, ax = plt.subplots(figsize=(9.6, 6), facecolor="#faf9f6")
    ax.set_facecolor("#faf9f6")
    fig.subplots_adjust(left=.035, right=.985, bottom=.075, top=.87)
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    title = "原始状态回放" if rank is None else f"第 {rank} 名 · 原始状态回放"
    fig.suptitle(title, fontproperties=font, fontsize=17, y=.97)
    run_tokens=episode.parents[2].name.split("_")
    label=f"累计有效进度 {data['metrics']['credited_progress_m']:.2f} 米 · {run_tokens[-2]} / seed {run_tokens[-1][1:]}"
    if cut is not None:
        label += f" · 裁剪于 {cut * cfg.feature_step_seconds:.2f}s / 第 {cut} 步"
    fig.text(.5, .91, label, ha="center", fontproperties=font, fontsize=10)
    caption = "根据保存状态重建的运动示意，非游戏录像；绿色线为参考路线。"
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
    marker,=ax.plot([],[],"o",color="#c0392b",markersize=7,markeredgecolor="#fff7f0",markeredgewidth=1.2,zorder=6)
    shaft,=ax.plot([],[],color="#946640",linewidth=3,zorder=3)
    hand,=ax.plot([],[],"o",color="#d9ae83",markersize=4,zorder=5)
    timer=ax.text(.02,.97,"",transform=ax.transAxes,va="top",fontsize=12,color="#24343e",fontproperties=font)
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]))
    ax.set_aspect("equal"); ax.set_axis_off()
    def draw(i):
        row=states[i]
        split = len(states)-1 if cut is None else cut
        kept.set_data(states[:min(i, split)+1,0], states[:min(i, split)+1,1])
        if cut is not None and i >= cut:
            marker.set_data([states[cut,0]], [states[cut,1]])
            if i > cut:
                dropped.set_data(states[cut:i+1,0], states[cut:i+1,1])
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
    if cut is not None:
        draw(len(states)-1)
        fig.savefig(output/"prune.png",dpi=120)
    plt.close(fig)
    meta = {"episode":str(episode.resolve()),"physical_steps":len(states)-1,
            "physical_seconds":(len(states)-1)*cfg.feature_step_seconds,
            "rendered_frames":len(frames),"fps":30,
            "type":"archived_state_reconstruction_not_game_recording",
            "credited_progress_m": data["metrics"]["credited_progress_m"]}
    if cut is not None:
        meta["prune_state_index"] = int(cut)
        meta["prune_seconds"] = cut * cfg.feature_step_seconds
    (output/"source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(output/"best-motion.mp4", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, help="评估排名 JSON；与 --top 一起渲染前若干条")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--mark-prune", action="store_true", help="按走廊 credited 封顶口径标出裁剪点")
    args = parser.parse_args()
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
                                       mark_prune=args.mark_prune))
    if args.ranking is not None:
        (args.output/"index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__=="__main__": main()
