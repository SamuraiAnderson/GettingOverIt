"""Plot the elite-pool (t, D) upper-left envelope from saved SIL snapshots.

Does not launch the game. Points come from `run_logging.load_elites`; the front
is the same `sil_archive.pareto_front` used in training.

    conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.present.plot_sil_td_envelope --elites-dir runs/corridor_sil_td_pareto_20260927_S_s47/elites --output runs/corridor_sil_td_pareto_20260927/td_envelope
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.dataset import Trajectory
from training.run_logging import load_elites
from training.sil_archive import dedup_td, dominates, pareto_front, td_point

FONT = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
STEP_SECONDS = TrainConfig.feature_step_seconds


def discover_pools(elites_dir: Path) -> list[Path]:
    return sorted(elites_dir.glob("elites_iter_*.npz"))


def load_trajectories(path: Path) -> list[Trajectory]:
    return [Trajectory(**row) for row in load_elites(path)]


def catalog_pool(trajs: list[Trajectory], *, source: str | None = None) -> dict:
    comparable = [traj for traj in trajs if td_point(traj) is not None]
    front = pareto_front(comparable)
    front_ids = {id(traj) for traj in front}
    records = []
    for index, traj in enumerate(trajs):
        point = td_point(traj)
        record = {
            "index": index,
            "score": float(traj.score),
            "iteration": int(traj.iteration),
            "archive_role": traj.archive_role,
            "comparable": point is not None,
        }
        if point is None:
            record.update(t=None, D=None, seconds=None, on_front=False)
        else:
            steps, distance = point
            record.update(
                t=int(steps),
                D=float(distance),
                seconds=float(steps) * STEP_SECONDS,
                on_front=id(traj) in front_ids,
            )
        records.append(record)
    envelope = sorted(
        (row for row in records if row["on_front"]),
        key=lambda row: (row["t"], -row["D"]),
    )
    return {
        "source": source,
        "n_trajectories": len(trajs),
        "n_comparable": len(comparable),
        "n_front": len(envelope),
        "points": records,
        "envelope": envelope,
    }


def peel_layers(trajs: list[Trajectory]) -> list[list[Trajectory]]:
    """Exact successive fronts after the training endpoint dedup; no bins/cap."""
    remaining = dedup_td(trajs)
    layers = []
    while remaining:
        front = pareto_front(remaining)
        layers.append(sorted(front, key=lambda t: (-td_point(t)[1], td_point(t)[0])))
        removed = {id(t) for t in front}
        remaining = [t for t in remaining if id(t) not in removed]
    return layers


def episode_identity(traj):
    return hashlib.sha256(np.asarray(traj.raw_states, np.float32).tobytes()
                          + np.asarray(traj.actions, np.float32).tobytes()).hexdigest()


def plot_layers(report, output):
    """Use this viewer for both endpoint fronts and the second-front inventory."""
    colors = ["#bf493c", "#236f9e", "#c79036"]
    fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor="#faf9f6")
    ax.set_facecolor("#faf9f6")
    rest = [r for r in report["points"] if r["layer"] > 3]
    _scatter(ax, rest, s=22, color="#b6b6b6", label=f"第四层及以后（{len(rest)}）")
    for layer, color in enumerate(colors, 1):
        rows = sorted((r for r in report["points"] if r["layer"] == layer), key=lambda r:r["t"])
        _scatter(ax, rows, front=True, color=color, marker="o", markersize=4, linewidth=1.2,
                 label=f"第 {layer} 层（{len(rows)}）")
        if layer == 2:
            for row in rows:
                ax.annotate(str(row["rank"]), (row["t"], row["D"]), xytext=(5, 5),
                            textcoords="offset points", fontsize=9, color=color)
    ax.set(xlabel="", ylabel="")
    ax.set_xlabel("封顶时间 t（环境步；120 步 = 1 秒）", fontproperties=FONT)
    ax.set_ylabel("最远确认路线进度 D（米）", fontproperties=FONT)
    ax.set_title("已保存历史候选：逐层剥离的 (t, D) 包络", fontproperties=FONT, fontsize=15)
    ax.grid(alpha=.2)
    ax.legend(prop=FONT, loc="lower right")
    fig.text(.5, .055, f"{report['n_snapshots']} 份完整池快照 → {report['n_unique_episodes']} 条去重轨迹；"
             "蓝色编号对应第二层轨迹图。", ha="center", fontproperties=FONT, fontsize=10)
    fig.text(.5, .025, "未入池且未保存的轨迹不在此统计；连线仅连接封顶点，不是某条轨迹的 D(t)。",
             ha="center", fontproperties=FONT, fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(output/"layers.png", dpi=150)
    plt.close(fig)

    rows = report["second_layer"]
    if not rows:
        return
    headers = ["编号", "采集轮次", "D（米）", "t（步）", "t（秒）", "仍在最新池", "第一层支配者：轮次 / D / t"]
    cells = []
    for r in rows:
        dominators = r["first_layer_dominators"]
        nearest = min(dominators, key=lambda d:(d["D"]-r["D"], r["t"]-d["t"]))
        cells.append([r["rank"], r["iteration"], f"{r['D']:.3f}", r["t"], f"{r['seconds']:.3f}",
                      "是" if r["in_latest_pool"] else "否",
                      f"{nearest['iteration']} / {nearest['D']:.3f} / {nearest['t']}"])
    fig, ax = plt.subplots(figsize=(13, 2.2 + .43*len(rows)), facecolor="#faf9f6")
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[.06, .09, .10, .08, .09, .12, .42])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_height(1/(len(rows)+2))
        cell.set_edgecolor("#d4d6d7")
        cell.set_facecolor("#ddeaf1" if r == 0 else ("#ffffff" if r%2 else "#f3f4f1"))
        cell.get_text().set_fontproperties(FONT)
        cell.get_text().set_fontsize(10)
    ax.set_title(f"第二层包络：{len(rows)} 条完整轨迹", fontproperties=FONT, fontsize=16, pad=18)
    fig.text(.5, .025, "每行只展示一位第一层支配者（D 最接近）；完整来源、所有支配者见 layers.json。",
             ha="center", fontproperties=FONT, fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .98))
    fig.savefig(output/"second_layer_inventory.png", dpi=150)
    plt.close(fig)


def inspect_archive_layers(runs, output):
    """Read only full archived episodes; never imply coverage of rejected rollouts."""
    from training.elite_archive import EliteTrajectoryArchive
    from training.reference_corridor import ReferenceCorridor
    from training.reference_fingerprint import (assert_fingerprint_compatible, load_fingerprint_sidecar,
                                               reference_fingerprint)
    from training.reward import is_water_trajectory
    from training.run_logging import save_elites

    base = json.loads((runs[-1]/"run_meta.json").read_text(encoding="utf-8-sig"))
    cfg = TrainConfig(**base["config"])
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    expected = base["reference_fingerprint"]
    actual = reference_fingerprint(cfg, route_meta=route.metadata,
                                   schema_id=expected["state_schema"], raw_dim=expected["raw_dim"])
    assert_fingerprint_compatible(expected, actual, context="current reference")
    episodes, sources, inputs, latest_trajs = {}, {}, [], []
    for run in runs:
        meta = json.loads((run/"run_meta.json").read_text(encoding="utf-8-sig"))
        assert_fingerprint_compatible(expected, meta["reference_fingerprint"], context=str(run))
        if meta["config"]["feature_step_seconds"] != cfg.feature_step_seconds:
            raise ValueError(f"Different time base: {run}")
        pools = discover_pools(run/"elites/full_episodes")
        if not pools:
            raise ValueError(f"No complete elite snapshots: {run}")
        for pool in pools:
            assert_fingerprint_compatible(expected, load_fingerprint_sidecar(pool), context=str(pool))
            with np.load(pool, allow_pickle=False) as data:
                metadata = json.loads(str(data["archive_metadata_json"]))
            if (metadata.get("schema") != "full_episode_elites_v1"
                    or metadata.get("origin") != "episode_reset" or metadata.get("truncation") != "none"):
                raise ValueError(f"Not a full episode archive: {pool}")
            latest_trajs = load_trajectories(pool)
            for index, traj in enumerate(latest_trajs):
                if (not EliteTrajectoryArchive._complete(traj) or td_point(traj) is None
                        or is_water_trajectory(traj.raw_states, cfg)):
                    raise ValueError(f"Invalid archived candidate: {pool} / {index}")
                key = episode_identity(traj)
                if key in episodes:
                    previous = episodes[key]
                    if (td_point(previous) != td_point(traj)
                            or not np.array_equal(previous.corridor_confirmed_progress, traj.corridor_confirmed_progress)
                            or not np.array_equal(previous.corridor_credit_events, traj.corridor_credit_events)):
                        raise ValueError(f"Identical episode has inconsistent scoring: {pool} / {index}")
                episodes[key] = traj
                sources.setdefault(key, []).append(dict(path=str(pool.resolve()), pool_index=index,
                                                         iteration=int(traj.iteration), role=traj.archive_role))
            inputs.append(dict(path=str(pool.resolve()), sha256=hashlib.sha256(pool.read_bytes()).hexdigest()))
        print(f"Read {run.name}: {len(pools)} snapshots, {len(episodes)} unique episodes", flush=True)
    layers = peel_layers(list(episodes.values()))
    latest_ids = {episode_identity(t): i for i, t in enumerate(latest_trajs)}
    points = []
    for level, layer in enumerate(layers, 1):
        for rank, traj in enumerate(layer, 1):
            key = episode_identity(traj)
            t, d = td_point(traj)
            dominators = [dict(iteration=int(other.iteration), t=td_point(other)[0], D=td_point(other)[1],
                               identity=episode_identity(other)) for other in layers[0]
                          if dominates(td_point(other), (t, d))]
            points.append(dict(layer=level, rank=rank, identity=key, iteration=int(traj.iteration),
                               t=t, D=d, seconds=t*cfg.feature_step_seconds, steps=len(traj.actions),
                               in_latest_pool=key in latest_ids, latest_pool_index=latest_ids.get(key),
                               first_layer_dominators=dominators, sources=sources[key]))
    report = dict(kind="offline_saved_history_pareto_layers", scope="full episodes that appeared in saved elite snapshots",
                  limitation="Never-saved rejected rollouts are unavailable; not the full training candidate population",
                  rule="training dedup_td, then repeatedly remove exact pareto_front; no bucket or capacity pruning",
                  step_seconds=cfg.feature_step_seconds, reference=route.metadata,
                  n_snapshots=len(inputs), n_episode_references=sum(map(len, sources.values())),
                  n_unique_episodes=len(episodes), n_unique_td_points=len(points),
                  layer_counts=[len(layer) for layer in layers], latest_pool=inputs[-1],
                  latest_pool_layer_counts=[len(layer) for layer in peel_layers(latest_trajs)],
                  second_layer=[r for r in points if r["layer"] == 2], points=points, inputs=inputs,
                  source_files=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                for p in [Path(__file__), ROOT/"src/training/sil_archive.py"]])
    output.mkdir(parents=True, exist_ok=True)
    (output/"layers.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(dict(config=base["config"], reference=route.metadata),
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    if len(layers) > 1:
        save_elites(output/"second_layer.npz", layers[1], expected, archive_metadata={
            "schema":"full_episode_elites_v1", "origin":"episode_reset", "truncation":"none",
            "selection":"offline_saved_history_pareto_layer", "pareto_layer":2,
            "display_title":"历史已保存候选 · 第二层包络", "display_role":"第二层",
            "source_report":str((output/"layers.json").resolve())})
    plot_layers(report, output)
    print(json.dumps({k:report[k] for k in ["n_snapshots", "n_unique_episodes", "n_unique_td_points",
                                           "layer_counts", "latest_pool_layer_counts"]}), flush=True)
    return report


def _scatter(ax, rows, *, front=False, **kwargs):
    xs = [row["t"] for row in rows]
    ys = [row["D"] for row in rows]
    if not xs:
        return
    ax.plot(xs, ys, **kwargs) if front else ax.scatter(xs, ys, **kwargs)


def plot_snapshot(catalog: dict, output: Path) -> Path:
    front = catalog["envelope"]
    rest = [row for row in catalog["points"] if row["comparable"] and not row["on_front"]]
    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor="#faf9f6")
    ax.set_facecolor("#faf9f6")
    _scatter(ax, rest, s=36, facecolors="none", edgecolors="#7a868c", linewidths=1.1, zorder=2,
             label=f"被支配 / 多样性补充（{len(rest)}）")
    if front:
        ax.plot([row["t"] for row in front], [row["D"] for row in front],
                color="#c0392b", linewidth=1.6, zorder=3, label="左上包络")
        ax.scatter([row["t"] for row in front], [row["D"] for row in front],
                   s=46, color="#c0392b", zorder=4, label=f"非支配前沿（{len(front)}）")
        fastest, farthest = front[0], max(front, key=lambda row: row["D"])
        ax.annotate(f"最快  t={fastest['t']}\nD={fastest['D']:.2f} m",
                    (fastest["t"], fastest["D"]), textcoords="offset points",
                    xytext=(8, -18), fontproperties=FONT, fontsize=9, color="#c0392b")
        if farthest is not fastest:
            ax.annotate(f"最远  t={farthest['t']}\nD={farthest['D']:.2f} m",
                        (farthest["t"], farthest["D"]), textcoords="offset points",
                        xytext=(8, 8), fontproperties=FONT, fontsize=9, color="#c0392b")
    ax.set_xlabel("封顶步数 t（越小越快）", fontproperties=FONT, fontsize=11)
    ax.set_ylabel("累计有效进度 D（米，越大越远）", fontproperties=FONT, fontsize=11)
    title = "精英池 (t, D) 左上包络"
    if catalog.get("source"):
        title += f"\n{Path(catalog['source']).name}"
    ax.set_title(title, fontproperties=FONT, fontsize=14)
    ax.grid(True, color="#d8d2c8", linewidth=0.6)
    legend = ax.legend(prop=FONT, loc="lower right")
    legend.get_frame().set_facecolor("#faf9f6")
    fig.text(0.5, 0.02, "左上更好：更快且更远。包络由 sil_archive.pareto_front 重算，不读评估分数。",
             ha="center", fontproperties=FONT, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output / "envelope.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_history(catalogs: list[dict], output: Path) -> Path | None:
    drawn = [item for item in catalogs if item["envelope"]]
    if len(drawn) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor="#faf9f6")
    ax.set_facecolor("#faf9f6")
    cmap = plt.cm.viridis
    for i, item in enumerate(drawn):
        color = cmap(i / max(len(drawn) - 1, 1))
        xs = [row["t"] for row in item["envelope"]]
        ys = [row["D"] for row in item["envelope"]]
        last = i == len(drawn) - 1
        ax.plot(xs, ys, color=color, linewidth=2.0 if last else 1.0,
                alpha=1.0 if last else 0.45, zorder=3 if last else 2)
        ax.scatter(xs, ys, color=color, s=28 if last else 16,
                   zorder=4 if last else 2)
    ax.set_xlabel("封顶步数 t（越小越快）", fontproperties=FONT, fontsize=11)
    ax.set_ylabel("累计有效进度 D（米，越大越远）", fontproperties=FONT, fontsize=11)
    ax.set_title(
        f"左上包络随迭代外推（{Path(drawn[0]['source']).stem} → {Path(drawn[-1]['source']).stem}）",
        fontproperties=FONT, fontsize=14)
    ax.grid(True, color="#d8d2c8", linewidth=0.6)
    fig.tight_layout()
    path = output / "envelope_history.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_report(catalogs: list[dict], output: Path) -> Path:
    latest = catalogs[-1]
    payload = {
        "step_seconds": STEP_SECONDS,
        "latest": latest,
        "history": [{"source": item["source"], "n_front": item["n_front"],
                     "n_comparable": item["n_comparable"],
                     "envelope": item["envelope"]} for item in catalogs],
    }
    path = output / "envelope.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def trajectory_comparison(trajs: list[Trajectory], index: int) -> dict:
    """Compare stored confirmed D_t, not the straight lines between terminal points."""
    if not 0 <= index < len(trajs):
        raise ValueError("Trajectory index is outside the pool")
    curves = []
    for traj in trajs:
        curve = np.asarray(traj.corridor_confirmed_progress, dtype=np.float64)
        if (traj.progress_mode != "confirmed_endpoint_v2" or curve.ndim != 1
                or len(curve) != len(traj.actions) + 1 or not np.isfinite(curve).all()
                or np.any(np.diff(curve) < -1e-8)):
            raise ValueError("Comparison requires full, monotone confirmed D_t histories")
        curves.append(curve)
    selected = curves[index]
    horizon = len(selected)
    # A shorter episode contributes its achieved progress at later time budgets.
    aligned = np.stack([np.pad(c, (0, max(0, horizon - len(c))), mode="edge")[:horizon]
                        for c in curves])
    envelope = aligned.max(axis=0)
    others = np.delete(aligned, index, axis=0)
    other_envelope = others.max(axis=0) if len(others) else np.zeros(horizon)
    leading = (selected > 0) & (selected >= envelope - 1e-8)
    starts = np.flatnonzero(leading & ~np.r_[False, leading[:-1]])
    ends = np.flatnonzero(leading & ~np.r_[leading[1:], False])
    point = td_point(trajs[index])
    if point is None:
        raise ValueError("Selected trajectory has no valid confirmed endpoint")
    return {"trajectory_index": index, "source_iteration": int(trajs[index].iteration),
            "cap_t": int(point[0]), "cap_D": float(point[1]),
            "t": list(range(horizon)), "D_t": selected.tolist(),
            "prefix_envelope": envelope.tolist(), "other_prefix_envelope": other_envelope.tolist(),
            "on_prefix_envelope_intervals": [[int(a), int(b)] for a, b in zip(starts, ends)]}


def plot_trajectory_comparison(trajs: list[Trajectory], index: int, catalog: dict,
                              output: Path) -> Path:
    data = trajectory_comparison(trajs, index)
    t, distance = np.asarray(data["t"]), np.asarray(data["D_t"])
    front = catalog["envelope"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), facecolor="#faf9f6",
                             gridspec_kw={"width_ratios": [1.5, 1]})
    zoom = min(t[-1], max(data["cap_t"], max(row["t"] for row in front)) * 1.08)
    for ax, limit, title in zip(axes, [zoom, t[-1]], ["确认进度增长阶段", "完整轨迹（含封顶后）"]):
        ax.set_facecolor("#faf9f6")
        ax.plot(t, data["prefix_envelope"], color="#7a868c", linewidth=3.5,
                label="池内逐帧上包络 max D_i(t)")
        ax.plot([r["t"] for r in front], [r["D"] for r in front], "o--",
                color="#c0392b", linewidth=1.1, markersize=4,
                label="封顶点前沿（虚线仅连接点）")
        ax.plot(t, distance, color="#32789d", linewidth=1.8,
                label=f"所选轨迹 D(t)（第 {data['source_iteration']} 轮）")
        ax.scatter([data["cap_t"]], [data["cap_D"]], s=90, marker="*", color="#32789d", zorder=5)
        ax.set_xlim(0, max(1, limit))
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontproperties=FONT, fontsize=12)
        ax.set_xlabel("时间 t（环境步；120 步 = 1 秒）", fontproperties=FONT)
        ax.set_ylabel("累计确认进度 D(t)（米）", fontproperties=FONT)
        ax.grid(True, color="#d8d2c8", linewidth=.6)
    axes[0].legend(prop=FONT, loc="upper left", fontsize=9)
    fig.suptitle(f"轨迹 (D(t), t) 与精英池包络 · 第 {data['source_iteration']} 轮 · "
                 f"D={data['cap_D']:.2f} m，t={data['cap_t']}", fontproperties=FONT, fontsize=14)
    fig.text(.5, .025, f"来源：{Path(catalog['source']).name}。D(t) 是单调的确认进度记录，回落不会令其下降；"
             "灰线由所有已保存轨迹逐帧取最大值。", ha="center", fontproperties=FONT, fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    path = output / "trajectory_vs_envelope.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    data["source_pool"] = catalog["source"]
    (output / "trajectory_vs_envelope.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


SAMPLING_NAMES = {"trajectory_uniform": "原采样", "phase_mixture_v1": "旧版：多数阶段混合",
                  "phase_event_mixture_v2": "v2：软阶段 + 起跳事件", "quality_dynamics_v3": "v3：轨迹价值 + 接触动力学",
                  "phase_balanced_v1": "纯阶段均衡"}
PHASE_NAMES = {
    "approach_support": "前段支撑 / 其他", "takeoff_preparation": "起跳准备（不限区域）",
    "approach_air_ascent": "前段腾空上升", "approach_air_other": "前段腾空其他",
    "tree_support": "树区支撑 / 其他", "tree_air_ascent": "树区腾空上升",
    "tree_air_other": "树区腾空其他", "elevated_tip_contact_candidate": "高处锤头接触候选",
}


def sampling_catalog(trajs, cfg, *, draws=20000, seed=7):
    """Exact marginals + draws from the SAME sampler as training; no network update."""
    from training.gru_sil import GRUSILSamplingPlan, eligible_sil_trajectories, DYNAMICS_SAMPLING_MODE
    from training.climb_diagnostics import PhaseRules, classify_motion, phase_counts, launch_event_windows
    from training.reference_corridor import trajectory_endpoint
    if type(draws) is not int or draws < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Sampling preview needs positive draws and nonnegative seed")
    candidates = eligible_sil_trajectories(trajs)
    rules = PhaseRules(**(cfg.gru_sil_phase_rules or {}))
    labels = {i: classify_motion(trajs[i].raw_states, trajs[i].contact, rules)[0] for i in candidates}
    events = {i: launch_event_windows(trajs[i].raw_states, trajs[i].contact,
                                     trajectory_endpoint(trajs[i]).cap + 1, rules) for i in candidates}
    event_masks = {i: np.zeros(len(trajs[i].actions), bool) for i in candidates}
    for i in candidates:
        for event in events[i]: event_masks[i][event["start"]:event["end"]] = True
    dynamics_plan = GRUSILSamplingPlan(trajs, candidates, cfg.recurrent_seq_len, DYNAMICS_SAMPLING_MODE,
                                       rules, step_seconds=cfg.feature_step_seconds)
    dynamics = dynamics_plan.dynamics
    chunks_phases = {}
    data = {"kind": "offline_sampling_preview_before_advantage_and_KL", "draws_per_mode": draws,
            "seed": seed, "sampling_iteration": 0, "seq_len": cfg.recurrent_seq_len,
            "step_seconds": cfg.feature_step_seconds, "phase_rules": vars(rules),
            "comparison_modes": ["phase_event_mixture_v2", DYNAMICS_SAMPLING_MODE],
            "dynamics": {i: {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in d.items()}
                         for i,d in dynamics.items()},
            "launch_events": events,
            "trajectories": [{"index": i, "D_m": trajectory_endpoint(trajs[i]).distance,
                              "cap": trajectory_endpoint(trajs[i]).cap + 1, "steps": len(trajs[i].actions),
                              "source_iteration": int(trajs[i].iteration)} for i in candidates], "modes": {}}
    for mode in SAMPLING_NAMES:
        plan = (dynamics_plan if mode == DYNAMICS_SAMPLING_MODE else
                GRUSILSamplingPlan(trajs, candidates, cfg.recurrent_seq_len, mode, rules))
        probabilities = plan.probabilities()
        np.testing.assert_allclose(sum(probabilities.values()), 1., rtol=0, atol=1e-12)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 714]))
        observed, branches = Counter(), Counter()
        for _ in range(draws):
            ti, a, b, cap, phase, branch = plan.draw(rng)
            observed[(ti, a, b, cap)] += 1
            branches[branch] += 1
        expected_frames, sampled_frames = Counter(), Counter()
        expected_event_frames = 0.
        dynamics_frames, trajectory_mass = Counter(), Counter()
        records = []
        for chunk, probability in probabilities.items():
            ti, a, b, cap = chunk
            if chunk not in chunks_phases:
                chunks_phases[chunk] = phase_counts(labels[ti][a:b])
            phases = chunks_phases[chunk]
            expected_frames.update({name: n * probability for name, n in phases.items()})
            sampled_frames.update({name: n * observed[chunk] for name, n in phases.items()})
            expected_event_frames += probability * int(event_masks[ti][a:b].sum())
            d = dynamics[ti]
            masks = {"core": d["core"], "context": (d["priority"]>0)&~d["core"],
                     "free_flight": d["free_flight"], "all_air": ~d["contact"]}
            dynamics_frames.update({k: probability*int(mask[a:b].sum()) for k,mask in masks.items()})
            trajectory_mass[ti] += probability
            records.append({"trajectory": ti, "start": a, "end": b, "cap": cap,
                            "probability": probability, "observed_draws": observed[chunk], "phase_counts": phases,
                            "weighting": plan.chunk_details.get(chunk)})
        total_expected, total_sampled = sum(expected_frames.values()), sum(sampled_frames.values())
        tree = lambda key: key.startswith("tree_") or key == "elevated_tip_contact_candidate"
        data["modes"][mode] = {"chunks": records, "branch_draws": dict(branches),
            "weighting": getattr(plan, "distribution_summary", None),
            "trajectory_probabilities": dict(trajectory_mass),
            "expected_dynamics_frame_share": {k: v/total_expected for k,v in dynamics_frames.items()},
            "expected_launch_event_frame_share": expected_event_frames / total_expected,
            "expected_phase_frame_share": {k: v / total_expected for k, v in expected_frames.items()},
            "sampled_phase_frame_share": {k: v / total_sampled for k, v in sampled_frames.items()},
            "expected_tree_frame_share": sum(v for k, v in expected_frames.items() if tree(k)) / total_expected,
            "sampled_tree_frame_share": sum(v for k, v in sampled_frames.items() if tree(k)) / total_sampled,
            "max_chunk_probability": max(probabilities.values()),
            "max_chunk_frequency_error": max(abs(observed[c] / draws - p) for c, p in probabilities.items()),
            "phase_groups": {k: {"chunks": len(v), "trajectories": len({c[0] for c in v})}
                             for k, v in plan.groups.items()}}
    return data


def plot_sampling(data, output):
    """Probability to include each action in one SIL chunk draw, before filtering."""
    order = sorted(data["trajectories"], key=lambda row: (-row["D_m"], row["cap"], row["index"]))
    positions = {row["index"]: i for i, row in enumerate(order)}
    horizon = max(row["cap"] for row in order)
    modes = data.get("comparison_modes", ("trajectory_uniform", "phase_mixture_v1"))
    matrices = []
    for mode in modes:
        matrix = np.full((len(order), horizon), np.nan)
        for row in data["modes"][mode]["chunks"]:
            matrix[positions[row["trajectory"]], row["start"]:row["end"]] = row["probability"] * 100
        matrices.append(matrix)
    vmax = max(float(np.nanmax(m)) for m in matrices)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 11), layout="constrained", facecolor="#faf9f6")
    cmap = plt.get_cmap("YlGnBu").copy(); cmap.set_bad("#e9e9e9")
    tick_labels = [f"{rank:02d}  池 {r['index']:02d}  D={r['D_m']:.2f} m" for rank, r in enumerate(order, 1)]
    for ax, mode, matrix in zip(axes, modes, matrices):
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap=cmap,
                       extent=(0, horizon, len(order)-.5, -.5))
        ax.set_yticks(range(len(order)), tick_labels, fontproperties=FONT, fontsize=9)
        ax.set_ylabel("精英按确认进度排序（保留原池索引）", fontproperties=FONT)
        ax.set_xlabel("动作时间 t（环境步；120 步 = 1 秒）", fontproperties=FONT)
        share = data["modes"][mode]["expected_tree_frame_share"]
        ax.set_title(f"{SAMPLING_NAMES[mode]}：树区 / 高处接触候选的期望帧占比 {share:.1%}", fontproperties=FONT, fontsize=12)
        for y, row in enumerate(order):
            ax.plot([row["cap"]-.5]*2, [y-.38,y+.38], color="#333333", linewidth=1)
        ax.set_xlim(0, horizon)
    bar = fig.colorbar(im, ax=axes, shrink=.85, pad=.02)
    bar.set_label("单次抽片段包含该动作的概率（%）", fontproperties=FONT)
    fig.suptitle("当前完整精英池的 SIL 采样机会：加权规则比较", fontproperties=FONT, fontsize=15)
    fig.supxlabel("黑短线为封顶边界；灰区不采样；只展示可采前缀（完整轨迹仍保留）。\n"
                  "同一色标、按真实连续片段概率展开；尚未经过正优势 / KL 筛选，不是训练效果。\n"
                  f"来源：{Path(data['source_pool']['path']).name}；{len(order)} 条精英；片段长度最多 {data['seq_len']} 步。",
                  fontproperties=FONT, fontsize=9)
    heatmap = output / "sampling_coverage.png"
    fig.savefig(heatmap, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 8), facecolor="#faf9f6", layout="constrained")
    ax.set_facecolor("#faf9f6")
    phases = list(PHASE_NAMES)
    colors = ("#8a939b", "#d3b185", "#ca8a37", "#226f9a", "#afbbc1")
    available = [m for m in SAMPLING_NAMES if m in data["modes"]]
    width = .8 / len(available)
    for j, (mode, color) in enumerate(zip(available, colors)):
        report = data["modes"][mode]
        y = np.arange(len(phases)) + (j-(len(available)-1)/2)*width
        expected = [100*report["expected_phase_frame_share"].get(p, 0) for p in phases]
        sampled = [100*report["sampled_phase_frame_share"].get(p, 0) for p in phases]
        ax.barh(y, expected, height=width*.92, color=color, label=SAMPLING_NAMES[mode])
        ax.scatter(sampled, y, marker="|", s=65, color="#171717", zorder=4,
                   label="离线抽样频率" if j == 0 else None)
    ax.set_yticks(range(len(phases)), [PHASE_NAMES[p] for p in phases], fontproperties=FONT, fontsize=10)
    ax.invert_yaxis(); ax.set_axisbelow(True); ax.grid(axis="x", alpha=.2)
    ax.set_xlabel("被采训练帧的阶段占比（%，按片段实际长度加权）", fontproperties=FONT)
    ax.set_ylabel("动作前状态的几何阶段", fontproperties=FONT)
    ax.set_title("SIL 阶段覆盖：精确期望与生产采样器的离线抽样", fontproperties=FONT, fontsize=14)
    ax.legend(prop=FONT, loc="lower right")
    fig.supxlabel(f"每模式 {data['draws_per_mode']:,} 次抽样，seed={data['seed']}；柱为精确期望，黑线为抽样观测。\n"
                  "阶段由几何代理定义；起跳准备跨区域；树区腾空其他包含下落，不能全部视为成功突破。\n"
                  f"来源：{Path(data['source_pool']['path']).name}；未执行模型更新。", fontproperties=FONT, fontsize=9)
    phases_path = output / "sampling_phases.png"
    fig.savefig(phases_path, dpi=150); plt.close(fig)
    return heatmap, phases_path, *plot_dynamics_sampling(data, output)


def plot_dynamics_sampling(data, output):
    """Separate trajectory value, physical opportunity, and actual chunk probability."""
    if "dynamics" not in data: return ()
    modes = data["comparison_modes"]
    order = sorted(data["trajectories"], key=lambda r: (-r["D_m"], r["index"]))
    colors = ("#bd8a43", "#226f9a")
    fig, ax = plt.subplots(figsize=(13,6), layout="constrained", facecolor="#faf9f6")
    x = np.arange(len(order))
    for j,(mode,color) in enumerate(zip(modes,colors)):
        mass = data["modes"][mode]["trajectory_probabilities"]
        values = [100*mass.get(row["index"], mass.get(str(row["index"]),0)) for row in order]
        bars = ax.bar(x+(j-.5)*.36,values,.34,color=color,label=SAMPLING_NAMES[mode])
        ax.bar_label(bars,fmt="%.1f",fontsize=8)
    ax.set_xticks(x,[f"{i+1}\nD={r['D_m']:.1f}" for i,r in enumerate(order)],fontsize=9)
    ax.set_xlabel("精英按确认进度排序（D：米）",fontproperties=FONT)
    ax.set_ylabel("一次 SIL 抽样选择该轨迹的概率（%）",fontproperties=FONT)
    ax.set_title("先决定哪些轨迹值得多练：确认进度排名决定总份额",fontproperties=FONT,fontsize=14)
    ax.legend(prop=FONT);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel(f"整条轨迹的所有可采片段概率之和；轨迹长不会稀释其总份额。\n"
                  f"来源：{Path(data['source_pool']['path']).name}；这些是筛选前机会，尚未训练。",fontproperties=FONT,fontsize=9)
    mass_path=output/"sampling_trajectory_mass.png";fig.savefig(mass_path,dpi=150);plt.close(fig)

    top=order[:2]
    fig,axes=plt.subplots(3,len(top),figsize=(15,10),sharex="col",squeeze=False,
                           layout="constrained",facecolor="#faf9f6")
    for column,row in enumerate(top):
        i,cap=row["index"],row["cap"]
        d=data["dynamics"].get(i,data["dynamics"].get(str(i)))
        t=np.arange(cap);core=np.asarray(d["core"]);priority=np.asarray(d["priority"])
        axes[0,column].plot(t,d["speed_gain_acceleration"],color="#226f9a",lw=1,label="速度增长分量")
        axes[0,column].plot(t,d["turning_acceleration"],color="#ba7541",lw=1,label="转向分量")
        axes[0,column].set_yscale("symlog",linthresh=5)
        axes[0,column].set_ylabel("扣除重力与阻力（m/s²；对称对数）",fontproperties=FONT)
        axes[0,column].set_title(f"轨迹 {column+1} · 池 {i} · D={row['D_m']:.2f} m",fontproperties=FONT,fontsize=12)
        axes[0,column].legend(prop=FONT,loc="upper right")
        axes[1,column].fill_between(t,priority,step="post",color="#dfbd81",label="准备 / 响应上下文")
        axes[1,column].fill_between(t,np.where(core,priority,0),step="post",color="#226f9a",label="接触加速 / 转向核心")
        axes[1,column].set(ylim=(0,1.05))
        axes[1,column].set_ylabel("动作重要性（0–1；采样前代理）",fontproperties=FONT)
        axes[1,column].legend(prop=FONT,loc="upper right")
        for mode,color in zip(modes,colors):
            values=np.zeros(cap)
            for chunk in data["modes"][mode]["chunks"]:
                if chunk["trajectory"]==i: values[chunk["start"]:chunk["end"]]=100*chunk["probability"]
            axes[2,column].step(t,values,where="post",color=color,label=SAMPLING_NAMES[mode])
        axes[2,column].set_ylabel("该动作被抽入训练片段的概率（%）",fontproperties=FONT)
        axes[2,column].set_xlabel("动作 t（环境步；120 步 = 1 秒）",fontproperties=FONT)
        axes[2,column].legend(prop=FONT,loc="upper right")
        for ax in axes[:,column]: ax.set_xlim(0,cap);ax.grid(alpha=.15)
    fig.suptitle("轨迹 1、2：质心动力学 → 动作重要性 → 实际连续片段概率",fontproperties=FONT,fontsize=15)
    fig.supxlabel("动力学基于六个刚体的质量加权速度；第一行是原始分量，第二行已筛掉静止支撑与纯锅体吸能碰撞。\n"
                  "核心需接触代理；前 12 步与后 6 步衰减，不延伸到腾空峰值。片段长最多 32 步，可能包含少量空中帧。\n"
                  f"来源：{Path(data['source_pool']['path']).name}；未执行模型更新，不能据此断言动作的因果价值。",
                  fontproperties=FONT,fontsize=9)
    detail_path=output/"sampling_dynamics.png";fig.savefig(detail_path,dpi=150);plt.close(fig)
    return mass_path,detail_path


def write_sampling_preview(pool, checkpoint, output, draws, seed):
    import torch
    from training.climb_experiment import identity
    from training.elite_archive import EliteTrajectoryArchive
    from training.reference_fingerprint import reference_fingerprint, sidecar_path
    from training.reference_corridor import ReferenceCorridor
    cfg = TrainConfig(**torch.load(checkpoint, map_location="cpu", weights_only=False)["config"])
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    fp = reference_fingerprint(cfg, route_meta=route.metadata, schema_id="anchor35", raw_dim=35)
    archive = EliteTrajectoryArchive(cfg, pool.parent, fp); archive.restore(pool)
    data = sampling_catalog(archive.trajectories, cfg, draws=draws, seed=seed)
    data.update(source_pool=identity(pool), checkpoint=identity(checkpoint), pool_fingerprint=identity(sidecar_path(pool)),
                source_files=[identity(ROOT/"src/training/gru_sil.py"),
                              identity(ROOT/"src/training/climb_diagnostics.py"), identity(Path(__file__))])
    paths = plot_sampling(data, output)
    path = output / "sampling.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return *paths, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, help="单个 elites_iter_*.npz")
    parser.add_argument("--elites-dir", type=Path, help="含 elites_iter_*.npz 的目录")
    parser.add_argument("--archive-runs", nargs="+", type=Path,
                        help="按时间顺序合并正式 run 的全部完整精英快照，离线查看第二层包络")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, help="叠加最后一个池快照中指定轨迹的完整 D_t 曲线")
    parser.add_argument("--sampling-checkpoint", type=Path, help="离线比较原 / 多数阶段 / 软阶段事件加权的采样机会；不训练")
    parser.add_argument("--sampling-draws", type=int, default=20000)
    parser.add_argument("--sampling-seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.archive_runs:
        if args.pool or args.elites_dir or args.sampling_checkpoint or args.trajectory_index is not None:
            parser.error("--archive-runs 不能与单池、历史曲线或采样预览模式混用")
    elif args.pool is None and args.elites_dir is None:
        parser.error("需要 --pool、--elites-dir 或 --archive-runs")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.archive_runs:
        inspect_archive_layers(args.archive_runs, args.output)
        return 0
    pools = [args.pool] if args.pool is not None else []
    if args.elites_dir is not None:
        found = discover_pools(args.elites_dir)
        if not found:
            raise FileNotFoundError(f"No elites_iter_*.npz in {args.elites_dir}")
        if args.pool is None:
            pools = found
        elif args.pool not in found:
            pools = [*found, args.pool]
    catalogs = [catalog_pool(load_trajectories(path), source=str(path.resolve())) for path in pools]
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = plot_snapshot(catalogs[-1], args.output)
    history = plot_history(catalogs, args.output)
    report = write_report(catalogs, args.output)
    if args.trajectory_index is not None:
        print(plot_trajectory_comparison(load_trajectories(pools[-1]), args.trajectory_index,
                                         catalogs[-1], args.output), flush=True)
    if args.sampling_checkpoint:
        for path in write_sampling_preview(pools[-1], args.sampling_checkpoint, args.output,
                                           args.sampling_draws, args.sampling_seed):
            print(path, flush=True)
    print(snapshot, flush=True)
    if history is not None:
        print(history, flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
