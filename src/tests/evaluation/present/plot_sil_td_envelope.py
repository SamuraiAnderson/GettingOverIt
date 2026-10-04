"""Plot the elite-pool (t, D) upper-left envelope from saved SIL snapshots.

Does not launch the game. Points come from `run_logging.load_elites`; the front
is the same `sil_archive.pareto_front` used in training.

    conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.evaluation.present.plot_sil_td_envelope --elites-dir runs/corridor_sil_td_pareto_20260927_S_s47/elites --output runs/corridor_sil_td_pareto_20260927/td_envelope
"""
from __future__ import annotations

import argparse
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
from training.sil_archive import pareto_front, td_point

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, help="单个 elites_iter_*.npz")
    parser.add_argument("--elites-dir", type=Path, help="含 elites_iter_*.npz 的目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, help="叠加最后一个池快照中指定轨迹的完整 D_t 曲线")
    args = parser.parse_args(argv)
    if args.pool is None and args.elites_dir is None:
        parser.error("需要 --pool 或 --elites-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
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
    print(snapshot, flush=True)
    if history is not None:
        print(history, flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
