"""从已经写下的 metrics.csv 和评估目录汇总样本。不启动训练。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .io import ROOT


def stats_for_run(base):
    path = Path(base) / "metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline=""))) if path.exists() else []

    def number(row, field, default=0.):
        value = row.get(field)
        return float(value) if value not in (None, "") else default

    ids = [int(r["iteration"]) for r in rows]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError(f"Duplicate or unordered iterations: {path}")
    samples = [number(r, "rollout_environment_steps", number(r, "buffer_size")) for r in rows]
    return {"iterations": len(rows), "last_iteration": max(ids, default=0),
            "physical_samples": int(sum(samples)),
            "warmup_samples": int(sum(s for s, r in zip(samples, rows) if number(r, "value_only") == 1)),
            "ppo_updates": int(sum(number(r, "n_updates") for r in rows if number(r, "value_only") != 1)),
            "collection_seconds": sum(number(r, "rollout_total_seconds") for r in rows),
            "update_seconds": sum(number(r, "update_and_sil_seconds") for r in rows),
            "max_frozen_ratio_error": max((number(r, "preupdate_ratio_max_error") for r in rows), default=None),
            "recent_samples_per_iteration": sum(samples[-5:]) / max(1, len(samples[-5:])),
            "recent_iteration_seconds": sum(number(r, "rollout_total_seconds") + number(r, "update_and_sil_seconds") for r in rows[-5:]) / max(1, len(rows[-5:]))}


def evaluation_summary(directory):
    """Re-score the saved states through the same criterion, with a 1 s hold too."""
    import numpy as np
    from types import SimpleNamespace
    from src.training.local_evaluation import score_corridor_episode, summarize as summarize_episodes
    from training.config import TrainConfig
    from training.reference_corridor import ReferenceCorridor
    directory = Path(directory)
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_name = Path(manifest["checkpoint"]).name
    checkpoint_iteration = int(checkpoint_name.removeprefix("ppo_iter_").removesuffix(".pt")) if checkpoint_name.startswith("ppo_iter_") else 0
    metrics_path = directory.parent.parent / "metrics.csv"
    checkpoint_samples = 0
    if checkpoint_iteration and metrics_path.exists():
        with metrics_path.open(encoding="utf-8-sig", newline="") as stream:
            checkpoint_samples = int(sum(float(row.get("rollout_environment_steps") or row["buffer_size"])
                                         for row in csv.DictReader(stream) if int(row["iteration"]) <= checkpoint_iteration))
    cfg = TrainConfig(**manifest["config"])
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    if route.metadata["sha256"] != manifest["reference"]["sha256"]:
        raise ValueError("Evaluation reference hash changed")
    records = []
    for row in result["episodes"]:
        raw = json.loads((directory / f"{row['mode']}_{row['trial']:03d}_agent{row['agent']}.json").read_text(encoding="utf-8"))
        states = np.array(raw["states"], dtype=np.float32)
        trajectory = SimpleNamespace(raw_states=states, actions=np.array([t["action"] for t in raw["trace"]]), contact=None)
        scored, _ = score_corridor_episode(
            trajectory, route, manifest["criterion"]["credited_progress_m"], dwell=10)
        arrival = scored["arrival_steps"]
        ys = states[:, 1]
        after_fall = None if arrival is None else bool(np.any((ys[arrival:] < cfg.water_y_threshold) | (np.maximum.accumulate(ys)[arrival:] - ys[arrival:] >= 3)))
        records.append({**row, **{k: scored[k] for k in (
            "success", "arrival_steps", "hold_success", "hold_incomplete", "reference_unconfirmed",
            "terminal_progress_m", "terminal_same_segment_progress_m", "continued_on_later_segment",
            "watered", "max_retreat_after_arrival_m")},
                        "success_hold_120": scored["hold_success"], "fall_after_success": after_fall})
    scored_summary = summarize_episodes(records)
    summary = {}
    for mode in ("deterministic", "exploration"):
        rows = [r for r in records if r["mode"] == mode]
        if not rows:
            continue
        summary[mode] = {**result["summary"][mode], **scored_summary[mode],
            "successes": sum(r["success"] for r in rows),
            "success_hold_120_count": sum(r["success_hold_120"] for r in rows),
            "falls_after_success": sum(r["fall_after_success"] is True for r in rows),
            "mean_outside_fraction": float(np.mean([r["outside_fraction"] for r in rows])),
            "rounds": len({r["trial"] for r in rows})}
    return {"directory": str(directory.relative_to(ROOT)), "failure": result["failure"],
            "physical_samples": sum(r["steps"] for r in records), "summary": summary, "episodes": records,
            "checkpoint_iteration": checkpoint_iteration, "training_samples_at_checkpoint": checkpoint_samples}
