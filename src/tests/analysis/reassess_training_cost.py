"""Read archived runs and probe policy log-probability round trips without a game.

Run in getting-over-it-analysis. Writes the requested JSON only; does not train.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import types
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from src.training.actor_critic import ActorCritic
from src.training.config import TrainConfig

ROOT = Path(__file__).resolve().parents[3]


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())} if len(a) else None


def timing_sessions(path):
    sessions, current, row = [], [], None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "实验归档目录:" in line:
            if current:
                sessions.append(current)
            current, row = [], None
        m = re.match(r"(\d\d):(\d\d):(\d\d)", line)
        if not m:
            continue
        t = sum(int(x) * k for x, k in zip(m.groups(), [3600, 60, 1]))
        m = re.search(r"=== PPO 迭代 (\d+)/", line)
        if m:
            row = {"iteration": int(m[1]), "start": t}
        if row is None:
            continue
        for needle, key in [("Phase 1: PPO 采集", "collect_start"),
                            ("采集完成: buffer", "collect_end"),
                            ("Phase 2: PPO 更新", "ppo_start"),
                            ("Phase 2.5: Self-Imitation", "sil_start")]:
            if needle in line:
                row[key] = t
        if re.search(r"INFO: iter=\d+\s", line):
            row["end"] = t
            for name, a, b in [("total_s", "start", "end"),
                               ("collect_s", "collect_start", "collect_end"),
                               ("ppo_s", "ppo_start", "sil_start"),
                               ("sil_s", "sil_start", "end"),
                               ("other_s", "collect_end", "ppo_start")]:
                if a in row and b in row:
                    row[name] = (row[b] - row[a]) % 86400
            current.append(row)
            row = None
    if current:
        sessions.append(current)
    # Appended/interleaved logs can contain an end without its collection start.
    # Exclude these incomplete iterations instead of reporting spuriously fast runs.
    return [complete for s in sessions
            if (complete := [r for r in s if "collect_s" in r and r["collect_s"] > 0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"runs": [], "timing": {}}
    for path in sorted((ROOT / "runs").glob("*/metrics.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        def vals(key):
            return [float(r[key]) for r in rows if r.get(key)]
        sizes = vals("buffer_size")
        updates = vals("n_updates")
        best = vals("explore_pool_best_sdy")
        report["runs"].append({"tag": path.parent.name, "rows": len(rows),
            "samples": int(sum(sizes)), "sample_rows": len(sizes),
            "iterations": [int(rows[0]["iteration"]), int(rows[-1]["iteration"])],
            "ppo_updates": stats(updates), "one_update_rows": sum(v == 1 for v in updates),
            "peak_pool_secured_dy": max(best) if best else None})
        log = path.parent / "console.log"
        if log.exists() and path.parent.name in {"p2_contact", "ou_me", "treefix2cont"}:
            sessions = timing_sessions(log)
            report["timing"][path.parent.name] = [{
                "iterations": [s[0]["iteration"], s[-1]["iteration"]],
                **{k: stats([r[k] for r in s if k in r]) for k in
                   ["total_s", "collect_s", "ppo_s", "sil_s", "other_s"]},
                "total_logged_s": sum(r["total_s"] for r in s),
            } for s in sessions]
    report["total_metric_rows"] = sum(r["rows"] for r in report["runs"])
    report["total_recorded_samples"] = sum(r["samples"] for r in report["runs"])
    cfg = TrainConfig()
    torch.manual_seed(20260916)
    torch.set_num_threads(2)
    model = ActorCritic(cfg).eval()
    report["model_parameters"] = sum(p.numel() for p in model.parameters())
    report["model_fp32_weight_mib"] = report["model_parameters"] * 4 / 2**20
    report["cuda_available"] = torch.cuda.is_available()
    report["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
    report["patch_window_memory_gib"] = {
        str(n): n * cfg.context_len * cfg.patch_channels * cfg.patch_size**2 * 4 / 2**30
        for n in [4000, 6400]}
    report["logprob_probe"] = []
    # Isolate the actual sampler/evaluator implementation with a fixed policy.
    # No optimizer, no checkpoint mutation and no game connection.
    n = 32768
    for mean_value in [0.0, 3.0, 4.0, 5.0]:
        def forward(self, dynamics, patches, actions, valid_mask=None):
            mean = torch.full((n, 2), mean_value)
            return Normal(mean, torch.ones_like(mean)), torch.zeros(n)
        model.forward = types.MethodType(forward, model)
        prev = torch.randn(n, 2)
        with torch.no_grad():
            av = model.get_action_and_value(None, None, None,
                noise_prev=prev, noise_phi=cfg.ou_phi)
            lp, _, _ = model.evaluate_actions(None, None, None, av.action,
                noise_prev=prev, noise_phi=cfg.ou_phi)
            lr = lp - av.log_prob
            ratio = lr.exp()
        report["logprob_probe"].append({"mean": mean_value, "std": 1.0,
            "n": n, "ou_phi": cfg.ou_phi,
            "saturated_action_fraction": float((av.action.abs() > 99.9).any(dim=1).float().mean()),
            "approx_kl_before_any_update": float(((ratio - 1) - lr).mean()),
            "clip_fraction_before_any_update": float(((ratio - 1).abs() > cfg.clip_epsilon).float().mean()),
            "median_abs_logprob_error": float(lr.abs().median())})
    elite = ROOT / "runs/p2_contact/elites/elites_iter_0335.npz"
    if elite.exists():
        with np.load(elite, allow_pickle=False) as z:
            actions = [z[k] for k in z.files if "action" in k and z[k].ndim == 2]
            if actions:
                a = np.concatenate(actions)
                report["elite_action_saturation"] = {"rows": len(a),
                    "fraction_any_axis_over_99_9": float(np.mean(np.any(np.abs(a) > 99.9, axis=1)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
