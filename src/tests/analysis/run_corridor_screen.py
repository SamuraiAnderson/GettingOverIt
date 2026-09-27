"""Finite, serial execution and reporting for the approved six-hour corridor plan.

Uses existing training/evaluation entry points unchanged. No scheduling service,
automatic retries, parameter search, or recurring runs. Deadlines are mandatory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "runs/corridor_screen_20260919"
SOURCE = "runs/p2_contact/checkpoints/ppo_iter_0335.pt"
FLAGS = {
    "B0": ["--features", "none", "--policy-backbone", "transformer", "--no-normalize-action-history"],
    "B1": ["--features", "none", "--policy-backbone", "transformer", "--normalize-action-history"],
    "R": ["--features", "relations", "--policy-backbone", "transformer", "--normalize-action-history"],
    "E": ["--features", "relations", "events", "--policy-backbone", "transformer", "--normalize-action-history"],
    "G": ["--features", "relations", "events", "--policy-backbone", "gru", "--normalize-action-history"],
}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def run_name(arm, seed):
    if arm not in FLAGS or seed not in (7, 17):
        raise ValueError("Only approved arms and training seeds are supported")
    return f"corridor_screen_20260919_{arm}_s{seed}"


def stats(arm, seed):
    return stats_for_run(ROOT / "runs" / run_name(arm, seed))


def stats_for_run(base):
    path = base / "metrics.csv"
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


def summarize_all():
    return summarize_runs(CAMPAIGN, [(a, s, ROOT / "runs" / run_name(a, s))
                                    for a in FLAGS for s in (7, 17)])


def summarize_runs(campaign, run_paths):
    runs = []
    for arm, seed, base in run_paths:
        if not base.exists():
            continue
        evaluations = []
        for directory in sorted((base / "checkpoints").glob("eval_*")):
            if (directory / "results.json").exists():
                evaluations.append(evaluation_summary(directory))
        runs.append({"arm": arm, "seed": seed, **stats_for_run(base), "evaluations": evaluations})
    output = {"updated_at": now(), "runs": runs,
              "training_physical_samples": sum(r["physical_samples"] for r in runs),
              "evaluation_physical_samples": sum(e["physical_samples"] for r in runs for e in r["evaluations"])}
    write_json(campaign / "summary.json", output)
    return output


class BudgetStop(Exception):
    pass


class Runner:
    def __init__(self, deadline, *, campaign=CAMPAIGN, flags=None, source=SOURCE):
        self.deadline = deadline
        self.campaign = campaign
        self.flags = FLAGS if flags is None else flags
        self.source = source
        self.job = None
        self.child = None
        self.invocation = datetime.now().strftime("%H%M%S")
        self.frozen_sources = json.loads((self.campaign / "source_hashes.json").read_text(encoding="utf-8-sig"))

    def run_name(self, arm, seed):
        if arm not in self.flags or seed not in (7, 17):
            raise ValueError("Only configured arms and approved training seeds are supported")
        return f"{self.campaign.name}_{arm}_s{seed}"

    def stats(self, arm, seed):
        return stats_for_run(ROOT / "runs" / self.run_name(arm, seed))

    def event(self, kind, **data):
        row = {"time": now(), "event": kind, **data}
        with (self.campaign / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def guard(self, estimated_seconds=0):
        if (self.campaign / "STOP").exists():
            raise BudgetStop("STOP marker requested a stop at the next job boundary")
        if datetime.now().astimezone().timestamp() + estimated_seconds + 60 >= self.deadline.timestamp():
            raise BudgetStop("Insufficient time for another bounded job before phase deadline")
        for source in self.frozen_sources:
            if hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest().lower() != source["sha256"].lower():
                raise ValueError(f"Frozen training source changed: {source['path']}")

    def kill_own_child(self):
        if self.child is not None and self.child.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.child.pid), "/T", "/F"], capture_output=True)
            self.child.wait(timeout=30)

    def call(self, module, arguments, label, estimate):
        self.guard(estimate)
        self.job = {"label": label, "started_at": now(), "command": [sys.executable, "-m", module, *map(str, arguments)]}
        self.event("job_started", **self.job)
        log = self.campaign / "logs" / f"{self.invocation}_{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            self.child = subprocess.Popen(self.job["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
            last_status = 0.
            try:
                while self.child.poll() is None:
                    if datetime.now().astimezone().timestamp() >= self.deadline.timestamp():
                        self.kill_own_child()
                        raise BudgetStop("Phase deadline reached; stopped only the launched process tree")
                    if time.monotonic() - last_status > 30:
                        live = {"status": "running", "deadline": self.deadline.isoformat(), "job": self.job,
                                "updated_at": now(), "child_pid": self.child.pid,
                                "training": [{"arm": a, "seed": s, **self.stats(a, s)} for a in self.flags for s in (7, 17)
                                             if (ROOT / "runs" / self.run_name(a, s)).exists()]}
                        write_json(self.campaign / "status.json", live)
                        last_status = time.monotonic()
                    time.sleep(1)
            except BaseException:
                self.kill_own_child()
                raise
        code = self.child.returncode
        self.event("job_finished", label=label, exit_code=code, log=str(log.relative_to(ROOT)))
        if code:
            raise RuntimeError(f"{label} failed (exit {code}); inspect {log}")

    def checkpoint(self, arm, seed):
        current = self.stats(arm, seed)
        path = ROOT / "runs" / self.run_name(arm, seed) / "checkpoints" / f"ppo_iter_{current['last_iteration']:04d}.pt"
        if not path.exists():
            raise RuntimeError(f"Metrics endpoint has no matching checkpoint: {path}; no silent rollback")
        return path

    def preflight(self, arm, seed):
        base = ROOT / "runs" / self.run_name(arm, seed)
        if self.stats(arm, seed)["last_iteration"]:
            meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
            if meta["status"] != "completed":
                raise RuntimeError(f"Existing run is not completed: {base}")
            self.checkpoint(arm, seed)
            return
        if base.exists():
            raise RuntimeError(f"Refusing to overwrite an existing incomplete run: {base}")
        arguments = ["--init-from", self.source, "--reference-trajectory", "--reference-corridor-radius", "1.5",
            "--geometry-profile", "native-pose-v1", "--no-random-deploy", "--recipe", "ppo-ou", "--seed", seed,
            "--num-agents", 2, "--steps-per-rollout", 1200, "--max-iterations", 3, "--value-warmup-iters", 2,
            "--lr", .0001, "--ent-coef", .001, "--event-source", "native", "--no-event-branch",
            "--action-segment-steps", 1, "--recurrent-seq-len", 16, "--feature-step-seconds", 1/120,
            "--run-dir", self.run_name(arm, seed), "--persist-game", "--save-rollout-inputs", *self.flags[arm]]
        self.call("src.training.main_ppo", arguments, f"{arm}_s{seed}_preflight", 300)

    def train_to(self, arm, seed, target):
        while self.stats(arm, seed)["physical_samples"] < target:
            current = self.stats(arm, seed)
            remaining = target - current["physical_samples"]
            if current["last_iteration"] == 3:
                endpoint = 20
                random_seed = 100000 + seed
            elif target > 48000 and current["physical_samples"] >= 48000:
                endpoint = current["last_iteration"] + min(20, max(1, math.ceil(remaining / 2400)))
                random_seed = 200000 + seed + current["last_iteration"] - 20
            else:
                endpoint = current["last_iteration"] + min(5, max(1, math.ceil(remaining / 2400)))
                random_seed = 100000 + seed + current["last_iteration"]
            count = endpoint - current["last_iteration"]
            estimate = 35 + count * max(25, current["recent_iteration_seconds"]) * 1.3
            # Reserve one full endpoint evaluation even if training must stop short.
            self.guard(estimate + 480)
            arguments = ["--resume", str(self.checkpoint(arm, seed)), "--seed", random_seed,
                         "--max-iterations", endpoint, "--run-dir", self.run_name(arm, seed),
                         "--persist-game", "--save-rollout-inputs"]
            self.call("src.training.main_ppo", arguments, f"{arm}_s{seed}_train_{endpoint:04d}", estimate)

    def evaluate(self, arm, seed, stage, rounds, evaluation_seed):
        base = ROOT / "runs" / self.run_name(arm, seed) / "checkpoints"
        directory = base / f"eval_{stage}"
        if directory.exists():
            if not (directory / "results.json").exists():
                raise RuntimeError(f"Existing incomplete evaluation: {directory}")
            result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
            if result["failure"] or any(result["summary"].get(mode, {}).get("episodes") != rounds * 2
                                         for mode in ("deterministic", "exploration")):
                raise RuntimeError(f"Evaluation is incomplete/failed: {directory}")
            return
        checkpoint = base / "rollouts/behavior_0001.pt" if stage == "initial" else self.checkpoint(arm, seed)
        arguments = ["--checkpoint", str(checkpoint), "--output", str(directory), "--steps", 1200,
                     "--rounds", rounds, "--num-agents", 2, "--seed", evaluation_seed, "--jitter", .01,
                     "--goal-progress", 5, "--dwell", 10, "--noise-phi", .85, "--native-diagnostics"]
        self.call("src.training.local_evaluation", arguments, f"{arm}_s{seed}_{stage}", 60 + rounds*4800/50)
        summary = evaluation_summary(directory)
        write_json(self.campaign / f"{arm}_s{seed}_{stage}.json", summary)
        self.event("evaluation", arm=arm, seed=seed, stage=stage, summary=summary["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "replicate", "summary"), required=True)
    parser.add_argument("--arms", choices=tuple(FLAGS), nargs="+", default=list(FLAGS))
    parser.add_argument("--deadline", help="Explicit timezone-aware ISO timestamp; never later than campaign deadline")
    parser.add_argument("--target-samples", type=int, choices=(72000, 96000), default=96000,
                        help="Common replication budget for all selected arms and seeds")
    args = parser.parse_args()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    if args.phase == "summary":
        output = summarize_all()
        print(json.dumps({k: v for k, v in output.items() if k != "runs"}))
        return
    if not args.deadline:
        parser.error("Execution requires an explicit deadline")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("Deadline must include timezone")
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text(encoding="utf-8-sig"))
    if deadline.timestamp() > datetime.fromisoformat(manifest["deadline"]).timestamp():
        parser.error("Phase deadline exceeds the approved six-hour budget")
    if hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest().lower() != manifest["initial_checkpoint_sha256"].lower():
        raise ValueError("Initial checkpoint changed")
    reference = ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json"
    if hashlib.sha256(reference.read_bytes()).hexdigest().lower() != manifest["reference_sha256"].lower():
        raise ValueError("Reference trajectory changed")
    if len(set(args.arms)) != len(args.arms):
        parser.error("Duplicate arms are not allowed")
    runner = Runner(deadline)
    lock = CAMPAIGN / "controller.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"started_at": now(), "phase": args.phase}))
    status, error = "completed", None
    try:
        runner.event("phase_started", phase=args.phase, arms=args.arms, deadline=deadline.isoformat())
        if args.phase == "screen":
            for arm in args.arms:
                runner.preflight(arm, 7)
                runner.evaluate(arm, 7, "initial", 2, 20260919)
                runner.train_to(arm, 7, 48000)
                runner.evaluate(arm, 7, "screen_048k", 4, 20260919)
        else:
            for arm in args.arms:
                for seed in (7, 17):
                    runner.preflight(arm, seed)
                    runner.evaluate(arm, seed, "initial", 2, 20260919)
                    runner.train_to(arm, seed, args.target_samples)
                    runner.evaluate(arm, seed, f"holdout_{args.target_samples // 1000:03d}k", 5, 20261019)
    except BudgetStop as exc:
        status, error = "budget_stopped", str(exc)
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        runner.kill_own_child()
        write_json(CAMPAIGN / "status.json", {"status": status, "phase": args.phase, "updated_at": now(), "error": error, "last_job": runner.job})
        runner.event("phase_finished", phase=args.phase, status=status, error=error)
        lock.unlink()
        summarize_all()


if __name__ == "__main__":
    main()
