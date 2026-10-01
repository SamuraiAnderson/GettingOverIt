"""Twelve-hour confirmed-endpoint run from the sparse-geometry actor.

The first segment uses --init-from so the legacy checkpoint is not resumed
under legacy_credit_v1. Later segments resume this run's own v2 checkpoints.

    conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.run_v2_formal --output runs/confirmed_endpoint_v2_formal_20260928 --hours 12
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import psutil

from .prepare_v2_formal import CAMPAIGN, POOL, SOURCE, training_arguments
from .run_corridor_formal import MIDPOINT
from .run_corridor_screen import ROOT, BudgetStop, now, write_json
from .run_corridor_sil_td_pareto import (
    ARCHIVE_MODE, ARM, EVAL_ROUNDS, EVAL_SEED, SEED, SILParetoRunner,
    _eval_line, _sil_rows, choose_next_target, resume_arguments,
)

MAX_HOURS = 16


class V2FormalRunner(SILParetoRunner):
    def train_midpoint(self, seed):
        self.call("src.training.main_ppo", training_arguments(), f"{ARM}_s{seed}_train_096k", 1500)
        stats = self.stats(ARM, seed)
        if not MIDPOINT <= stats["physical_samples"] < MIDPOINT + 8 * 1200:
            raise RuntimeError("Training did not reach the 96k sample endpoint")
        if not stats["ppo_updates"] or stats["max_frozen_ratio_error"] > .001:
            raise RuntimeError("Missing PPO updates or failed frozen-policy audit")
        self.checkpoint(ARM, seed)
        self.write_training_audit(seed)

    def write_training_audit(self, seed):
        base = ROOT / "runs" / self.run_name(ARM, seed)
        meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
        config = meta["config"]
        if config.get("reference_progress_mode") != "confirmed_endpoint_v2":
            raise RuntimeError("Run left confirmed_endpoint_v2")
        if config.get("sil_archive_mode") != ARCHIVE_MODE or not config.get("sil_enabled"):
            raise RuntimeError("Corridor SIL td_pareto was not kept")
        if config.get("ppo_reset_log_std_on_init") or float(config.get("ppo_log_std_anneal_rate") or 0) != 0:
            raise RuntimeError("Sigma schedule changed")
        _rows, eligible, fired, skipped = _sil_rows(base)
        if not eligible or len(fired) + len(skipped) != len(eligible):
            raise RuntimeError("SIL-eligible round missing both updates and skip marker")
        write_json(self.campaign / f"{ARM}_s{seed}_training_audit.json", {
            "reference_progress_mode": config.get("reference_progress_mode"),
            "sil_archive_mode": config.get("sil_archive_mode"),
            "sil_eligible_iterations": len(eligible),
            "sil_fired_iterations": len(fired),
            "sil_skipped_iterations": len(skipped),
            **self.stats(ARM, seed),
        })


def report(runner, status, error=None):
    evaluations = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted(runner.campaign.glob(f"{ARM}_s{SEED}_*.json"))
                   if not path.name.endswith("_training_audit.json")]
    run = {"arm": ARM, "seed": SEED, **runner.stats(ARM, SEED), "evaluations": evaluations}
    output = {"status": status, "updated_at": now(), "error": error, "runs": [run],
              "training_physical_samples": run["physical_samples"],
              "evaluation_physical_samples": sum(item.get("physical_samples", 0) for item in evaluations)}
    write_json(runner.campaign / "summary.json", output)
    lines = ["# 十二小时训练：确认终点 D，seed 47", "",
             f"状态：**{status}**；更新时间：{now()}；硬截止：{runner.deadline.isoformat()}。", "",
             "从 sparse geometry `ppo_iter_0118.pt` 用 `--init-from` 暖启动。",
             "进度口径 `confirmed_endpoint_v2`。critic 重新初始化，动作权重和 σ 继承。",
             "精英池冷启动用 `td_v2_rerank/elites_v2.npz`，不改原来的 `elites_iter_0118.npz`。",
             "8 agents，1200 步，走廊 SIL，`td_pareto`，ou_phi=0.85。游戏在隔离桌面 GoiRuntime。", "",
             f"已记录训练物理样本 {output['training_physical_samples']:,}。", ""]
    for evaluation in evaluations:
        stage = Path(evaluation.get("directory", evaluation.get("stage", "eval"))).name
        lines.append(f"- {stage}：{_eval_line(evaluation)}。")
    if error:
        lines += ["", f"停止原因：{error}"]
    (runner.campaign / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_paths(game_root):
    paths = [path for folder in ("src/training", "src/env", "src/start")
             for path in (ROOT / folder).rglob("*.py")]
    paths += [Path(__file__), ROOT / "src/tests/analysis/prepare_v2_formal.py",
              ROOT / "src/tests/analysis/run_corridor_formal.py",
              ROOT / "src/tests/analysis/run_corridor_screen.py",
              ROOT / "src/tests/analysis/run_corridor_sil_td_pareto.py",
              ROOT / "src/training/local_evaluation.py", SOURCE, POOL,
              POOL.with_name(POOL.name + ".fingerprint.json"),
              ROOT / "src/config/project.json",
              ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json",
              game_root / "BepInEx/plugins/GameRuntime_v2.dll"]
    return paths


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args(argv)
    if not 0 < args.hours <= MAX_HOURS:
        parser.error(f"Budget must be in (0, {MAX_HOURS}] hours")
    if args.output.resolve().parent != ROOT / "runs":
        parser.error("Output must be a direct child of runs/")
    return args


def main(argv=None):
    args = parse_args(argv)
    campaign = args.output.resolve()
    if campaign.name != CAMPAIGN:
        raise RuntimeError(f"This controller only runs {CAMPAIGN}")
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take it over")
    training_dir = ROOT / "runs" / f"{campaign.name}_{ARM}_s{SEED}"
    if training_dir.exists():
        raise RuntimeError("The v2 training directory already exists")
    if not SOURCE.is_file() or not POOL.is_file():
        raise RuntimeError("Missing init checkpoint or rescored elite pool")
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8"))
    game_root = Path(project["game"]["executable_path"]).parent
    runtime = game_root / "GoiData/runtime_config.json"
    campaign.mkdir(exist_ok=True)
    (campaign / "logs").mkdir(exist_ok=True)
    if (campaign / "manifest.json").exists():
        raise RuntimeError("Campaign manifest already exists; refusing to start over it")
    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=args.hours)
    original = runtime.read_bytes() if runtime.exists() else None
    if original is not None:
        (campaign / "runtime_original.json").write_bytes(original)
    manifest = {
        "status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
        "budget_hours": args.hours, "arms": [ARM], "training_seeds": [SEED],
        "initial_checkpoint": str(SOURCE), "sil_init_pool": str(POOL),
        "reference_progress_mode": "confirmed_endpoint_v2",
        "sil_archive_mode": ARCHIVE_MODE, "isolated_desktop": "GoiRuntime",
        "agents": 8, "steps_per_rollout": 1200, "batch_size": 128, "lr": 3e-5,
        "midpoint": MIDPOINT, "controller_pid": os.getpid(),
        "controller_create_time": psutil.Process().create_time(),
        "commands": [training_arguments()],
    }
    write_json(campaign / "source_hashes.json", [
        {"path": os.path.relpath(path, ROOT), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in _source_paths(game_root) if path.is_file()
    ])
    write_json(campaign / "manifest.json", manifest)
    runner = V2FormalRunner(deadline, campaign)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        report(runner, "running")
        runner.train_midpoint(SEED)
        if not (campaign / f"{ARM}_s{SEED}_midpoint_096k.json").exists():
            runner.evaluate(ARM, SEED, "midpoint_096k", EVAL_ROUNDS, EVAL_SEED)
        report(runner, "running")
        mid = json.loads((campaign / f"{ARM}_s{SEED}_midpoint_096k.json").read_text(encoding="utf-8"))
        eval_seconds = max(540., mid["wall_seconds"] * 1.5)
        while True:
            current = runner.stats(ARM, SEED)
            target, estimate = choose_next_target(
                current, (deadline - datetime.now().astimezone()).total_seconds(), eval_seconds)
            write_json(campaign / "budget_decision.json", {
                "decided_at": now(), "target_samples": target, "estimated_seconds": estimate,
                "basis": "timing only; next 96k milestone; evaluation not consulted"})
            runner.train_to(SEED, target)
            runner.evaluate(ARM, SEED, f"milestone_{target // 1000:03d}k", EVAL_ROUNDS, EVAL_SEED)
            report(runner, "running")
    except BudgetStop as exc:
        status, error = "budget_stopped", str(exc)
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        runner.kill_own_child()
        if original is not None:
            runtime.write_bytes(original)
        elif runtime.exists():
            runtime.unlink()
        restored = runtime.read_bytes() == original if original is not None else not runtime.exists()
        manifest.update(status=status, finished_at=now(), error=error, runtime_restored=restored)
        write_json(campaign / "manifest.json", manifest)
        runner.status(status, error=error, runtime_restored=restored)
        report(runner, status, error)
        runner.event("finished", status=status, error=error, runtime_restored=restored)


if __name__ == "__main__":
    main()
