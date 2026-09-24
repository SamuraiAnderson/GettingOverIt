"""Finite C + corridor-SIL comparison against the saved formal 96k anchor."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import psutil

from .run_corridor_formal import FormalRunner
from .run_corridor_screen import ROOT, SOURCE, BudgetStop, now, write_json

SEEDS = (31, 47)
BASELINE = ROOT / "runs/corridor_formal_20260922"
SIL_SEED = ROOT / "runs/best_motion_20260923/sil_seed_590.npz"


def training_arguments(campaign, seed, endpoint):
    args = ["--init-from", SOURCE, "--reference-trajectory",
            "--reference-corridor-radius", 1.5, "--reference-deviation-weight", .01,
            "--geometry-profile", "native-pose-v1", "--no-random-deploy", "--recipe", "ppo-ou",
            "--seed", seed, "--num-agents", 8, "--steps-per-rollout", 1200,
            "--max-iterations", 20 if endpoint == "samples" else 10,
            "--value-warmup-iters", 2, "--lr", .00003, "--ent-coef", .001,
            "--ppo-batch-size", 128, "--save-interval", 1, "--torch-threads", 4,
            "--features", "none", "--policy-backbone", "transformer", "--normalize-action-history",
            "--event-source", "native", "--no-event-branch", "--action-segment-steps", 1,
            "--run-dir", f"{campaign.name}_C_s{seed}", "--persist-game", "--save-rollout-inputs",
            "--gamma", .99, "--gae-lambda", .95, "--corridor-sil", "--sil-seed", SIL_SEED]
    if endpoint == "samples":
        args += ["--target-physical-samples", 96000]
    return list(map(str, args))


def validate_configuration(config, seed):
    path = ROOT / f"runs/corridor_formal_20260922_C_s{seed}/checkpoints/eval_midpoint_096k/manifest.json"
    baseline = json.loads(path.read_text(encoding="utf-8"))["config"]
    permitted = {"sil_enabled", "sil_advantage_weighting_enabled", "max_iterations", "checkpoint_dir", "log_dir"}
    differences = {key: {"baseline": value, "current": config.get(key)}
                   for key, value in baseline.items() if config.get(key) != value}
    unexpected = set(differences) - permitted
    if unexpected:
        raise ValueError(f"Unexpected formal configuration changes: {unexpected}")
    if not config["sil_enabled"] or not config["sil_advantage_weighting_enabled"]:
        raise ValueError("Corridor SIL was not enabled")
    return differences


class SILRunner(FormalRunner):
    def run_name(self, arm, seed):
        if arm != "C" or seed not in SEEDS:
            raise ValueError("Only C seeds 31 and 47 are authorized")
        return f"{self.campaign.name}_C_s{seed}"

    def status(self, status="running", **extra):
        write_json(self.campaign / "status.json", {
            "status": status, "updated_at": now(), "deadline": self.deadline.isoformat(),
            "controller_pid": self.pid, "controller_create_time": self.created,
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "job": self.job, "training": [{"arm": "C", "seed": s, **self.stats("C", s)} for s in SEEDS],
            **extra,
        })

    def train(self, seed, endpoint):
        self.call("src.training.main_ppo", training_arguments(self.campaign, seed, endpoint),
                  f"C_s{seed}_train", 1200)
        stats = self.stats("C", seed)
        if endpoint == "samples" and not 96000 <= stats["physical_samples"] < 105600:
            raise RuntimeError("Training did not reach the agreed sample endpoint")
        if endpoint == "iterations" and stats["iterations"] != 10:
            raise RuntimeError("Training did not complete exactly 10 iterations")
        if not stats["ppo_updates"] or stats["max_frozen_ratio_error"] > .001:
            raise RuntimeError("Missing PPO updates or failed frozen-policy audit")
        self.checkpoint("C", seed)
        base = ROOT / "runs" / self.run_name("C", seed)
        meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
        differences = validate_configuration(meta["config"], seed)
        with (base / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        sil_rows = [r for r in rows if float(r.get("sil_updates") or 0) > 0]
        if not sil_rows or any(not np.isfinite(float(r.get("sil_loss") or "nan"))
                               or not r.get("sil_adv_pos_frac") for r in sil_rows):
            raise RuntimeError("No finite advantage-weighted SIL updates")
        write_json(self.campaign / f"C_s{seed}_training_audit.json", {
            "configuration_differences": differences, "sil_iterations": len(sil_rows),
            "sil_updates": sum(float(r["sil_updates"]) for r in sil_rows), **stats,
        })


def comparison(runner, status, error=None):
    runs = []
    for seed in SEEDS:
        old = json.loads((BASELINE / f"C_s{seed}_midpoint_096k.json").read_text(encoding="utf-8"))
        path = runner.campaign / f"C_s{seed}_midpoint_096k.json"
        new = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        runs.append({"seed": seed, "training": runner.stats("C", seed), "baseline": old,
                     "sil": new, "det_median_delta_m": None if new is None else (
                         new["summary"]["deterministic"]["credited_progress_median_m"]
                         - old["summary"]["deterministic"]["credited_progress_median_m"])})
    pooled = {}
    for group in ("baseline", "sil"):
        if group == "sil" and any(r[group] is None for r in runs):
            continue
        pooled[group] = {}
        for mode in ("deterministic", "exploration"):
            episodes = [e for r in runs for e in r[group]["episodes"] if e["mode"] == mode]
            pooled[group][mode] = {
                "episodes": len(episodes),
                "credited_progress_median_m": float(np.median([e["credited_progress_m"] for e in episodes])),
                "success_hold_120_count": sum(e["success_hold_120"] for e in episodes),
                "successes": sum(e["success"] for e in episodes),
            }
    write_json(runner.campaign / "comparison.json", {
        "status": status, "updated_at": now(), "error": error, "runs": runs, "pooled_episodes": pooled,
        "caveats": ["The quoted 2.72m anchor is C seed 31, not the group aggregate.",
                    "Only two independent training seeds; parallel agents are not independent seeds.",
                    "The SIL seed was selected from a historical T midpoint evaluation; this is a seeded transfer comparison, not a new untouched holdout.",
                    "Training is continuous to retain the in-memory SIL pool; formal training used process restarts and reseeding."],
    })
    lines = ["# C + corridor SIL 对照", "", f"状态：{status}；更新：{now()}。", "",
             "8 agents × 1200 步，batch 128，lr 3e-5，gamma .99 / lambda .95，native-pose-v1，固定起点。",
             "同一历史源权重初始化；两轮 critic 预热计入样本。SIL 使用 sil_seed_590.npz，默认第 6 轮起执行优势加权更新。",
             "每个 seed 连续训练保留内存精英池；达到约定终点后自动评估。评估与正式中期一致：seed 20260922，2 rounds × 8 agents × det/exploration。", ""]
    for row in runs:
        lines += [f"## Seed {row['seed']}", "",
                  f"已训练 {row['training']['physical_samples']:,} 个实际样本，{row['training']['iterations']} 轮。", ""]
        for group, label in (("baseline", "正式 C 96k 锚点"), ("sil", "本次 C + SIL")):
            result = row[group]
            if result is None:
                lines.append(f"- {label}：尚未完成评估。")
                continue
            det = result["summary"]["deterministic"]
            exp = result["summary"]["exploration"]
            lines.append(f"- {label}：实际样本 {result['training_samples_at_checkpoint']:,}；det 中位数 {det['credited_progress_median_m']:.3f}m，exploration {exp['credited_progress_median_m']:.3f}m；det 保持 1 秒 {det['success_hold_120_count']}/{det['episodes']}。")
        if row["det_median_delta_m"] is not None:
            lines.append(f"- 同 seed det 中位数变化：{row['det_median_delta_m']:+.3f}m。")
        lines.append("")
    lines += ["2.72m 对应原 C seed 31。完整分 seed、两个 seed 的回合汇总及实际样本量见 comparison.json。",
              "种子轨迹来自历史 T 组中期评估；本次衡量优质轨迹冷启动的增益，不是未经挑选数据的独立保留集结论。",
              "训练连续执行，不复刻历史 run 分段恢复时的随机数重置。仅两个训练 seed，不将复制体回合数当独立复验。", "",
              "在本目录创建 STOP 文件可终止自有进程树；每轮已完成 checkpoint 保留。结束时恢复启动前游戏运行配置。"]
    if error:
        lines += ["", f"停止原因：{error}"]
    (runner.campaign / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", choices=("samples", "iterations"), default="samples")
    parser.add_argument("--hours", type=float, default=2., help="Safety ceiling, not a training allocation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    campaign = args.output.resolve()
    if campaign.parent != ROOT / "runs" or not 0 < args.hours <= 2:
        parser.error("Use a fresh direct child of runs and at most two hours")
    if args.dry_run:
        print(json.dumps({"commands": [training_arguments(campaign, s, args.endpoint) for s in SEEDS]}, indent=2))
        return
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take over")
    if any((ROOT / "runs" / f"{campaign.name}_C_s{s}").exists() for s in SEEDS):
        raise RuntimeError("Refusing to overwrite a training run")
    campaign.mkdir(exist_ok=False)
    (campaign / "logs").mkdir()
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8"))
    game_root = Path(project["game"]["executable_path"]).parent
    runtime = game_root / "GoiData/runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    if original is not None:
        (campaign / "runtime_original.json").write_bytes(original)
    paths = [p for folder in ("src/training", "src/env", "src/start") for p in (ROOT / folder).rglob("*.py")]
    paths += [Path(__file__), ROOT / "src/tests/analysis/run_corridor_formal.py",
              ROOT / "src/tests/analysis/run_corridor_screen.py", ROOT / "src/training/local_evaluation.py",
              ROOT / SOURCE, SIL_SEED, ROOT / "src/config/project.json",
              ROOT / "src/tests/control_interaction/test_l7_surface_airdrop.py",
              ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json",
              game_root / "BepInEx/plugins/GameRuntime_v2.dll"]
    write_json(campaign / "source_hashes.json", [
        {"path": os.path.relpath(p, ROOT), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=args.hours)
    manifest = {"status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
                "endpoint": args.endpoint, "target_physical_samples_per_seed": 96000 if args.endpoint == "samples" else None,
                "max_iterations": 20 if args.endpoint == "samples" else 10, "training_seeds": SEEDS,
                "initial_checkpoint": SOURCE, "sil_seed": str(SIL_SEED), "baseline": str(BASELINE),
                "warmup_included": True, "sil_first_iteration": 6, "evaluation_seed": 20260922,
                "evaluation_rounds_per_mode": 2, "controller_pid": os.getpid(),
                "controller_create_time": psutil.Process().create_time(),
                "commands": [training_arguments(campaign, s, args.endpoint) for s in SEEDS]}
    write_json(campaign / "manifest.json", manifest)
    runner = SILRunner(deadline, campaign)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        comparison(runner, "running")
        for seed in SEEDS:
            runner.train(seed, args.endpoint)
            runner.evaluate("C", seed, "midpoint_096k", 2, 20260922)
            comparison(runner, "running")
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
        comparison(runner, status, error)
        runner.event("finished", status=status, error=error, runtime_restored=restored)


if __name__ == "__main__":
    main()
