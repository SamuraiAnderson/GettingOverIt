"""Six-hour formal SIL-only training: inherit checkpoint σ, no forced anneal."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import psutil

from .run_corridor_formal import FormalRunner, MIDPOINT, MAX_BATCH_SAMPLES
from .run_corridor_screen import ROOT, SOURCE, BudgetStop, now, write_json

SEEDS = (31, 47, 61)
ARM = "S"
BASELINE = ROOT / "runs/corridor_formal_20260922"
SIL_SEED = ROOT / "runs/best_motion_20260923/sil_seed_590.npz"
ENDPOINT_CANDIDATES = (384000, 288000, 240000, 192000, 144000, MIDPOINT)


def training_arguments(campaign, seed):
    return list(map(str, [
        "--init-from", SOURCE, "--reference-trajectory",
        "--reference-corridor-radius", 1.5, "--reference-deviation-weight", .01,
        "--geometry-profile", "native-pose-v1", "--no-random-deploy", "--recipe", "ppo-ou",
        "--seed", seed, "--num-agents", 8, "--steps-per-rollout", 1200,
        "--max-iterations", 20, "--value-warmup-iters", 2, "--lr", .00003, "--ent-coef", .001,
        "--ppo-batch-size", 128, "--save-interval", 1, "--torch-threads", 4,
        "--features", "none", "--policy-backbone", "transformer", "--normalize-action-history",
        "--event-source", "native", "--no-event-branch", "--action-segment-steps", 1,
        "--run-dir", f"{campaign.name}_{ARM}_s{seed}", "--persist-game", "--save-rollout-inputs",
        "--gamma", .99, "--gae-lambda", .95, "--corridor-sil", "--sil-seed", SIL_SEED,
        "--target-physical-samples", MIDPOINT,
    ]))


def resume_arguments(campaign, seed, current, target):
    remaining = target - current["physical_samples"]
    count = min(5, max(1, math.ceil(remaining / MAX_BATCH_SAMPLES)))
    endpoint = current["last_iteration"] + count
    base = ROOT / "runs" / f"{campaign.name}_{ARM}_s{seed}"
    return list(map(str, [
        "--resume", base / "checkpoints" / f"ppo_iter_{current['last_iteration']:04d}.pt",
        "--sil-resume-pool", base / "elites" / f"elites_iter_{current['last_iteration']:04d}.npz",
        "--corridor-sil", "--seed", 100000 + seed + current["last_iteration"],
        "--max-iterations", endpoint, "--additional-physical-samples", remaining,
        "--torch-threads", 4, "--save-interval", 1,
        "--run-dir", f"{campaign.name}_{ARM}_s{seed}", "--persist-game", "--save-rollout-inputs",
    ])), count


def configuration_reference(seed):
    path = ROOT / f"runs/corridor_formal_20260922_C_s{seed}/checkpoints/eval_midpoint_096k/manifest.json"
    if not path.exists():
        path = ROOT / "runs/corridor_formal_20260922_C_s31/checkpoints/eval_midpoint_096k/manifest.json"
    return path


def validate_configuration(config, seed):
    baseline = json.loads(configuration_reference(seed).read_text(encoding="utf-8"))["config"]
    permitted = {"sil_enabled", "sil_advantage_weighting_enabled", "max_iterations",
                 "checkpoint_dir", "log_dir"}
    differences = {key: {"baseline": value, "current": config.get(key)}
                   for key, value in baseline.items() if config.get(key) != value}
    unexpected = set(differences) - permitted
    if unexpected:
        raise ValueError(f"Unexpected formal configuration changes: {unexpected}")
    if not config.get("sil_enabled") or not config.get("sil_advantage_weighting_enabled"):
        raise ValueError("Corridor SIL was not enabled")
    if config.get("ppo_reset_log_std_on_init"):
        raise ValueError("Formal SIL must inherit checkpoint σ")
    if float(config.get("ppo_log_std_anneal_rate") or 0) != 0:
        raise ValueError("Formal SIL must not force-anneal σ")
    if abs(float(config.get("ou_phi", 0)) - .85) > 1e-9:
        raise ValueError("ou_phi must stay 0.85")
    return differences


def choose_endpoint(current, seconds_left, evaluation_seconds):
    """Use timing only; never inspect an evaluation score to allocate training."""
    n = len(current)
    estimates = []
    for target in ENDPOINT_CANDIDATES:
        estimate = 180. + n * evaluation_seconds
        for row in current:
            count = math.ceil(max(0, target - row["physical_samples"]) /
                              max(1, row["recent_samples_per_iteration"]))
            estimate += count * (max(60., row["recent_iteration_seconds"]) * 1.5 + 5.)
            estimate += math.ceil(count / 5) * 45.
        estimates.append({"target_samples": target, "estimated_seconds": estimate})
        if estimate < seconds_left:
            return target, estimates
    raise BudgetStop("Insufficient time for all holdout evaluations")


class SILFormalRunner(FormalRunner):
    def __init__(self, deadline, campaign):
        super().__init__(deadline, campaign)
        self.flags = {ARM: {"gamma": .99, "gae_lambda": .95}}

    def run_name(self, arm, seed):
        if arm != ARM or seed not in SEEDS:
            raise ValueError("Only the SIL arm and seeds 31/47/61 are authorized")
        return f"{self.campaign.name}_{ARM}_s{seed}"

    def status(self, status="running", **extra):
        write_json(self.campaign / "status.json", {
            "status": status, "updated_at": now(), "deadline": self.deadline.isoformat(),
            "controller_pid": self.pid, "controller_create_time": self.created,
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "job": self.job,
            "training": [{"arm": ARM, "seed": s, **self.stats(ARM, s)} for s in SEEDS
                         if (ROOT / "runs" / self.run_name(ARM, s)).exists()],
            **extra,
        })

    def train_midpoint(self, seed):
        self.call("src.training.main_ppo", training_arguments(self.campaign, seed),
                  f"{ARM}_s{seed}_train_096k", 1500)
        stats = self.stats(ARM, seed)
        if not MIDPOINT <= stats["physical_samples"] < MIDPOINT + MAX_BATCH_SAMPLES:
            raise RuntimeError("Training did not reach the 96k sample endpoint")
        if not stats["ppo_updates"] or stats["max_frozen_ratio_error"] > .001:
            raise RuntimeError("Missing PPO updates or failed frozen-policy audit")
        self.checkpoint(ARM, seed)
        base = ROOT / "runs" / self.run_name(ARM, seed)
        meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
        differences = validate_configuration(meta["config"], seed)
        with (base / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        eligible = [r for r in rows if int(r["iteration"]) >= 6 and not float(r.get("value_only") or 0)]
        fired = [r for r in eligible if float(r.get("sil_updates") or 0) > 0]
        skipped = [r for r in eligible if float(r.get("sil_skipped") or 0) > 0]
        if not eligible or len(fired) + len(skipped) != len(eligible):
            raise RuntimeError("SIL-eligible round missing both updates and skip marker")
        if any(not np.isfinite(float(r.get("sil_loss") or "nan"))
               or not r.get("sil_adv_pos_frac") or not r.get("sil_budget_cap")
               or not np.isfinite(float(r.get("sil_value_loss") or "nan"))
               for r in fired):
            raise RuntimeError("Fired SIL round missing finite loss / pos_frac / budget cap / value loss")
        write_json(self.campaign / f"{ARM}_s{seed}_training_audit.json", {
            "configuration_reference": str(configuration_reference(seed)),
            "configuration_differences": differences,
            "sil_eligible_iterations": len(eligible),
            "sil_fired_iterations": len(fired),
            "sil_skipped_iterations": len(skipped),
            "sil_updates": sum(float(r["sil_updates"]) for r in fired),
            "sil_pos_frac_by_iteration": {r["iteration"]: float(r["sil_adv_pos_frac"])
                                          for r in eligible if r.get("sil_adv_pos_frac")},
            **stats,
        })

    def train_to(self, seed, target):
        while self.stats(ARM, seed)["physical_samples"] < target:
            current = self.stats(ARM, seed)
            args, count = resume_arguments(self.campaign, seed, current, target)
            pool = Path(args[args.index("--sil-resume-pool") + 1])
            if not pool.exists():
                raise RuntimeError(f"Missing SIL elite pool: {pool}")
            estimate = 45 + count * (max(60., current["recent_iteration_seconds"]) * 1.5 + 5)
            self.call("src.training.main_ppo", args,
                      f"{ARM}_s{seed}_train_{int(args[args.index('--max-iterations') + 1]):04d}",
                      estimate)
            if self.stats(ARM, seed)["physical_samples"] <= current["physical_samples"]:
                raise RuntimeError("Training made no physical-sample progress")
            self.checkpoint(ARM, seed)


def _eval_line(result):
    from src.training.local_evaluation import format_eval_line
    det = result["summary"]["deterministic"]
    exp = result["summary"]["exploration"]
    return (f"实际样本 {result['training_samples_at_checkpoint']:,}；"
            f"det {format_eval_line(det)}；exploration {format_eval_line(exp)}")


def report(runner, status, error=None):
    runs = []
    for seed in SEEDS:
        evaluations = [json.loads(p.read_text(encoding="utf-8"))
                       for p in sorted(runner.campaign.glob(f"{ARM}_s{seed}_*.json"))
                       if not p.name.endswith("_training_audit.json")]
        baseline_path = BASELINE / f"C_s{seed}_midpoint_096k.json"
        runs.append({
            "arm": ARM, "seed": seed, **runner.stats(ARM, seed), "evaluations": evaluations,
            "formal_c_096k": (json.loads(baseline_path.read_text(encoding="utf-8"))
                              if baseline_path.exists() else None),
        })
    output = {"status": status, "updated_at": now(), "error": error, "runs": runs,
              "training_physical_samples": sum(r["physical_samples"] for r in runs),
              "evaluation_physical_samples": sum(e.get("physical_samples", 0) for r in runs
                                                 for e in r["evaluations"])}
    write_json(runner.campaign / "summary.json", output)
    lines = ["# 六小时正式训练：仅 SIL，继承 checkpoint σ", "",
             f"状态：**{status}**；更新时间：{now()}；硬截止：{runner.deadline.isoformat()}。", "",
             "只跑 S 臂：C 配方（gamma=.99、lambda=.95）+ 防线化 corridor SIL。不跑纯 C，不跑 T。",
             "σ 继承 `p2_contact` checkpoint，不重置、不按轮递减；`actor_log_std` 仍走 PPO 梯度。ou_phi=0.85。",
             "8 Players，batch 128，lr=3e-5，B1 Transformer，归一化动作历史，PPO+OU，走廊半宽 1.5 m。",
             "训练 seeds 31/47/61。SIL 种子 sil_seed_590.npz。游戏默认在隔离桌面 GoiRuntime 启动。", "",
             f"已记录训练物理样本 {output['training_physical_samples']:,}（含两轮 critic 预热），"
             f"已完成评估物理样本 {output['evaluation_physical_samples']:,}。", "",
             "每 seed 先连续训练到 96k 以保留内存精英池，再按耗时选择共同终点；续训恢复精英池。"
             "预算决策不读评估成绩。最多允许一个 9,600 样本 rollout 的超额。", ""]
    for run in runs:
        lines += [f"## SIL / seed {run['seed']}", "",
                  f"训练样本 {run['physical_samples']:,}；PPO minibatch 更新 {run['ppo_updates']:,}。", ""]
        if run["formal_c_096k"] is not None:
            lines.append(f"- 正式 C 96k 锚点（跨批参考，无 SIL）：{_eval_line(run['formal_c_096k'])}。")
        for evaluation in run["evaluations"]:
            stage = Path(evaluation.get("directory", evaluation.get("stage", "eval"))).name
            lines.append(f"- {stage}：{_eval_line(evaluation)}。")
        lines.append("")
    lines += ["## 判据与执行边界", "",
              "到达单独计数。保持用容差、允许继续前进：到达后观察 1 秒，投影不低于 4.7 米、至少 95% 时间在走廊内、窗口末回到 5 米、无落水；0.3 米与 95% 为待验证参数。参考缺口记为无法确认，观察窗不足单列。当前阶段用 credited 中位数和下四分位区分改进。",
              "确定性推进、探索推进和末端保留进度分开看；修改 σ 不会改变冻结模型的均值动作。",
              "只有 3 个独立训练 seed。正式 C 锚点来自分段重启、且当时尚无 σ 重置，不能把跨批差异单独归因给 SIL。",
              "STOP 文件在下一轮控制检查时停止当前自有子进程树。硬截止或故障不自动加时。", ""]
    if error:
        lines += [f"停止原因：{error}", ""]
    (runner.campaign / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=6)
    args = parser.parse_args(argv)
    if not 0 < args.hours <= 6:
        parser.error("Budget must be in (0, 6] hours")
    if args.output.resolve().parent != ROOT / "runs":
        parser.error("Output must be a fresh direct child of runs/")
    return args


def main(argv=None):
    args = parse_args(argv)
    campaign = args.output.resolve()
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take it over")
    for seed in SEEDS:
        if (ROOT / "runs" / f"{campaign.name}_{ARM}_s{seed}").exists():
            raise RuntimeError("A planned training directory already exists")
    campaign.mkdir(exist_ok=False)
    (campaign / "logs").mkdir()
    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=args.hours)
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
    manifest = {"status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
                "budget_hours": args.hours, "arms": [ARM], "training_seeds": SEEDS,
                "initial_checkpoint": SOURCE, "sil_seed": str(SIL_SEED), "baseline": str(BASELINE),
                "agents": 8, "batch_size": 128, "lr": .00003, "torch_threads": 4, "mode": "serial",
                "midpoint": MIDPOINT, "endpoint_candidates": list(ENDPOINT_CANDIDATES),
                "sigma_schedule": {"ppo_reset_log_std_on_init": False, "ppo_init_log_std": -1.6,
                                   "ppo_log_std_anneal_rate": 0., "ppo_log_std_min": -2.5,
                                   "ou_phi": .85},
                "sil_guards": {"sil_gate_margin": 0., "sil_target_clamp": 3., "sil_budget_ratio": 1.,
                               "sil_value_coef": .1},
                "midpoint_evaluation_seed": 20260922, "holdout_seed": 20261022,
                "evaluation_rounds": {"midpoint": 2, "holdout": 3},
                "controller_pid": os.getpid(), "controller_create_time": psutil.Process().create_time(),
                "commands": [training_arguments(campaign, s) for s in SEEDS]}
    write_json(campaign / "manifest.json", manifest)
    runner = SILFormalRunner(deadline, campaign)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        report(runner, "running")
        for seed in SEEDS:
            runner.train_midpoint(seed)
            runner.evaluate(ARM, seed, "midpoint_096k", 2, 20260922)
            report(runner, "running")
        mid_evals = [json.loads(p.read_text(encoding="utf-8")) for p in campaign.glob("*_midpoint_096k.json")]
        final_eval_seconds = max(540., max(e["wall_seconds"] for e in mid_evals) * 1.5)
        target, estimates = choose_endpoint([runner.stats(ARM, s) for s in SEEDS],
                                            (deadline - datetime.now().astimezone()).total_seconds(),
                                            final_eval_seconds)
        write_json(campaign / "budget_decision.json", {
            "decided_at": now(), "target_samples": target, "estimates": estimates,
            "per_holdout_seconds": final_eval_seconds,
            "basis": "timing only; all three SIL runs retained; holdout not yet evaluated"})
        for seed in SEEDS:
            runner.train_to(seed, target)
            if runner.stats(ARM, seed)["physical_samples"] < target:
                raise RuntimeError("Holdout sample endpoint not reached")
            runner.evaluate(ARM, seed, f"holdout_{target // 1000:03d}k", 3, 20261022)
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
