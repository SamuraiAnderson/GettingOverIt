"""One bounded six-hour C/T replication with the measured 8-player configuration."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from .run_corridor_screen import (
    ROOT, SOURCE, BudgetStop, Runner, evaluation_summary, now, stats_for_run, write_json,
)

sys.path.insert(0, str(ROOT / "src"))
SEEDS = (31, 47, 61)
PARAMETERS = {"C": {"gamma": .99, "gae_lambda": .95},
              "T": {"gamma": .998, "gae_lambda": .995}}
ORDER = [(arm, seed) for seed in SEEDS for arm in PARAMETERS]
MIDPOINT = 96000
MAX_BATCH_SAMPLES = 8 * 1200


def choose_endpoint(current, seconds_left, evaluation_seconds):
    """Use timing only; never inspect an evaluation score to allocate training."""
    estimates = []
    for target in (384000, 288000, 240000, 192000, 144000, MIDPOINT):
        estimate = 180. + 6 * evaluation_seconds
        for row in current:
            count = math.ceil(max(0, target-row["physical_samples"]) / max(1, row["recent_samples_per_iteration"]))
            # Measured training phases omit startup, archive writes and checkpoints.
            estimate += count * (max(60., row["recent_iteration_seconds"]) * 1.5 + 5.)
            estimate += math.ceil(count / 5) * 45.
        estimates.append({"target_samples": target, "estimated_seconds": estimate})
        if estimate < seconds_left:
            return target, estimates
    raise BudgetStop("Insufficient time for all six holdout evaluations")


class FormalRunner(Runner):
    def __init__(self, deadline, campaign):
        super().__init__(deadline, campaign=campaign, flags=PARAMETERS)
        self.pid = os.getpid()
        self.created = psutil.Process().create_time()
        self.env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
        self.job_seconds = {}

    def run_name(self, arm, seed):
        if arm not in PARAMETERS or seed not in SEEDS:
            raise ValueError("Unplanned arm/seed")
        return f"{self.campaign.name}_{arm}_s{seed}"

    def status(self, status="running", **extra):
        write_json(self.campaign / "status.json", {
            "status": status, "updated_at": now(), "deadline": self.deadline.isoformat(),
            "controller_pid": self.pid, "controller_create_time": self.created,
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "job": self.job, "training": [{"arm": a, "seed": s, **self.stats(a, s)} for a, s in ORDER],
            **extra,
        })

    def guard(self, estimated_seconds=0):
        super().guard(estimated_seconds)
        if shutil.disk_usage(ROOT).free < 8 * 2**30:
            raise BudgetStop("Less than 8 GiB free; preserve saved checkpoints and stop")

    def call(self, module, arguments, label, estimate):
        self.guard(estimate)
        if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
            raise RuntimeError("An existing game is running; refusing to take it over")
        command = [sys.executable, "-m", module, *map(str, arguments)]
        self.job = {"label": label, "started_at": now(), "command": command}
        self.event("job_started", **self.job)
        log = self.campaign / "logs" / f"{label}.log"
        started = time.monotonic()
        with log.open("x", encoding="utf-8") as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=stream,
                                          stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            last_status = 0.
            try:
                while self.child.poll() is None:
                    if datetime.now().astimezone() >= self.deadline:
                        raise BudgetStop("Six-hour deadline reached")
                    if time.monotonic()-last_status >= 30:
                        self.guard()
                        self.status()
                        last_status = time.monotonic()
                    time.sleep(1)
            except BaseException:
                self.kill_own_child()
                raise
        elapsed = time.monotonic()-started
        self.job_seconds[label] = elapsed
        self.event("job_finished", label=label, exit_code=self.child.returncode, seconds=elapsed)
        if self.child.returncode:
            raise RuntimeError(f"{label} failed; inspect {log}")
        self.status()

    def preflight(self, arm, seed):
        if (ROOT / "runs" / self.run_name(arm, seed)).exists():
            raise RuntimeError("Refusing to overwrite existing training outputs")
        credit = [item for key, value in PARAMETERS[arm].items()
                  for item in ("--"+key.replace("_", "-"), value)]
        args = ["--init-from", SOURCE, "--reference-trajectory", "--reference-corridor-radius", 1.5,
                "--reference-deviation-weight", .01, "--geometry-profile", "native-pose-v1",
                "--no-random-deploy", "--recipe", "ppo-ou", "--seed", seed,
                "--num-agents", 8, "--steps-per-rollout", 1200, "--max-iterations", 3,
                "--value-warmup-iters", 2, "--lr", .00003, "--ent-coef", .001,
                "--ppo-batch-size", 128, "--save-interval", 1, "--torch-threads", 4,
                "--features", "none", "--policy-backbone", "transformer", "--normalize-action-history",
                "--event-source", "native", "--no-event-branch", "--action-segment-steps", 1,
                "--run-dir", self.run_name(arm, seed), "--persist-game", "--save-rollout-inputs", *credit]
        self.call("src.training.main_ppo", args, f"{arm}_s{seed}_preflight", 360)
        current = self.stats(arm, seed)
        if current["ppo_updates"] < 1 or current["max_frozen_ratio_error"] > .001:
            raise RuntimeError("Preflight did not complete a valid PPO update")
        self.checkpoint(arm, seed)

    def train_to(self, arm, seed, target):
        while self.stats(arm, seed)["physical_samples"] < target:
            current = self.stats(arm, seed)
            # A round is at most 9,600 physical samples; use that upper bound so
            # the common endpoint cannot overshoot by several entire rollouts.
            count = min(5, max(1, math.ceil((target-current["physical_samples"]) / MAX_BATCH_SAMPLES)))
            endpoint = current["last_iteration"] + count
            estimate = 45 + count*(max(60., current["recent_iteration_seconds"])*1.5+5)
            args = ["--resume", self.checkpoint(arm, seed), "--seed", 100000+seed+current["last_iteration"],
                    "--max-iterations", endpoint, "--torch-threads", 4,
                    "--run-dir", self.run_name(arm, seed), "--persist-game", "--save-rollout-inputs"]
            self.call("src.training.main_ppo", args, f"{arm}_s{seed}_train_{endpoint:04d}", estimate)
            if self.stats(arm, seed)["physical_samples"] <= current["physical_samples"]:
                raise RuntimeError("Training made no physical-sample progress")
            self.checkpoint(arm, seed)

    def evaluate(self, arm, seed, stage, rounds, evaluation_seed):
        base = ROOT / "runs" / self.run_name(arm, seed) / "checkpoints"
        directory = base / f"eval_{stage}"
        if directory.exists():
            raise RuntimeError(f"Refusing to overwrite evaluation: {directory}")
        checkpoint = base / "rollouts/behavior_0001.pt" if stage == "initial" else self.checkpoint(arm, seed)
        args = ["--checkpoint", checkpoint, "--output", directory, "--steps", 1200,
                "--rounds", rounds, "--num-agents", 8, "--seed", evaluation_seed,
                "--jitter", .01, "--goal-progress", 5, "--dwell", 10, "--noise-phi", .85,
                "--native-diagnostics"]
        label = f"{arm}_s{seed}_{stage}"
        self.call("src.training.local_evaluation", args, label, 90+rounds*150)
        summary = evaluation_summary(directory)
        if summary["failure"] or any(summary["summary"].get(mode, {}).get("episodes") != 8*rounds
                                     for mode in ("deterministic", "exploration")):
            raise RuntimeError("Incomplete evaluation")
        summary["wall_seconds"] = self.job_seconds[label]
        write_json(self.campaign / f"{label}.json", summary)
        self.event("evaluation", arm=arm, seed=seed, stage=stage, summary=summary["summary"])


def report(runner, status, error=None):
    runs = []
    for arm, seed in ORDER:
        evaluations = [json.loads(p.read_text(encoding="utf-8"))
                       for p in sorted(runner.campaign.glob(f"{arm}_s{seed}_*.json"))]
        runs.append({"arm": arm, "seed": seed, **runner.stats(arm, seed), "evaluations": evaluations})
    output = {"status": status, "updated_at": now(), "error": error, "runs": runs,
              "training_physical_samples": sum(r["physical_samples"] for r in runs),
              "evaluation_physical_samples": sum(e["physical_samples"] for r in runs for e in r["evaluations"])}
    write_json(runner.campaign / "summary.json", output)
    lines = ["# 六小时正式训练：8 Players，C/T 三 seed 对照", "",
             f"状态：**{status}**；更新时间：{now()}；硬截止：{runner.deadline.isoformat()}。", "",
             "共同配置：8 Players 串行，batch 128，lr=3e-5，B1 Transformer，归一化动作历史，PPO+OU。",
             "C：gamma=.99、lambda=.95；T：gamma=.998、lambda=.995。训练 seeds 31/47/61；其余参数相同。", "",
             f"已记录训练物理样本 {output['training_physical_samples']:,}（含两轮 critic 预热），已完成评估物理样本 {output['evaluation_physical_samples']:,}。", "",
             "全部从同一历史源权重初始化，不继承短测权重。中期 96k；共同终点由时间预算决定，不依评估成绩挑选。最多允许一个 9,600 样本 rollout 的超额。", "",
             "初始评估共享一次；中期每 seed 每模式 16 回合；保留集每 seed 每模式 24 回合。确定性与探索分开统计，同轮复制体不是独立训练 seed。缺失评估不当作零成功。", ""]
    for run in runs:
        lines += [f"## {run['arm']} / seed {run['seed']}", "",
                  f"训练样本 {run['physical_samples']:,}；PPO minibatch 更新 {run['ppo_updates']:,}。", ""]
        for evaluation in run["evaluations"]:
            lines += [f"{Path(evaluation['directory']).name}，权重对应训练样本 {evaluation['training_samples_at_checkpoint']:,}：", ""]
            for mode, values in evaluation["summary"].items():
                lines.append(f"- {mode}：保持满 1 秒 {values['success_hold_120_count']}/{values['episodes']}；"
                             f"10 步成功 {values['successes']}/{values['episodes']}；成功后跌落 {values['falls_after_success']}；"
                             f"有效进度中位数 {values['credited_progress_median_m']:.2f} 米。")
            lines.append("")
    lines += ["## 判据与执行边界", "",
              "主要指标：同一连续路线段内累计有效进度与当前投影均前进 5 米，并在走廊内连续保持 120 物理步（1 秒）。另保留原 10 步判据。成功后跌落按落水或从已达最高点回落 3 米判断，不等于返回起点。", "",
              "只有 3 个独立训练 seed；不得把同轮复制体当成独立复验。相对旧实验同时改变了学习率、batch 和并行数，不能把跨轮改善单独归因给其中一项。", "",
              "STOP 文件在下一轮控制检查时停止当前自有子进程树；每个已完成训练迭代保存 checkpoint。硬截止或故障不自动加时、不放宽概率审计/KL 阈值。", ""]
    if error:
        lines += [f"停止原因：{error}", ""]
    (runner.campaign / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=6)
    args = parser.parse_args()
    if not 0 < args.hours <= 6:
        parser.error("Budget must be in (0, 6] hours")
    campaign = args.output.resolve()
    if campaign.parent != ROOT / "runs":
        parser.error("Output must be a fresh direct child of runs/")
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running")
    for arm, seed in ORDER:
        if (ROOT / "runs" / f"{campaign.name}_{arm}_s{seed}").exists():
            raise RuntimeError("A planned training directory already exists")
    campaign.mkdir(exist_ok=False)
    (campaign / "logs").mkdir()
    started = datetime.now().astimezone()
    deadline = started+timedelta(hours=args.hours)
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8"))
    game_root = Path(project["game"]["executable_path"]).parent
    runtime = game_root / "GoiData/runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    if original is not None:
        (campaign / "runtime_original.json").write_bytes(original)
    paths = [p for folder in ("src/training", "src/env", "src/start") for p in (ROOT/folder).rglob("*.py")]
    paths += [Path(__file__), ROOT/"src/tests/analysis/run_corridor_screen.py", ROOT/SOURCE,
              ROOT/"src/tests/control_interaction/test_l7_surface_airdrop.py", ROOT/"src/config/project.json",
              ROOT/"src/Data/video_reference/37258923076-1-192/reference_trajectory.json",
              game_root/"BepInEx/plugins/GameRuntime_v2.dll"]
    hashes = [{"path": os.path.relpath(p, ROOT), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    write_json(campaign/"source_hashes.json", hashes)
    manifest = {"status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
                "budget_hours": args.hours, "parameters": PARAMETERS, "order": ORDER, "initial_checkpoint": SOURCE,
                "agents": 8, "batch_size": 128, "lr": .00003, "torch_threads": 4, "mode": "serial",
                "midpoint": MIDPOINT, "endpoint_candidates": [384000, 288000, 240000, 192000, 144000, MIDPOINT],
                "training_seeds": SEEDS, "midpoint_evaluation_seed": 20260922, "holdout_seed": 20261022,
                "evaluation_rounds": {"shared_initial": 2, "midpoint": 2, "holdout": 3},
                "controller_pid": os.getpid(), "controller_create_time": psutil.Process().create_time()}
    write_json(campaign/"manifest.json", manifest)
    runner = FormalRunner(deadline, campaign)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        report(runner, "running")
        for arm, seed in ORDER:
            runner.preflight(arm, seed)
            if (arm, seed) == ORDER[0]:
                runner.evaluate(arm, seed, "initial", 2, 20260922)
            runner.train_to(arm, seed, MIDPOINT)
            runner.evaluate(arm, seed, "midpoint_096k", 2, 20260922)
            report(runner, "running")
        mid_evals = [json.loads(p.read_text(encoding="utf-8")) for p in campaign.glob("*_midpoint_096k.json")]
        final_eval_seconds = max(540., max(e["wall_seconds"] for e in mid_evals)*2.)
        target, estimates = choose_endpoint([runner.stats(a, s) for a, s in ORDER],
                                            (deadline-datetime.now().astimezone()).total_seconds(), final_eval_seconds)
        write_json(campaign/"budget_decision.json", {"decided_at": now(), "target_samples": target,
                   "estimates": estimates, "per_holdout_seconds": final_eval_seconds,
                   "basis": "timing only; all six runs retained; holdout not yet evaluated"})
        for arm, seed in ORDER:
            runner.train_to(arm, seed, target)
            runner.evaluate(arm, seed, f"holdout_{target//1000:03d}k", 3, 20261022)
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
        write_json(campaign/"manifest.json", manifest)
        runner.status(status, error=error, runtime_restored=restored)
        report(runner, status, error)
        runner.event("finished", status=status, error=error, runtime_restored=restored)


if __name__ == "__main__":
    main()
