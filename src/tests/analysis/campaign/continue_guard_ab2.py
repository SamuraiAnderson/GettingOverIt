"""续训 corridor_sil_guard_ab2_20260923 全部 4 个 run 各 2 轮，并按同口径评估趋势。

背景：校准门控的 96k A/B 结果分裂（S_s47 +2.28m / S_s31 -2.88m det 塌陷）。
本续训回答趋势问题：S_s31 的塌陷是否持续/可恢复，S_s47 的增益是否保持。
S 臂恢复完整精英池（--sil-resume-pool）与 --corridor-sil；C 臂纯 PPO 续训。
评估沿用 96k 口径：seed 20260922，2 rounds × 8 agents × det/exploration。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import psutil

from .run_corridor_formal import FormalRunner
from .run_corridor_screen import ROOT, BudgetStop, now, write_json, stats_for_run

ORIGIN = ROOT / "runs/corridor_sil_guard_ab2_20260923"
ARMS = ("C", "S")
SEEDS = (31, 47)
EXTRA_ITERS = 2


def origin_run(arm, seed):
    return ROOT / "runs" / f"{ORIGIN.name}_{arm}_s{seed}"


def resume_arguments(campaign, arm, seed, previous):
    iteration = previous["last_iteration"]
    base = origin_run(arm, seed)
    args = ["--resume", base / "checkpoints" / f"ppo_iter_{iteration:04d}.pt",
            "--seed", seed, "--max-iterations", iteration + EXTRA_ITERS,
            "--torch-threads", 4, "--save-interval", 1,
            "--run-dir", f"{campaign.name}_{arm}_s{seed}", "--persist-game", "--save-rollout-inputs"]
    if arm == "S":
        args += ["--sil-resume-pool", base / "elites" / f"elites_iter_{iteration:04d}.npz",
                 "--corridor-sil"]
    return list(map(str, args))


class PlusRunner(FormalRunner):
    def __init__(self, deadline, campaign, previous):
        self.previous = previous
        super().__init__(deadline, campaign)

    def run_name(self, arm, seed):
        if arm not in ARMS or seed not in SEEDS:
            raise ValueError("Only arms C/S and seeds 31/47 are authorized")
        return f"{self.campaign.name}_{arm}_s{seed}"

    def stats(self, arm, seed):
        current = super().stats(arm, seed)
        prev = self.previous[(arm, seed)]
        current["additional_physical_samples"] = current["physical_samples"]
        for name in ("physical_samples", "iterations", "ppo_updates"):
            current[name] += prev[name]
        if not current["last_iteration"]:
            current["last_iteration"] = prev["last_iteration"]
        return current

    def status(self, status="running", **extra):
        write_json(self.campaign / "status.json", {
            "status": status, "updated_at": now(), "deadline": self.deadline.isoformat(),
            "controller_pid": self.pid, "controller_create_time": self.created,
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "job": self.job,
            "training": [{"arm": a, "seed": s, **self.stats(a, s)} for a in ARMS for s in SEEDS
                         if (ROOT / "runs" / self.run_name(a, s)).exists()],
            **extra,
        })

    def train(self, arm, seed):
        prev = self.previous[(arm, seed)]
        self.call("src.training.main_ppo", resume_arguments(self.campaign, arm, seed, prev),
                  f"{arm}_s{seed}_train_plus{EXTRA_ITERS}", 600)
        current = stats_for_run(ROOT / "runs" / self.run_name(arm, seed))
        if current["last_iteration"] != prev["last_iteration"] + EXTRA_ITERS:
            raise RuntimeError(f"{arm}_s{seed} did not complete exactly {EXTRA_ITERS} iterations")
        if current["physical_samples"] <= 0:
            raise RuntimeError(f"{arm}_s{seed} made no sample progress")
        self.checkpoint(arm, seed)


def report(runner, status, error=None):
    rows = []
    lines = [f"# A/B 续训 {EXTRA_ITERS} 轮趋势探针", "", f"状态：{status}；更新：{now()}。", "",
             f"从 {ORIGIN.name} 的 96k 终点恢复（S 臂含精英池），每 run 续训 {EXTRA_ITERS} 轮后按同口径评估。",
             "评估：seed 20260922，2 rounds × 8 agents × det/exploration。续训重设随机种子与 KL 适配器，"
             "不是逐位复刻不间断运行。", ""]
    for seed in SEEDS:
        for arm in ARMS:
            old = json.loads((ORIGIN / f"{arm}_s{seed}_midpoint_096k.json").read_text(encoding="utf-8"))
            path = runner.campaign / f"{arm}_s{seed}_plus{EXTRA_ITERS}.json"
            new = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            rows.append({"arm": arm, "seed": seed, "training": runner.stats(arm, seed),
                         "eval_096k": old, "eval_plus2": new})
            lines += [f"## {arm} / seed {seed}", ""]
            for label, result in ((f"{EXTRA_ITERS} 轮前(96k)", old), (f"续训后(+{EXTRA_ITERS})", new)):
                if result is None:
                    lines.append(f"- {label}：评估尚未完成。")
                    continue
                det = result["summary"]["deterministic"]
                exp = result["summary"]["exploration"]
                from src.training.local_evaluation import format_eval_line
                lines.append(f"- {label}：det {format_eval_line(det)}；"
                             f"exploration {format_eval_line(exp)}。")
            lines.append("")
    if error:
        lines += ["", f"停止原因：{error}"]
    write_json(runner.campaign / "comparison.json",
               {"status": status, "error": error, "updated_at": now(), "runs": rows})
    (runner.campaign / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1.)
    args = parser.parse_args()
    campaign = args.output.resolve()
    if campaign.parent != ROOT / "runs" or not 0 < args.hours <= 2:
        parser.error("Use a fresh direct child of runs and at most two hours")
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take over")
    previous = {}
    for arm in ARMS:
        for seed in SEEDS:
            base = origin_run(arm, seed)
            meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
            if meta["status"] != "completed":
                raise RuntimeError(f"Continuation source not completed: {base}")
            previous[(arm, seed)] = stats_for_run(base)
            if (ROOT / "runs" / f"{campaign.name}_{arm}_s{seed}").exists():
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
              ROOT / "src/config/project.json",
              ROOT / "src/tests/control_interaction/test_l7_surface_airdrop.py",
              ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json",
              game_root / "BepInEx/plugins/GameRuntime_v2.dll"]
    for arm in ARMS:
        for seed in SEEDS:
            cmd_args = resume_arguments(campaign, arm, seed, previous[(arm, seed)])
            paths.append(Path(cmd_args[1]))
            if arm == "S":
                paths.append(Path(cmd_args[cmd_args.index("--sil-resume-pool") + 1]))
    write_json(campaign / "source_hashes.json", [
        {"path": os.path.relpath(p, ROOT), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=args.hours)
    manifest = {"status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
                "origin": str(ORIGIN), "extra_iterations": EXTRA_ITERS,
                "previous": {f"{a}_s{s}": v for (a, s), v in previous.items()},
                "commands": {f"{a}_s{s}": resume_arguments(campaign, a, s, previous[(a, s)])
                             for a in ARMS for s in SEEDS},
                "controller_pid": os.getpid(), "controller_create_time": psutil.Process().create_time()}
    write_json(campaign / "manifest.json", manifest)
    runner = PlusRunner(deadline, campaign, previous)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        report(runner, "running")
        for seed in SEEDS:
            for arm in ARMS:
                runner.train(arm, seed)
                runner.evaluate(arm, seed, f"plus{EXTRA_ITERS}", 2, 20260922)
                # 续训 run 的 metrics.csv 只含新增轮次，评估样本量需并上 origin 部分
                result_path = campaign / f"{arm}_s{seed}_plus{EXTRA_ITERS}.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["additional_training_samples_at_checkpoint"] = result["training_samples_at_checkpoint"]
                result["training_samples_at_checkpoint"] += previous[(arm, seed)]["physical_samples"]
                write_json(result_path, result)
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
