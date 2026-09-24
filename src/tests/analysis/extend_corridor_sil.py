"""Resume both completed C+SIL seeds to 288k cumulative physical samples."""
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psutil

from .run_corridor_sil import SILRunner, SEEDS, validate_configuration
from .run_corridor_screen import ROOT, BudgetStop, now, write_json, stats_for_run

ORIGIN = ROOT / "runs/corridor_sil_20260923"
CAMPAIGN = ROOT / "runs/corridor_sil_extended_20260923"
TARGET = 288000


def origin_run(seed):
    return ROOT / "runs" / f"{ORIGIN.name}_C_s{seed}"


def resume_arguments(seed, previous):
    iteration = previous["last_iteration"]
    base = origin_run(seed)
    return list(map(str, [
        "--resume", base / "checkpoints" / f"ppo_iter_{iteration:04d}.pt",
        "--sil-resume-pool", base / "elites" / f"elites_iter_{iteration:04d}.npz",
        "--corridor-sil", "--seed", seed, "--max-iterations", iteration + 60,
        "--additional-physical-samples", TARGET - previous["physical_samples"],
        "--torch-threads", 4, "--save-interval", 1,
        "--run-dir", f"{CAMPAIGN.name}_C_s{seed}", "--persist-game", "--save-rollout-inputs",
    ]))


class ExtensionRunner(SILRunner):
    def __init__(self, deadline, campaign, previous):
        self.previous = previous
        super().__init__(deadline, campaign)

    def stats(self, arm, seed):
        current = super().stats(arm, seed)
        current["additional_physical_samples"] = current["physical_samples"]
        current["additional_iterations"] = current["iterations"]
        for name in ("physical_samples", "iterations", "warmup_samples", "ppo_updates"):
            current[name] += self.previous[seed][name]
        if not current["last_iteration"]:
            current["last_iteration"] = self.previous[seed]["last_iteration"]
        return current


def report(runner, status, error=None):
    rows = []
    lines = ["# C + SIL 续训至累计 288k", "", f"状态：{status}；更新：{now()}。", "",
             "从上一轮 96k 终点恢复模型、优化器与完整 SIL 精英池，配方保持不变。每个 seed 达到累计 288,000 实际样本后自动评估。", ""]
    for seed in SEEDS:
        old = json.loads((ORIGIN / f"C_s{seed}_midpoint_096k.json").read_text(encoding="utf-8"))
        path = CAMPAIGN / f"C_s{seed}_endpoint_288k.json"
        new = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        current = runner.stats("C", seed)
        rows.append({"seed": seed, "training": current, "sil_096k": old, "sil_288k": new})
        lines += [f"## Seed {seed}", "", f"累计 {current['physical_samples']:,} 样本；本次新增 {current['additional_physical_samples']:,}；累计 {current['iterations']} 轮。", ""]
        for label, result in (("SIL 96k", old), ("SIL 288k", new)):
            if result is None:
                lines.append(f"- {label}：评估尚未完成。")
            else:
                det, exp = (result["summary"][mode] for mode in ("deterministic", "exploration"))
                lines.append(f"- {label}：det 有效进度中位数 {det['credited_progress_median_m']:.3f}m；探索 {exp['credited_progress_median_m']:.3f}m；det 1 秒保持成功 {det['success_hold_120_count']}/{det['episodes']}。")
        lines.append("")
    lines += ["评估沿用中期口径：seed=20260922，2 rounds × 8 agents，det/exploration 分开；原 96k 文件保留。",
              "288k 与 96k 的比较衡量训练延长后的变化，不是相同样本预算下的 SIL 消融。原正式 C 没有 288k 对照。续训重设随机种子，不是逐位复刻不间断运行。",
              "STOP 文件可停止控制器自有子进程；正常完成或停止后恢复游戏运行配置。"]
    if error:
        lines.append(f"停止原因：{error}")
    write_json(CAMPAIGN / "comparison.json", {"status": status, "error": error, "updated_at": now(), "runs": rows})
    (CAMPAIGN / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("Existing game is running; refusing takeover")
    previous = {seed: stats_for_run(origin_run(seed)) for seed in SEEDS}
    for seed in SEEDS:
        meta = json.loads((origin_run(seed) / "run_meta.json").read_text(encoding="utf-8"))
        if meta["status"] != "completed" or previous[seed]["physical_samples"] >= TARGET:
            raise RuntimeError("Invalid continuation source")
        if (ROOT / "runs" / f"{CAMPAIGN.name}_C_s{seed}").exists():
            raise RuntimeError("Refusing overwrite")
    CAMPAIGN.mkdir(exist_ok=False)
    (CAMPAIGN / "logs").mkdir()
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8"))
    game_root = Path(project["game"]["executable_path"]).parent
    runtime = game_root / "GoiData/runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    if original is not None:
        (CAMPAIGN / "runtime_original.json").write_bytes(original)
    paths = [p for folder in ("src/training", "src/env", "src/start") for p in (ROOT / folder).rglob("*.py")]
    paths += [Path(__file__), ROOT / "src/tests/analysis/run_corridor_sil.py",
              ROOT / "src/tests/analysis/run_corridor_formal.py", ROOT / "src/tests/analysis/run_corridor_screen.py",
              ROOT / "src/tests/control_interaction/test_l7_surface_airdrop.py", ROOT / "src/config/project.json",
              ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json",
              game_root / "BepInEx/plugins/GameRuntime_v2.dll"]
    for seed in SEEDS:
        args = resume_arguments(seed, previous[seed])
        paths += [Path(args[1]), Path(args[3])]
    write_json(CAMPAIGN / "source_hashes.json", [
        {"path": os.path.relpath(p, ROOT), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=3)
    manifest = {"status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
                "target_cumulative_samples_per_seed": TARGET, "previous": previous,
                "commands": {seed: resume_arguments(seed, previous[seed]) for seed in SEEDS},
                "controller_pid": os.getpid(), "controller_create_time": psutil.Process().create_time()}
    write_json(CAMPAIGN / "manifest.json", manifest)
    runner = ExtensionRunner(deadline, CAMPAIGN, previous)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        report(runner, "running")
        for seed in SEEDS:
            runner.call("src.training.main_ppo", resume_arguments(seed, previous[seed]), f"C_s{seed}_resume", 2400)
            current = runner.stats("C", seed)
            if not TARGET <= current["physical_samples"] < TARGET + 9600:
                raise RuntimeError("Cumulative sample endpoint not reached")
            if current["max_frozen_ratio_error"] > .001:
                raise RuntimeError("Policy audit failed")
            base = ROOT / "runs" / runner.run_name("C", seed)
            meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
            validate_configuration(meta["config"], seed)
            import csv
            with (base / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                metrics = list(csv.DictReader(stream))
            if any(float(r.get("value_only") or 0) for r in metrics):
                raise RuntimeError("Unexpected repeated critic warmup")
            if not all(
                (float(r.get("sil_updates") or 0) > 0 and np.isfinite(float(r.get("sil_loss") or "nan")))
                or float(r.get("sil_skipped") or 0) > 0  # 门控/预算跳过是合法行为
                for r in metrics
            ):
                raise RuntimeError("SIL updates missing or nonfinite")
            runner.evaluate("C", seed, "endpoint_288k", 2, 20260922)
            result_path = CAMPAIGN / f"C_s{seed}_endpoint_288k.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["additional_training_samples_at_checkpoint"] = result["training_samples_at_checkpoint"]
            result["training_samples_at_checkpoint"] += previous[seed]["physical_samples"]
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
        manifest.update(status=status, error=error, finished_at=now(), runtime_restored=restored)
        write_json(CAMPAIGN / "manifest.json", manifest)
        runner.status(status, error=error, runtime_restored=restored)
        report(runner, status, error)
        runner.event("finished", status=status, error=error, runtime_restored=restored)


if __name__ == "__main__":
    main()
