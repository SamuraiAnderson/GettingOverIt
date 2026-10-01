"""96k 小预算双臂 A/B：纯 C vs C + 防线化 corridor-SIL（门控/限幅/预算）。

背景：corridor_sil_20260923 的无防线 SIL 在 96k 处把 det 中位数打到 0.004/0.000m
（塌陷机制：门控退化 → 均匀模仿含 OU 噪声的饱和种子 + SIL 梯度预算压倒 PPO）。
本 A/B 在相同种子/样本终点/评估口径下比较：
  - C 臂：纯 PPO（--corridor-sil 不加，走廊模式默认停用 SIL），连续训练；
  - S 臂：C + 防线化 SIL（池/当前行为相对门控、sil_target_clamp=3.0 限幅、
    sil_budget_ratio=1.0 预算从属于 PPO），同一 sil_seed_590.npz 冷启动。
两臂均与正式 96k 锚点（corridor_formal_20260922）并列报告；S 臂额外报告门控
开火/跳过轨迹。S 臂门控可能大部分轮次关闭——"不劣化"本身是一等结果。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psutil

from .run_corridor_formal import FormalRunner
from .run_corridor_screen import ROOT, SOURCE, BudgetStop, now, write_json

SEEDS = (31, 47)
ARMS = ("C", "S")  # C = 纯 PPO；S = C + 防线化 SIL
BASELINE = ROOT / "runs/corridor_formal_20260922"
SIL_SEED = ROOT / "runs/best_motion_20260923/sil_seed_590.npz"


def training_arguments(campaign, arm, seed):
    args = ["--init-from", SOURCE, "--reference-trajectory",
            "--reference-corridor-radius", 1.5, "--reference-deviation-weight", .01,
            "--geometry-profile", "native-pose-v1", "--no-random-deploy", "--recipe", "ppo-ou",
            "--seed", seed, "--num-agents", 8, "--steps-per-rollout", 1200,
            "--max-iterations", 20,
            "--value-warmup-iters", 2, "--lr", .00003, "--ent-coef", .001,
            "--ppo-batch-size", 128, "--save-interval", 1, "--torch-threads", 4,
            "--features", "none", "--policy-backbone", "transformer", "--normalize-action-history",
            "--event-source", "native", "--no-event-branch", "--action-segment-steps", 1,
            "--run-dir", f"{campaign.name}_{arm}_s{seed}", "--persist-game", "--save-rollout-inputs",
            "--gamma", .99, "--gae-lambda", .95]
    if arm == "S":
        args += ["--corridor-sil", "--sil-seed", SIL_SEED]
    args += ["--target-physical-samples", 96000]
    return list(map(str, args))


def configuration_reference(seed):
    """New seeds share the formal hyperparameters, not its evaluation scores."""
    path = ROOT / f"runs/corridor_formal_20260922_C_s{seed}/checkpoints/eval_midpoint_096k/manifest.json"
    if not path.exists():
        path = ROOT / "runs/corridor_formal_20260922_C_s31/checkpoints/eval_midpoint_096k/manifest.json"
    return path


def validate_configuration(config, seed, arm):
    path = configuration_reference(seed)
    baseline = json.loads(path.read_text(encoding="utf-8"))["config"]
    permitted = {"max_iterations", "checkpoint_dir", "log_dir"}
    if arm == "S":
        permitted |= {"sil_enabled", "sil_advantage_weighting_enabled"}
    differences = {key: {"baseline": value, "current": config.get(key)}
                   for key, value in baseline.items() if config.get(key) != value}
    unexpected = set(differences) - permitted
    if unexpected:
        raise ValueError(f"Unexpected formal configuration changes: {unexpected}")
    if arm == "S" and not (config["sil_enabled"] and config["sil_advantage_weighting_enabled"]):
        raise ValueError("Corridor SIL was not enabled")
    if arm == "C" and config.get("sil_enabled"):
        raise ValueError("Pure-C arm must keep SIL disabled")
    return differences


class GuardABRunner(FormalRunner):
    def __init__(self, deadline, campaign, seeds=SEEDS):
        self.seeds = tuple(seeds)
        super().__init__(deadline, campaign)

    def run_name(self, arm, seed):
        if arm not in ARMS or seed not in self.seeds:
            raise ValueError("Only arms C/S and configured training seeds are authorized")
        return f"{self.campaign.name}_{arm}_s{seed}"

    def status(self, status="running", **extra):
        write_json(self.campaign / "status.json", {
            "status": status, "updated_at": now(), "deadline": self.deadline.isoformat(),
            "controller_pid": self.pid, "controller_create_time": self.created,
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "job": self.job,
            "training": [{"arm": a, "seed": s, **self.stats(a, s)} for a in ARMS for s in self.seeds
                         if (ROOT / "runs" / self.run_name(a, s)).exists()],
            **extra,
        })

    def train(self, arm, seed):
        self.call("src.training.main_ppo", training_arguments(self.campaign, arm, seed),
                  f"{arm}_s{seed}_train", 1500)
        stats = self.stats(arm, seed)
        if not 96000 <= stats["physical_samples"] < 105600:
            raise RuntimeError("Training did not reach the agreed sample endpoint")
        if not stats["ppo_updates"] or stats["max_frozen_ratio_error"] > .001:
            raise RuntimeError("Missing PPO updates or failed frozen-policy audit")
        self.checkpoint(arm, seed)
        base = ROOT / "runs" / self.run_name(arm, seed)
        meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
        differences = validate_configuration(meta["config"], seed, arm)
        with (base / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        audit = {"configuration_reference": str(configuration_reference(seed)),
                 "configuration_differences": differences, **stats}
        if arm == "S":
            eligible = [r for r in rows if int(r["iteration"]) >= 6
                        and not float(r.get("value_only") or 0)]
            fired = [r for r in eligible if float(r.get("sil_updates") or 0) > 0]
            skipped = [r for r in eligible if float(r.get("sil_skipped") or 0) > 0]
            if not eligible or len(fired) + len(skipped) != len(eligible):
                raise RuntimeError("SIL-eligible round missing both updates and skip marker")
            if any(not np.isfinite(float(r.get("sil_loss") or "nan"))
                   or not r.get("sil_adv_pos_frac") or not r.get("sil_budget_cap")
                   or not np.isfinite(float(r.get("sil_value_loss") or "nan"))
                   for r in fired):
                raise RuntimeError("Fired SIL round missing finite loss / pos_frac / budget cap / value loss")
            audit.update({
                "sil_eligible_iterations": len(eligible),
                "sil_fired_iterations": len(fired),
                "sil_skipped_iterations": len(skipped),
                "sil_updates": sum(float(r["sil_updates"]) for r in fired),
                "sil_pos_frac_by_iteration": {r["iteration"]: float(r["sil_adv_pos_frac"])
                                              for r in eligible if r.get("sil_adv_pos_frac")},
                "sil_gate_levels_by_iteration": {
                    r["iteration"]: [float(r["sil_pool_level"]), float(r["sil_behavior_level"])]
                    for r in eligible if r.get("sil_pool_level")},
                "sil_value_loss_last": float(fired[-1]["sil_value_loss"]) if fired else None,
                "late_action_saturation": float(np.mean(
                    [float(r["action_saturation_fraction"]) for r in rows[-3:]])),
                "entropy_first_last": [float(rows[2]["entropy"]), float(rows[-1]["entropy"])],
            })
        elif any(float(r.get("sil_updates") or 0) > 0 for r in rows):
            raise RuntimeError("Pure-C arm produced SIL updates")
        write_json(self.campaign / f"{arm}_s{seed}_training_audit.json", audit)


def _median_line(result):
    from src.training.local_evaluation import format_eval_line
    det = result["summary"]["deterministic"]
    exp = result["summary"]["exploration"]
    return (f"实际样本 {result['training_samples_at_checkpoint']:,}；"
            f"det {format_eval_line(det)}；"
            f"exploration {format_eval_line(exp)}")


def comparison(runner, status, error=None):
    runs = []
    for seed in runner.seeds:
        row = {"seed": seed}
        for arm in ARMS:
            path = runner.campaign / f"{arm}_s{seed}_midpoint_096k.json"
            row[arm] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            audit_path = runner.campaign / f"{arm}_s{seed}_training_audit.json"
            row[f"{arm}_training"] = (json.loads(audit_path.read_text(encoding="utf-8"))
                                      if audit_path.exists() else runner.stats(arm, seed))
        baseline_path = BASELINE / f"C_s{seed}_midpoint_096k.json"
        row["baseline"] = (json.loads(baseline_path.read_text(encoding="utf-8"))
                           if baseline_path.exists() else None)
        runs.append(row)
    pooled = {}
    for group in ("C", "S", "baseline"):
        if any(r[group] is None for r in runs):
            continue
        pooled[group] = {}
        for mode in ("deterministic", "exploration"):
            episodes = [e for r in runs for e in r[group]["episodes"] if e["mode"] == mode]
            pooled[group][mode] = {
                "episodes": len(episodes),
                "credited_progress_median_m": float(np.median(
                    [e["credited_progress_m"] for e in episodes])),
                "success_hold_120_count": sum(e["success_hold_120"] for e in episodes),
                "successes": sum(e["success"] for e in episodes),
            }
    write_json(runner.campaign / "comparison.json", {
        "status": status, "updated_at": now(), "error": error, "runs": runs,
        "pooled_episodes": pooled,
        "training_seeds": runner.seeds,
        "caveats": [f"Only {len(runner.seeds)} independent training seeds; parallel agents are not independent seeds.",
                    "The SIL seed was selected from a historical T midpoint evaluation; this is a seeded transfer comparison, not a new untouched holdout.",
                    "Both arms are fresh continuous runs; the formal anchors used process restarts and reseeding.",
                    "Historical anchors are shown only when the same training seed exists; configuration templates do not supply evaluation results.",
                    "The S arm gate may skip SIL in most rounds; 'no degradation' is a first-class outcome."],
    })
    lines = ["# 纯 C vs C + 防线化 corridor SIL（96k A/B）", "",
             f"状态：{status}；更新：{now()}。", "",
             "8 agents × 1200 步，batch 128，lr 3e-5，gamma .99 / lambda .95，native-pose-v1，固定起点。",
             "两臂同初始化、同 96k 样本终点、同评估（seed 20260922，2 rounds × 8 agents × det/exploration）。",
             "S 臂 = C + 防线化 SIL（池/当前行为相对门控，margin=0 / raw 目标限幅 ±3 / 预算 ≤ 当轮 PPO 更新数），"
             "种子 sil_seed_590.npz；C 臂纯 PPO。正式 96k 锚点并列参考。", ""]
    for row in runs:
        lines += [f"## Seed {row['seed']}", ""]
        for arm, label in (("C", "本次纯 C"), ("S", "本次 C + 防线 SIL")):
            result = row[arm]
            if result is None:
                lines.append(f"- {label}：尚未完成评估。")
                continue
            lines.append(f"- {label}：{_median_line(result)}。")
        if row["baseline"] is not None:
            lines.append(f"- 正式 C 96k 锚点：{_median_line(row['baseline'])}。")
        else:
            lines.append("- 此 seed 无正式 C 历史锚点；按本次 C/S 配对比较。")
        audit = row.get("S_training") or {}
        if "sil_eligible_iterations" in audit:
            lines.append(
                f"- S 臂门控：可开火 {audit['sil_eligible_iterations']} 轮，"
                f"实际开火 {audit['sil_fired_iterations']} 轮、跳过 {audit['sil_skipped_iterations']} 轮；"
                f"pos_frac 逐轮 {audit['sil_pos_frac_by_iteration']}；"
                f"末三轮饱和率 {audit['late_action_saturation']:.3f}。")
        if row["C"] is not None and row["S"] is not None:
            delta = (row["S"]["summary"]["deterministic"]["credited_progress_median_m"]
                     - row["C"]["summary"]["deterministic"]["credited_progress_median_m"])
            lines.append(f"- 同 seed det 中位数变化（S − C）：{delta:+.3f}m。")
        lines.append("")
    lines += [f"两臂均为新连续训练；正式锚点来自分段重启流程，仅作参考。本批共 {len(runner.seeds)} 个独立训练 seed，"
              "不把复制体回合当独立复验。S 臂门控若大量跳过，则本 A/B 主要衡量安全性而非增益。", "",
              "在本目录创建 STOP 文件可终止自有进程树；每轮已完成 checkpoint 保留。"
              "结束时恢复启动前游戏运行配置。"]
    if error:
        lines += ["", f"停止原因：{error}"]
    (runner.campaign / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3., help="Safety ceiling, not a training allocation")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Distinct training seeds; each runs both C and S (default: 31 47)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= len(args.seeds) <= 6 or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= seed < 2**32 for seed in args.seeds)):
        parser.error("Use one to six distinct training seeds in [0, 2**32)")
    max_hours = min(6., max(3., 1.5 * len(args.seeds)))
    if not 0 < args.hours <= max_hours:
        parser.error(f"At most {max_hours:g} hours for {len(args.seeds)} seeds")
    if args.output.resolve().parent != ROOT / "runs":
        parser.error("Use a fresh direct child of runs")
    return args


def main():
    args = parse_args()
    seeds = tuple(args.seeds)
    campaign = args.output.resolve()
    commands = [training_arguments(campaign, a, s) for s in seeds for a in ARMS]
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take over")
    if any((ROOT / "runs" / f"{campaign.name}_{a}_s{s}").exists() for s in seeds for a in ARMS):
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
    paths += sorted({configuration_reference(seed) for seed in seeds})
    write_json(campaign / "source_hashes.json", [
        {"path": os.path.relpath(p, ROOT), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=args.hours)
    manifest = {"status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
                "target_physical_samples_per_seed": 96000, "max_iterations": 20,
                "arms": ARMS, "training_seeds": seeds, "budget_hours": args.hours,
                "configuration_references": {str(s): str(configuration_reference(s)) for s in seeds},
                "initial_checkpoint": SOURCE, "sil_seed": str(SIL_SEED), "baseline": str(BASELINE),
                "sil_guards": {"sil_gate_margin": 0., "sil_target_clamp": 3., "sil_budget_ratio": 1.,
                               "sil_value_coef": .1},
                "sigma_schedule": {"ppo_reset_log_std_on_init": False, "ppo_init_log_std": -1.6,
                                   "ppo_log_std_anneal_rate": 0., "ppo_log_std_min": -2.5},
                "warmup_included": True, "sil_first_iteration": 6, "evaluation_seed": 20260922,
                "evaluation_rounds_per_mode": 2, "controller_pid": os.getpid(),
                "controller_create_time": psutil.Process().create_time(),
                "commands": commands}
    write_json(campaign / "manifest.json", manifest)
    runner = GuardABRunner(deadline, campaign, seeds)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        comparison(runner, "running")
        for seed in seeds:
            for arm in ARMS:
                runner.train(arm, seed)
                runner.evaluate(arm, seed, "midpoint_096k", 2, 20260922)
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
