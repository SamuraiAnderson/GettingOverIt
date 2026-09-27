"""Long unattended SIL training for seed 47 with endpoint (t, D) Pareto archive.

Fresh pot-center start from the p2_contact checkpoint. Does not resume the
20260926 player-origin run or load sil_seed_590. Isolated-desktop launch stays
the GameLauncher default.

    conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.run_corridor_sil_td_pareto --output runs/corridor_sil_td_pareto_20260927b --hours 12
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import psutil

from .run_corridor_formal import FormalRunner, MIDPOINT, MAX_BATCH_SAMPLES
from .run_corridor_screen import ROOT, SOURCE, BudgetStop, now, write_json

SEED = 47
SEEDS = (SEED,)
ARM = "S"
ARCHIVE_MODE = "td_pareto"
BASELINE = ROOT / "runs/corridor_formal_20260922"
MAP_ELITES_096K = ROOT / "runs/corridor_sil_formal_20260926/S_s47_midpoint_096k.json"
SIL_SEED = ROOT / "runs/best_motion_20260923/sil_seed_590.npz"
STEP = MIDPOINT
MAX_HOURS = 16
EVAL_ROUNDS = 2
EVAL_SEED = 20260922


def training_arguments(campaign, seed):
    if seed != SEED:
        raise ValueError("Only seed 47 is authorized")
    return list(map(str, [
        "--init-from", SOURCE, "--reference-trajectory",
        "--reference-corridor-radius", 1.5, "--reference-deviation-weight", .01,
        "--geometry-profile", "native-pose-v1", "--no-random-deploy", "--recipe", "ppo-ou",
        "--seed", seed, "--num-agents", 8, "--steps-per-rollout", 1200,
        "--max-iterations", 20, "--value-warmup-iters", 2, "--lr", .00003, "--ent-coef", .001,
        "--ppo-batch-size", 128, "--save-interval", 1, "--torch-threads", 4,
        "--features", "none", "--policy-backbone", "transformer", "--normalize-action-history",
        "--event-source", "native", "--no-event-branch", "--action-segment-steps", 1,
        "--sparse-geometry",
        "--run-dir", f"{campaign.name}_{ARM}_s{seed}", "--persist-game", "--save-rollout-inputs",
        "--gamma", .99, "--gae-lambda", .95, "--corridor-sil",
        "--sil-archive-mode", ARCHIVE_MODE, "--target-physical-samples", MIDPOINT,
    ]))


def resume_arguments(campaign, seed, current, target):
    if seed != SEED:
        raise ValueError("Only seed 47 is authorized")
    remaining = target - current["physical_samples"]
    count = min(5, max(1, math.ceil(remaining / MAX_BATCH_SAMPLES)))
    endpoint = current["last_iteration"] + count
    base = ROOT / "runs" / f"{campaign.name}_{ARM}_s{seed}"
    return list(map(str, [
        "--resume", base / "checkpoints" / f"ppo_iter_{current['last_iteration']:04d}.pt",
        "--sil-resume-pool", base / "elites" / f"elites_iter_{current['last_iteration']:04d}.npz",
        "--corridor-sil", "--sil-archive-mode", ARCHIVE_MODE,
        "--seed", 100000 + seed + current["last_iteration"],
        "--max-iterations", endpoint, "--additional-physical-samples", remaining,
        "--torch-threads", 4, "--save-interval", 1,
        "--run-dir", f"{campaign.name}_{ARM}_s{seed}", "--persist-game", "--save-rollout-inputs",
    ])), count


def configuration_reference(seed):
    path = ROOT / f"runs/corridor_formal_20260922_C_s{seed}/checkpoints/eval_midpoint_096k/manifest.json"
    if not path.exists():
        path = ROOT / "runs/corridor_formal_20260922_C_s47/checkpoints/eval_midpoint_096k/manifest.json"
    return path


def validate_configuration(config, seed):
    baseline = json.loads(configuration_reference(seed).read_text(encoding="utf-8"))["config"]
    permitted = {
        "sil_enabled", "sil_advantage_weighting_enabled", "max_iterations",
        "checkpoint_dir", "log_dir", "sil_archive_mode", "sil_td_bucket_m",
        "sil_td_bucket_enable_at", "sil_td_behavior_slots", "sil_td_windows_per_traj",
        "reference_trajectory", "sparse_geometry", "sparse_window_cells",
        "sparse_window_meters", "sparse_max_contour_tokens", "sparse_relation_k",
        "sparse_mlp_hidden",
    }
    differences = {key: {"baseline": value, "current": config.get(key)}
                   for key, value in baseline.items() if config.get(key) != value}
    unexpected = set(differences) - permitted
    if unexpected:
        raise ValueError(f"Unexpected formal configuration changes: {unexpected}")
    if Path(str(config.get("reference_trajectory") or "")).name != "reference_trajectory_game_world.json":
        raise ValueError("Pot-center SIL must use reference_trajectory_game_world.json")
    if not config.get("sil_enabled") or not config.get("sil_advantage_weighting_enabled"):
        raise ValueError("Corridor SIL was not enabled")
    if not config.get("sparse_geometry"):
        raise ValueError("Sparse geometry model was not enabled")
    if config.get("sil_archive_mode") != ARCHIVE_MODE:
        raise ValueError("Long SIL training must use td_pareto")
    if config.get("ppo_reset_log_std_on_init"):
        raise ValueError("Formal SIL must inherit checkpoint σ")
    if float(config.get("ppo_log_std_anneal_rate") or 0) != 0:
        raise ValueError("Formal SIL must not force-anneal σ")
    if abs(float(config.get("ou_phi", 0)) - .85) > 1e-9:
        raise ValueError("ou_phi must stay 0.85")
    return differences


def next_milestone(samples):
    return (int(samples) // STEP + 1) * STEP


def choose_next_target(current, seconds_left, evaluation_seconds):
    """Use timing only; never inspect an evaluation score to allocate training."""
    target = next_milestone(current["physical_samples"])
    remaining = max(0, target - current["physical_samples"])
    count = math.ceil(remaining / max(1, current["recent_samples_per_iteration"]))
    estimate = 180. + evaluation_seconds
    estimate += count * (max(60., current["recent_iteration_seconds"]) * 1.5 + 5.)
    estimate += math.ceil(max(1, count) / 5) * 45.
    if estimate < seconds_left:
        return target, estimate
    raise BudgetStop("Insufficient time for the next 96k milestone and evaluation")


def _eval_line(result):
    from src.training.local_evaluation import format_eval_line
    det = result["summary"]["deterministic"]
    exp = result["summary"]["exploration"]
    return (f"实际样本 {result['training_samples_at_checkpoint']:,}；"
            f"det {format_eval_line(det)}；exploration {format_eval_line(exp)}")


def _sil_rows(base):
    with (base / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    eligible = [r for r in rows if int(r["iteration"]) >= 6 and not float(r.get("value_only") or 0)]
    fired = [r for r in eligible if float(r.get("sil_updates") or 0) > 0]
    skipped = [r for r in eligible if float(r.get("sil_skipped") or 0) > 0]
    return rows, eligible, fired, skipped


class SILParetoRunner(FormalRunner):
    def __init__(self, deadline, campaign):
        super().__init__(deadline, campaign)
        self.flags = {ARM: {"gamma": .99, "gae_lambda": .95}}

    def run_name(self, arm, seed):
        if arm != ARM or seed != SEED:
            raise ValueError("Only the SIL arm and seed 47 are authorized")
        return f"{self.campaign.name}_{ARM}_s{seed}"

    def status(self, status="running", **extra):
        write_json(self.campaign / "status.json", {
            "status": status, "updated_at": now(), "deadline": self.deadline.isoformat(),
            "controller_pid": self.pid, "controller_create_time": self.created,
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "job": self.job,
            "training": [{"arm": ARM, "seed": SEED, **self.stats(ARM, SEED)}]
            if (ROOT / "runs" / self.run_name(ARM, SEED)).exists() else [],
            **extra,
        })

    def write_training_audit(self, seed):
        base = ROOT / "runs" / self.run_name(ARM, seed)
        meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
        differences = validate_configuration(meta["config"], seed)
        _rows, eligible, fired, skipped = _sil_rows(base)
        if not eligible or len(fired) + len(skipped) != len(eligible):
            raise RuntimeError("SIL-eligible round missing both updates and skip marker")
        if any(not np.isfinite(float(r.get("sil_loss") or "nan"))
               or not r.get("sil_adv_pos_frac") or not r.get("sil_budget_cap")
               or not np.isfinite(float(r.get("sil_value_loss") or "nan"))
               for r in fired):
            raise RuntimeError("Fired SIL round missing finite loss / pos_frac / budget cap / value loss")
        if fired and "sil_archive_size" in fired[0]:
            if any(float(r.get("sil_archive_size") or 0) < 1 for r in fired):
                raise RuntimeError("td_pareto archive emptied")
        last = fired[-1] if fired else {}
        write_json(self.campaign / f"{ARM}_s{seed}_training_audit.json", {
            "configuration_reference": str(configuration_reference(seed)),
            "configuration_differences": differences,
            "sil_archive_mode": meta["config"].get("sil_archive_mode"),
            "sil_eligible_iterations": len(eligible),
            "sil_fired_iterations": len(fired),
            "sil_skipped_iterations": len(skipped),
            "sil_updates": sum(float(r["sil_updates"]) for r in fired),
            "sil_pos_frac_by_iteration": {r["iteration"]: float(r["sil_adv_pos_frac"])
                                          for r in eligible if r.get("sil_adv_pos_frac")},
            "sil_archive_last": {
                key: float(last[key]) for key in (
                    "sil_archive_size", "sil_archive_pareto", "sil_archive_behavior",
                    "sil_archive_bucketed", "sil_archive_capped",
                ) if last.get(key) not in (None, "")
            },
            **self.stats(ARM, seed),
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
        self.write_training_audit(seed)

    def train_to(self, seed, target):
        while self.stats(ARM, seed)["physical_samples"] < target:
            current = self.stats(ARM, seed)
            args, count = resume_arguments(self.campaign, seed, current, target)
            pool = Path(args[args.index("--sil-resume-pool") + 1])
            if not pool.exists():
                raise RuntimeError(f"Missing SIL elite pool: {pool}")
            if "--sil-archive-mode" not in args or args[args.index("--sil-archive-mode") + 1] != ARCHIVE_MODE:
                raise RuntimeError("Resume lost td_pareto")
            estimate = 45 + count * (max(60., current["recent_iteration_seconds"]) * 1.5 + 5)
            self.call("src.training.main_ppo", args,
                      f"{ARM}_s{seed}_train_{int(args[args.index('--max-iterations') + 1]):04d}",
                      estimate)
            if self.stats(ARM, seed)["physical_samples"] <= current["physical_samples"]:
                raise RuntimeError("Training made no physical-sample progress")
            self.checkpoint(ARM, seed)
        self.write_training_audit(seed)


def report(runner, status, error=None):
    evaluations = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(runner.campaign.glob(f"{ARM}_s{SEED}_*.json"))
                   if not p.name.endswith("_training_audit.json")]
    baseline_path = BASELINE / f"C_s{SEED}_midpoint_096k.json"
    run = {
        "arm": ARM, "seed": SEED, **runner.stats(ARM, SEED), "evaluations": evaluations,
        "formal_c_096k": (json.loads(baseline_path.read_text(encoding="utf-8"))
                          if baseline_path.exists() else None),
        "map_elites_sil_096k": (json.loads(MAP_ELITES_096K.read_text(encoding="utf-8"))
                                if MAP_ELITES_096K.exists() else None),
    }
    output = {"status": status, "updated_at": now(), "error": error, "runs": [run],
              "training_physical_samples": run["physical_samples"],
              "evaluation_physical_samples": sum(e.get("physical_samples", 0) for e in evaluations)}
    write_json(runner.campaign / "summary.json", output)
    lines = ["# 长时间训练：seed 47，走廊 SIL，(t,D) Pareto 归档", "",
             f"状态：**{status}**；更新时间：{now()}；硬截止：{runner.deadline.isoformat()}。", "",
             "只跑 seed 47，S 臂：C 配方（gamma=.99、lambda=.95）+ 防线化 corridor SIL + `td_pareto`。",
             "不跑 seed 31/61，不跑纯 C，不跑 T，不从 20260926 MAP-Elites 96k 续训。",
             "σ 继承 `p2_contact` checkpoint，不重置、不按轮递减；`actor_log_std` 仍走 PPO 梯度。ou_phi=0.85。",
             "8 Players，batch 128，lr=3e-5，B1 Transformer，归一化动作历史，PPO+OU，走廊半宽 1.5 m。",
             "新模型：32×32 粗图 + 连续轮廓 / 接触关系 token 残差（零初始化投影）。",
             "计分：verified 锅心 + HELLO anchor35 + 游戏世界参考线。不用 sil_seed_590。",
             "游戏默认在隔离桌面 GoiRuntime 启动。", "",
             "归档：终点 (t,D) 非支配前沿，支配含单维相等；相同点去重。不到 16 条不分桶；",
             "超容量先留最快和最远，再稀疏，不按 score 截。另留最多 3 条行为多样性。",
             "采样按轨迹对齐窗口数。门控按候选 D，快短点不会靠中位数关掉远点。", "",
             f"已记录训练物理样本 {output['training_physical_samples']:,}（含两轮 critic 预热），"
             f"已完成评估物理样本 {output['evaluation_physical_samples']:,}。", "",
             "先连续训练到 96k 以保留内存精英池，再按 96k 台阶续训；续训恢复精英池并锁定 td_pareto。",
             "预算决策不读评估成绩。最多允许一个 9,600 样本 rollout 的超额。", ""]
    lines += [f"## SIL / seed {run['seed']}", "",
              f"训练样本 {run['physical_samples']:,}；PPO minibatch 更新 {run['ppo_updates']:,}。", ""]
    if run["formal_c_096k"] is not None:
        lines.append(f"- 正式 C 96k 锚点（跨批参考，无 SIL）：{_eval_line(run['formal_c_096k'])}。")
    if run["map_elites_sil_096k"] is not None:
        lines.append(f"- 20260926 MAP-Elites SIL 96k（跨批参考，同 seed，不同归档）：{_eval_line(run['map_elites_sil_096k'])}。")
    for evaluation in run["evaluations"]:
        stage = Path(evaluation.get("directory", evaluation.get("stage", "eval"))).name
        lines.append(f"- {stage}：{_eval_line(evaluation)}。")
    lines += ["", "## 判据与执行边界", "",
              "到达单独计数。保持用容差、允许继续前进：到达后观察 1 秒，投影不低于 4.7 米、至少 95% 时间在走廊内、窗口末回到 5 米、无落水；0.3 米与 95% 为待验证参数。参考缺口记为无法确认，观察窗不足单列。当前阶段用 credited 中位数和下四分位区分改进。",
              "确定性推进、探索推进和末端保留进度分开看；修改 σ 不会改变冻结模型的均值动作。",
              "只有 1 个训练 seed。正式 C 与 20260926 SIL 锚点来自分段重启，不能把跨批差异单独归因给归档。",
              "STOP 文件在下一轮控制检查时停止当前自有子进程树。硬截止或故障不自动加时。", ""]
    if error:
        lines += [f"停止原因：{error}", ""]
    (runner.campaign / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def _source_paths(game_root):
    paths = [p for folder in ("src/training", "src/env", "src/start") for p in (ROOT / folder).rglob("*.py")]
    paths += [Path(__file__), ROOT / "src/tests/analysis/run_corridor_formal.py",
              ROOT / "src/tests/analysis/run_corridor_screen.py", ROOT / "src/training/local_evaluation.py",
              ROOT / SOURCE, ROOT / "src/config/project.json",
              ROOT / "src/tests/control_interaction/test_l7_surface_airdrop.py",
              ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json",
              game_root / "BepInEx/plugins/GameRuntime_v2.dll"]
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--continue-campaign", action="store_true",
                        help="Resume an existing campaign after a controller-side audit stop")
    args = parser.parse_args(argv)
    if not 0 < args.hours <= MAX_HOURS:
        parser.error(f"Budget must be in (0, {MAX_HOURS}] hours")
    if args.output.resolve().parent != ROOT / "runs":
        parser.error("Output must be a direct child of runs/")
    return args


def main(argv=None):
    args = parse_args(argv)
    campaign = args.output.resolve()
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take it over")
    training_dir = ROOT / "runs" / f"{campaign.name}_{ARM}_s{SEED}"
    if not (ROOT / SOURCE).exists():
        raise RuntimeError("Missing p2_contact checkpoint")
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8"))
    game_root = Path(project["game"]["executable_path"]).parent
    runtime = game_root / "GoiData/runtime_config.json"
    if args.continue_campaign:
        if not campaign.is_dir() or not training_dir.is_dir():
            raise RuntimeError("Continue requires the existing campaign and training directories")
        if (campaign / "STOP").exists():
            (campaign / "STOP").unlink()
        previous = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
        started = datetime.fromisoformat(previous["started_at"])
        stored_deadline = datetime.fromisoformat(previous["deadline"])
        clock = datetime.now().astimezone()
        deadline = stored_deadline if stored_deadline > clock else clock + timedelta(hours=args.hours)
        original = ((campaign / "runtime_original.json").read_bytes()
                    if (campaign / "runtime_original.json").exists() else None)
        manifest = previous
        manifest.update(status="running", continued_at=now(), error=None,
                        deadline=deadline.isoformat(),
                        controller_pid=os.getpid(),
                        controller_create_time=psutil.Process().create_time())
        if deadline != stored_deadline:
            manifest["budget_hours"] = args.hours
            manifest["deadline_extended_at"] = now()
    else:
        if training_dir.exists():
            raise RuntimeError("A planned training directory already exists")
        campaign.mkdir(exist_ok=False)
        (campaign / "logs").mkdir()
        started = datetime.now().astimezone()
        deadline = started + timedelta(hours=args.hours)
        original = runtime.read_bytes() if runtime.exists() else None
        if original is not None:
            (campaign / "runtime_original.json").write_bytes(original)
        manifest = {"status": "running", "started_at": started.isoformat(), "deadline": deadline.isoformat(),
                    "budget_hours": args.hours, "arms": [ARM], "training_seeds": SEEDS,
                    "initial_checkpoint": SOURCE, "sil_seed": None, "baseline": str(BASELINE),
                    "reference_anchor_id": "pot_visible_center_v1",
                    "tcp_protocol": "anchor35",
                    "reference_trajectory": "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json",
                    "sil_archive_mode": ARCHIVE_MODE, "resume_from": None,
                    "isolated_desktop": "GoiRuntime",
                    "agents": 8, "batch_size": 128, "lr": .00003, "torch_threads": 4, "mode": "serial",
                    "midpoint": MIDPOINT, "milestone_step": STEP,
                    "sigma_schedule": {"ppo_reset_log_std_on_init": False, "ppo_init_log_std": -1.6,
                                       "ppo_log_std_anneal_rate": 0., "ppo_log_std_min": -2.5,
                                       "ou_phi": .85},
                    "sil_guards": {"sil_gate_margin": 0., "sil_target_clamp": 3., "sil_budget_ratio": 1.,
                                   "sil_value_coef": .1},
                    "sil_td": {"bucket_m": .10, "bucket_enable_at": 16, "behavior_slots": 3,
                               "windows_per_traj": 0},
                    "midpoint_evaluation_seed": EVAL_SEED, "milestone_evaluation_seed": EVAL_SEED,
                    "evaluation_rounds": {"midpoint": EVAL_ROUNDS, "milestone": EVAL_ROUNDS},
                    "controller_pid": os.getpid(), "controller_create_time": psutil.Process().create_time(),
                    "commands": [training_arguments(campaign, SEED)]}
    write_json(campaign / "source_hashes.json", [
        {"path": os.path.relpath(p, ROOT), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in _source_paths(game_root)])
    write_json(campaign / "manifest.json", manifest)
    runner = SILParetoRunner(deadline, campaign)
    status, error = "completed", None
    try:
        runner.guard()
        runner.status()
        report(runner, "running")
        if runner.stats(ARM, SEED)["physical_samples"] < MIDPOINT:
            runner.train_midpoint(SEED)
        elif not (campaign / f"{ARM}_s{SEED}_training_audit.json").exists():
            runner.write_training_audit(SEED)
        if not (campaign / f"{ARM}_s{SEED}_midpoint_096k.json").exists():
            incomplete = training_dir / "checkpoints" / "eval_midpoint_096k"
            stale_log = campaign / "logs" / f"{ARM}_s{SEED}_midpoint_096k.log"
            if incomplete.exists():
                results = incomplete / "results.json"
                failed = (not results.exists() or
                          bool(json.loads(results.read_text(encoding="utf-8")).get("failure")))
                if failed:
                    shutil.rmtree(incomplete)
            if stale_log.exists() and not (campaign / f"{ARM}_s{SEED}_midpoint_096k.json").exists():
                stale_log.unlink()
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
                "per_eval_seconds": eval_seconds,
                "basis": "timing only; next 96k milestone; evaluation not consulted"})
            runner.train_to(SEED, target)
            if runner.stats(ARM, SEED)["physical_samples"] < target:
                raise RuntimeError("Milestone sample endpoint not reached")
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
