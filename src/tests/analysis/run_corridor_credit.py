"""Bounded reward/credit-horizon comparison, including automatic endpoint evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .run_corridor_screen import ROOT, SOURCE, Runner, BudgetStop, now, write_json

CAMPAIGN = ROOT / "runs/corridor_credit_20260919"
PARAMETERS = {
    "C": {"gamma": .99, "gae_lambda": .95, "reference_deviation_weight": .01},
    "D": {"gamma": .99, "gae_lambda": .95, "reference_deviation_weight": .001},
    "T": {"gamma": .998, "gae_lambda": .995, "reference_deviation_weight": .01},
}
FLAGS = {
    arm: ["--features", "none", "--policy-backbone", "transformer", "--normalize-action-history",
          *[item for key, value in parameters.items() for item in ("--" + key.replace("_", "-"), str(value))]]
    for arm, parameters in PARAMETERS.items()
}
ORDER = [(arm, seed) for seed in (7, 17) for arm in PARAMETERS]


def report(runner, status, error=None):
    """Use completed evaluation summaries only; missing evaluations stay missing."""
    runs = []
    for arm, seed in ORDER:
        base = ROOT / "runs" / runner.run_name(arm, seed)
        if not base.exists():
            continue
        evaluations = [json.loads(p.read_text(encoding="utf-8"))
                       for p in sorted(CAMPAIGN.glob(f"{arm}_s{seed}_*.json"))]
        runs.append({"arm": arm, "seed": seed, **runner.stats(arm, seed), "evaluations": evaluations})
    output = {"updated_at": now(), "status": status, "error": error, "runs": runs,
              "training_physical_samples": sum(r["physical_samples"] for r in runs),
              "evaluation_physical_samples": sum(e["physical_samples"] for r in runs for e in r["evaluations"])}
    write_json(CAMPAIGN / "summary.json", output)
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text(encoding="utf-8"))
    lines = ["# 奖励与信用分配时间对照", "",
             f"状态：**{status}**；更新：{output['updated_at']}；硬截止：{manifest['deadline']}。", "",
             f"已记录训练物理样本 **{output['training_physical_samples']:,}**，已完成评估物理样本 **{output['evaluation_physical_samples']:,}**。训练包含 critic 预热。", "",
             "C：原配方；D：走廊外每步惩罚上限 0.01 → 0.001；T：gamma=0.998、GAE lambda=0.995，其余与 C 相同。T 比较联合时间配方，不能单独归因给 gamma 或 lambda。", "",
             "三组使用相同 B1 表示和旧源权重，各训练 seeds 7、17。中期 48k；最终默认 96k，若时间不足则统一下调。具体参数见 [manifest.json](manifest.json)。", "",
             "主要指标为到达 5 米目标后连续保持 120 个物理步（1 秒）；并保留原 10 步成功、成功后跌落和有效进度。每次评估的每种模式回合数单独列出；未完成评估不计作零成功。", "",
             "## 已完成评估", ""]
    for r in runs:
        lines.append(f"### {r['arm']} / seed {r['seed']}：当前 {r['physical_samples']:,} 训练样本")
        lines.append("")
        for evaluation in r["evaluations"]:
            stage = Path(evaluation["directory"]).name
            lines.append(f"{stage}；评估权重对应训练样本 {evaluation['training_samples_at_checkpoint']:,}。")
            lines.append("")
            for mode, m in evaluation["summary"].items():
                lines.append(f"- {mode}：1 秒保持 {m['success_hold_120_count']}/{m['episodes']}；"
                             f"10 步成功 {m['successes']}/{m['episodes']}；成功后跌落 {m['falls_after_success']}；"
                             f"跌落率 {m['fall_rate']:.0%}；有效进度中位数 {m['credited_progress_median_m']:.2f} 米。")
            lines.append("")
    lines += ["## 解释边界", "", "只有两个训练 seed；同轮两个 agents 共享试验 seed，合并回合不能当作完全独立样本。奖励值随组别改变，不用累计奖励直接排名。全部候选无稳定成功时，不自动扩训或修改成功门槛。", "",
              "脚本在阶段之间自动继续，到截止时间只停止自己启动的进程树；STOP 文件在下个任务边界生效。原始权重和已有运行不会被覆盖。", ""]
    if error:
        lines += [f"停止原因：{error}", ""]
    (CAMPAIGN / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def choose_target(runner):
    """Budget-only decision, fixed for all six runs before any holdout is read."""
    for target in (96000, 72000, 48000):
        estimate = 300.  # report and startup reserve
        for arm, seed in ORDER:
            s = runner.stats(arm, seed)
            iterations = max(0, target - s["physical_samples"]) / max(1, s["recent_samples_per_iteration"])
            estimate += iterations * max(25., s["recent_iteration_seconds"]) * 1.4 + 90 + 540
        if datetime.now().astimezone().timestamp() + estimate < runner.deadline.timestamp():
            return target, estimate
    raise BudgetStop("Insufficient time for all six paired holdout evaluations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", help="Explicit timezone-aware campaign deadline")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(args.deadline or manifest["deadline"])
    if deadline.tzinfo is None or deadline.timestamp() > datetime.fromisoformat(manifest["deadline"]).timestamp():
        parser.error("Deadline must include timezone and stay within the approved budget")
    runner = Runner(deadline, campaign=CAMPAIGN, flags=FLAGS)
    if args.summary:
        state = json.loads((CAMPAIGN / "status.json").read_text(encoding="utf-8"))
        report(runner, state["status"], state.get("error"))
        return
    if not args.deadline:
        parser.error("Training requires an explicit deadline")
    if manifest["parameters"] != PARAMETERS:
        raise ValueError("Campaign parameters differ from the frozen manifest")
    if hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest().lower() != manifest["initial_checkpoint_sha256"].lower():
        raise ValueError("Common initial checkpoint changed")
    lock = CAMPAIGN / "controller.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"started_at": now(), "deadline": deadline.isoformat()}))
    status, error = "completed", None
    try:
        runner.guard()
        manifest.update(status="running", controller_started_at=now())
        write_json(CAMPAIGN / "manifest.json", manifest)
        runner.event("phase_started", phase="all", order=ORDER, deadline=deadline.isoformat())
        for arm, seed in ORDER:
            runner.preflight(arm, seed)
            runner.evaluate(arm, seed, "initial", 2, 20260920)
            runner.train_to(arm, seed, 48000)
            runner.evaluate(arm, seed, "screen_048k", 4, 20260920)
            report(runner, "running")
        decision_path = CAMPAIGN / "budget_decision.json"
        if decision_path.exists():
            target = json.loads(decision_path.read_text(encoding="utf-8"))["target_samples"]
        else:
            target, estimate = choose_target(runner)
            write_json(decision_path, {"decided_at": now(), "target_samples": target,
                                      "remaining_seconds_estimate": estimate, "basis": "time only; all arms retained"})
        runner.event("endpoint_budget", target_samples=target)
        for arm, seed in ORDER:
            runner.train_to(arm, seed, target)
            runner.evaluate(arm, seed, f"holdout_{target // 1000:03d}k", 5, 20261020)
            report(runner, "running")
    except BudgetStop as exc:
        status, error = "budget_stopped", str(exc)
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        runner.kill_own_child()
        manifest.update(status=status, finished_at=now(), error=error)
        write_json(CAMPAIGN / "manifest.json", manifest)
        write_json(CAMPAIGN / "status.json", {"status": status, "updated_at": now(),
                   "deadline": deadline.isoformat(), "error": error, "last_job": runner.job})
        runner.event("phase_finished", status=status, error=error)
        lock.unlink()
        totals = report(runner, status, error)
        manifest.update(training_physical_samples=totals["training_physical_samples"],
                        evaluation_physical_samples=totals["evaluation_physical_samples"])
        write_json(CAMPAIGN / "manifest.json", manifest)


if __name__ == "__main__":
    main()
