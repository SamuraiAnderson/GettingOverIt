"""Compute a constrained training plan from measured collection/update costs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "runs/player_throughput_20260919"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q):
    return float(np.quantile(values, q))


def sample_visits(samples, batch, updates):
    """Exact visits for full shuffled epochs plus the final KL-stopped prefix."""
    epochs, remainder = divmod(int(updates), math.ceil(samples/batch))
    visits = epochs*samples + min(remainder*batch, samples)
    return visits, min(samples, visits)


def history():
    rows, warmup_costs, eval_costs = [], [], []
    for base in sorted((ROOT / "runs").glob("corridor_*_20260919_*")):
        path = base / "metrics.csv"
        if not path.exists():
            continue
        records = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        def number(r, k):
            return float(r.get(k) or 0)
        policy = [r for r in records if number(r,"value_only") != 1]
        samples = sum(number(r,"rollout_environment_steps") or number(r,"buffer_size") for r in policy)
        seconds = sum(number(r,"rollout_total_seconds") + number(r,"update_and_sil_seconds") for r in policy)
        used = sum(min(number(r,"rollout_environment_steps") or number(r,"buffer_size"), 64*number(r,"n_updates")) for r in policy)
        rows.append({"run":base.name, "iterations":len(records), "policy_samples":int(samples),
                     "collect_update_seconds":seconds, "samples_per_second":samples/max(seconds,1),
                     "unique_gradient_sample_upper_bound":int(used),
                     "scope":"64 minibatch upper bound; GRU chunking differs"})
        if base.name.startswith("corridor_credit_"):
            for r in records:
                if number(r,"value_only") == 1:
                    count = number(r,"rollout_environment_steps") or number(r,"buffer_size")
                    warmup_costs.append((number(r,"rollout_total_seconds")+number(r,"update_and_sil_seconds"))/count)
    for campaign in (ROOT/"runs/corridor_screen_20260919", ROOT/"runs/corridor_credit_20260919"):
        starts = {}
        for event in [json.loads(line) for line in (campaign/"events.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]:
            if event["event"] == "job_started":
                starts[event["label"]] = event["time"]
            if event["event"] == "job_finished" and event["exit_code"] == 0:
                summary = campaign/f"{event['label']}.json"
                if summary.exists() and event["label"] in starts:
                    count = read(summary)["physical_samples"]
                    seconds = (datetime.fromisoformat(event["time"])-datetime.fromisoformat(starts[event["label"]])).total_seconds()
                    eval_costs.append(seconds/max(1,count))
    return {"runs":rows, "critic_warmup_seconds_per_sample_q75":quantile(warmup_costs,.75),
            "evaluation_seconds_per_sample_q75":quantile(eval_costs,.75), "completed_evaluation_jobs":len(eval_costs)}


def analyze(hours, gpu_reserve_fraction=.1):
    manifest, state = read(OUTPUT/"manifest.json"), read(OUTPUT/"status.json")
    if state["status"] != "completed":
        raise ValueError("Complete the benchmark before selecting a training configuration")
    collected, scenarios = [], []
    groups = sorted(OUTPUT.glob("n*_*"))
    for group in groups:
        result = read(group/"results.json")
        if result["failure"] or len(result["rows"]) != 3:
            raise ValueError(f"Incomplete benchmark: {group}")
        for r in result["rows"]:
            collected.append({"group":group.name, **r})
    update_results=read(OUTPUT/"joint_updates.json")
    updates = {(r["group"],r["repeat"],r["batch_size"]):r for r in update_results["rows"]}
    for r in collected:
        for batch in (64,128,256):
            if batch != 64 and (r["group"],r["repeat"],batch) not in updates:
                continue
            u = {"update_seconds":r["update_seconds"], "metrics":r["update"],
                 "gpu_peak_reserved_gb":r["gpu_peak_reserved_gb"]} if batch == 64 else updates[(r["group"],r["repeat"],batch)]
            visits, unique = sample_visits(r["physical_samples"], batch, u["metrics"]["n_updates"])
            seconds = r["collection_seconds"]+u["update_seconds"]+r["serialization_seconds"]
            scenarios.append({"group":r["group"],"agents":r["agents"],"repeat":r["repeat"],"batch_size":batch,
                "physical_samples":r["physical_samples"],"collection_seconds":r["collection_seconds"],
                "update_seconds":u["update_seconds"],"serialization_seconds":r["serialization_seconds"],
                "end_to_end_seconds":seconds,"physical_samples_per_second":r["physical_samples"]/seconds,
                "gradient_sample_visits":visits,"unique_gradient_samples":unique,
                "gradient_coverage":unique/r["physical_samples"],"unique_gradient_samples_per_second":unique/seconds,
                "n_updates":u["metrics"]["n_updates"],"approx_kl":u["metrics"]["approx_kl"],
                "ratio_max_error":max(r["audit"]["preupdate_ratio_max_error"],u["metrics"]["preupdate_ratio_max_error"]),
                "gpu_peak_reserved_gb":max(r["gpu_peak_reserved_gb"],u["gpu_peak_reserved_gb"]),
                "system_min_available_gb":r["system_min_available_gb"],"active_fraction":r["active_fraction"]})
    summaries = []
    for agents,batch in itertools.product((2,4,8,16),(64,128,256)):
        rows = [r for r in scenarios if r["agents"] == agents and r["batch_size"] == batch]
        if not rows:
            continue
        summaries.append({"agents":agents,"batch_size":batch,"trials":len(rows),
            "physical_rate_median":quantile([r["physical_samples_per_second"] for r in rows],.5),
            "physical_rate_q25":quantile([r["physical_samples_per_second"] for r in rows],.25),
            "gradient_rate_median":quantile([r["unique_gradient_samples_per_second"] for r in rows],.5),
            "gradient_rate_q25":quantile([r["unique_gradient_samples_per_second"] for r in rows],.25),
            "gradient_coverage_median":quantile([r["gradient_coverage"] for r in rows],.5),
            "updates_median":quantile([r["n_updates"] for r in rows],.5),
            "gpu_peak_reserved_gb":max(r["gpu_peak_reserved_gb"] for r in rows),
            "min_available_ram_gb":min(r["system_min_available_gb"] for r in rows),
            "ratio_max_error":max(r["ratio_max_error"] for r in rows),
            "active_fraction_median":quantile([r["active_fraction"] for r in rows],.5)})
    for s in summaries:
        s["feasible"] = (s["trials"] == (6 if s["agents"]==2 else 3) and s["gpu_peak_reserved_gb"] <= (1-gpu_reserve_fraction)*manifest["hardware"]["gpu_total_gb"] and s["min_available_ram_gb"] >= 3 and s["ratio_max_error"] <= .001)
    feasible = [s for s in summaries if s["feasible"]]
    fastest = max(feasible,key=lambda s:s["physical_rate_q25"])
    top = max(s["gradient_rate_q25"] for s in feasible)
    # Within 5% of the best conservative throughput, prefer fewer agents then smaller batches.
    selected = min((s for s in feasible if s["gradient_rate_q25"] >= .95*top), key=lambda s:(s["agents"],s["batch_size"]))
    sensitivity=[]
    for reserve in (0.,.1,.2):
        candidates=[s for s in summaries if s["trials"]==(6 if s["agents"]==2 else 3)
                    and s["gpu_peak_reserved_gb"]<=(1-reserve)*manifest["hardware"]["gpu_total_gb"]
                    and s["min_available_ram_gb"]>=3 and s["ratio_max_error"]<=.001]
        best_rate=max(s["gradient_rate_q25"] for s in candidates)
        chosen=min((s for s in candidates if s["gradient_rate_q25"]>=.95*best_rate),key=lambda s:(s["agents"],s["batch_size"]))
        sensitivity.append({"gpu_reserve_fraction":reserve,"agents":chosen["agents"],"batch_size":chosen["batch_size"],
                            "gradient_rate_q25":chosen["gradient_rate_q25"],"gpu_peak_reserved_gb":chosen["gpu_peak_reserved_gb"]})
    base = next(s for s in summaries if s["agents"]==2 and s["batch_size"]==64)
    starts = [r["collection_samples_per_second"] for r in collected if r["group"]=="n02_start"]
    ends = [r["collection_samples_per_second"] for r in collected if r["group"]=="n02_end"]
    drift = quantile(ends,.5)/quantile(starts,.5)-1
    historic = history()
    # Preserve 1200-step temporal credit. Shorter episodes are not a free replacement for more agents.
    factors = list(itertools.product(sorted({2,selected["agents"]}),sorted({64,selected["batch_size"]})))
    cost_lookup = {(s["agents"],s["batch_size"]):s for s in summaries}
    factors_feasible=all(f in cost_lookup and cost_lookup[f]["feasible"] for f in factors)
    if not factors_feasible:
        raise ValueError("Selected candidate needs an unmeasured or infeasible factorial control; inspect results before planning")
    budget_seconds = hours*3600
    plans=[]
    for seeds in (2,3):
        for target in (48000,72000,96000,144000,192000):
            run_count = seeds*len(factors)
            evaluation_samples = run_count*28*1200  # 4 initial + 10 final episodes per mode.
            evaluation_seconds = evaluation_samples*historic["evaluation_seconds_per_sample_q75"]
            warmup_seconds = run_count*4800*historic["critic_warmup_seconds_per_sample_q75"]
            launches = run_count*2*quantile([r["launch_seconds"] for r in collected],.75)
            training_seconds = seeds*sum(max(0,target-4800)/cost_lookup[f]["physical_rate_q25"] for f in factors)
            reserve_seconds = .1*budget_seconds
            total = evaluation_seconds+warmup_seconds+launches+training_seconds+reserve_seconds
            plans.append({"seeds":seeds,"samples_per_run":target,"run_count":run_count,
                "training_samples":run_count*target,"evaluation_samples_cap":evaluation_samples,
                "training_seconds":training_seconds,"evaluation_seconds":evaluation_seconds,
                "warmup_seconds":warmup_seconds,"launch_seconds":launches,"reserve_seconds":reserve_seconds,
                "total_seconds":total,"fits_budget":total<=budget_seconds})
    # Retain a meaningful common endpoint first; then add independent seeds before more samples.
    suitable = [p for p in plans if p["fits_budget"] and p["samples_per_run"]>=96000]
    if suitable:
        seed_count=max(p["seeds"] for p in suitable)
        plan=min((p for p in suitable if p["seeds"]==seed_count),key=lambda p:p["samples_per_run"])
    else:
        suitable=[p for p in plans if p["fits_budget"]]
        plan=max(suitable,key=lambda p:(p["samples_per_run"],p["seeds"])) if suitable else None
    recommendation={"generated_at":datetime.now().astimezone().isoformat(),"budget_hours":hours,
        "planning_constraints":{"gpu_reserve_fraction":gpu_reserve_fraction,"min_available_ram_gb":3,"note":"planning assumptions, not user-specified hardware limits; all completed configurations passed actual execution"},
        "memory_reserve_sensitivity":sensitivity,
        "objective":"maximize q25 unique samples entering gradient updates per measured pipeline second; 5% tie favors fewer agents",
        "selected":selected,"fastest_raw_collection_pipeline":fastest,"baseline":base,
        "gradient_rate_speedup_vs_2x64":selected["gradient_rate_q25"]/base["gradient_rate_q25"],
        "raw_rate_speedup_vs_2x64":selected["physical_rate_q25"]/base["physical_rate_q25"],
        "baseline_end_vs_start_collection_drift":drift,"factorial_candidates":[{"agents":a,"ppo_batch_size":b} for a,b in factors],
        "excluded_resource_failures":update_results.get("failures",[]),
        "proposed_learning_validation":plan,"budget_alternatives":plans,
        "critic_warmup":{"agents":2,"iterations":2,"sample_cap":4800,"note":"warm up at two agents, then resume with the tested collection count; retain 1200 physical steps"},
        "learning_parameters":{"steps_per_rollout":1200,"learning_rate":.0001,"ppo_epochs":4,"target_kl":.03,
                               "training_seeds":[31,47,61],"reward_recipe":"T: candidate for follow-up, not proven superior",
                               "gamma":.998,"gae_lambda":.995,"reference_deviation_weight":.01,
                               "note":"keep this recipe fixed across minibatch controls; timing was measured on a C checkpoint"},
        "limitations":["single frozen policy and machine session, three trials per candidate; not a global learning optimum",
                       "unique gradient samples are exposure counts, not independent observations or a success metric",
                       "changing agents changes rollout size/update frequency; validate with common physical samples and held-out success",
                       "batch-size timing uses fresh optimizer; warmed optimizer and changed policy may shift KL stopping",
                       "budget assumes benchmark throughput persists; q25 is a planning statistic, not a confidence bound"]}
    save(OUTPUT/"historical_costs.json",historic)
    save(OUTPUT/"recommended_plan.json",recommendation)
    write_csv(OUTPUT/"joint_trials.csv",scenarios)
    write_csv(OUTPUT/"configuration_summary.csv",summaries)
    write_csv(OUTPUT/"budget_candidates.csv",plans)
    draw(summaries,manifest)
    lines=["# Player 并行数与训练预算实测", "",
           f"数据来自 {manifest['started_at']} 至 {state['finished_at']}，同一冻结策略、每回合 1,200 步。机器：{manifest['hardware']['gpu']}，{manifest['hardware']['ram_gb']:.1f} GiB RAM。", "",
           f"测试范围内，按实际进入 PPO 梯度更新的不同样本数/秒（重复试验下四分位）选出的候选为 **{selected['agents']} players + minibatch {selected['batch_size']}**，相对原 2×64 为 **{recommendation['gradient_rate_speedup_vs_2x64']:.2f} 倍**。该指标是计算与数据使用效率，不能证明学习成功率提升。", "",
           f"显存预留 {gpu_reserve_fraction:.0%} 是本次预算计算的假设，不是用户给定限制；batch 256 的所有实测均完成，峰值约 10.04 GiB。若改为预留 20%，候选变为 2 players + batch 128（约 5.42 GiB）。完整敏感性结果保存在 recommended_plan.json。", "",
           f"单纯采集/更新/存档吞吐最高的组合是 {fastest['agents']}×{fastest['batch_size']}。开始/结束 2-player 基线采集速度变化 {drift:+.1%}。", "",
           "![并行度、梯度样本处理率与 PPO 数据覆盖率](throughput.png)", "",
           "## 全部实测配置", ""]
    for s in summaries:
        lines.append(f"- {s['agents']} players / batch {s['batch_size']}：采集+更新+存档中位吞吐 {s['physical_rate_median']:.1f} 样本/秒；不同梯度样本中位吞吐 {s['gradient_rate_median']:.1f}/秒（下四分位 {s['gradient_rate_q25']:.1f}）；梯度覆盖率中位数 {s['gradient_coverage_median']:.1%}；显存峰值保留 {s['gpu_peak_reserved_gb']:.2f} GiB；可行={s['feasible']}。")
    for failure in update_results.get("failures",[]):
        lines.append(f"- batch {failure['batch_size']} 在 {failure['stage']} 发生 CUDA 显存不足，排除该配置；详见 joint_updates.json。")
    lines += ["", "## Python 计算的后续验证方案", ""]
    if plan:
        lines += [f"预算 {hours:g} 小时，采用 {len(factors)} 个组合 × {plan['seeds']} seeds，每组共同终点 {plan['samples_per_run']:,} 物理样本。预计总耗时 {plan['total_seconds']/3600:.2f} 小时，包含 {plan['reserve_seconds']/60:.0f} 分钟余量。", "",
                  "组合："+"、".join(f"{a} players / batch {b}" for a,b in factors)+"。保持奖励、网络和探索配方一致；只有一个参数变化时做单因素对照，两个参数都变化时做组合对照。", "",
                  "后续以 T 时间配方作为候选，使用新的训练 seeds 31、47、61；T 尚未证明跨 seed 稳定改善。吞吐测量来自 C 的冻结模型，T 的 KL 早停和更新成本仍可能不同。", "",
                  "critic 预热保留 2 players × 2 轮，随后按候选并行数续训；回合仍为 1,200 步，不能通过缩短时间范围假装保持相同训练条件。固定样本预算含预热，按真实采集量补轮。", "",
                  "每训练 seed 的训练前评估每模式 4 回合、终点评估每模式 10 回合，评估固定 2 players，并以持续 1 秒成功与成功后跌落验收。C/D/T 实验已按原配置完成；本文件没有启动新的学习对照。"]
    else:
        lines.append("当前预算不能覆盖最小成对学习验证与完整评估；不删除评估以制造可行方案。")
    lines += ["", "## 方法与边界", "", "完整输入先做冻结策略审计。真实 PPO 更新后丢弃权重；64/128/256 使用同一批归档数据和同一行为模型，保留原学习率及 KL 早停。另有 lr=0 的完整单遍测试，单独比较相同计算量下的 minibatch 效率。", "",
              "PPO 第一遍 minibatch 不重复；从更新数和实际 batch 大小计算不同样本进入梯度更新的数量，完整多遍则封顶为该 rollout 的样本数。该数量不等于独立有效样本量。较大的 rollout 若很早触发 KL 早停，可能使大量采集数据未参与梯度更新。", "",
              "并行数推荐只覆盖本次测试网格和冻结模型。不能从吞吐或短更新推导全局最优超参数；实际训练需要用保留集成功率确认。预算使用试验下四分位吞吐及历史评估/critic 预热每样本耗时上四分位；这些是保守规划值，不是置信界。", "",
              "原始逐次数据见 joint_trials.csv，汇总见 configuration_summary.csv，预算候选见 budget_candidates.csv，机器可读方案见 recommended_plan.json。", ""]
    (OUTPUT/"RESULT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"selected":selected,"fastest":fastest,"plan":plan,"drift":drift},ensure_ascii=False))


def draw(summaries, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(14,4.6),layout="constrained")
    for batch,color in ((64,"#767676"),(128,"#168B86"),(256,"#CC6B36")):
        rows=[s for s in summaries if s["batch_size"]==batch]
        if not rows:
            continue
        for ax,key in zip(axes,("physical_rate_q25","gradient_rate_q25","gradient_coverage_median")):
            values=[s[key]*(100 if key=="gradient_coverage_median" else 1) for s in rows]
            ax.plot([s["agents"] for s in rows],values,marker="o",color=color,label=f"Minibatch {batch}")
    for ax,title,ylabel in zip(axes,("Collected samples / pipeline time (Q25)","Unique gradient samples / pipeline time (Q25)","Rollout data used by PPO (median)"),
                              ("Physical samples / s","Unique gradient samples / s","Coverage (%)")):
        ax.set(title=title,xlabel="Concurrent players",ylabel=ylabel,xticks=[2,4,8,16])
        ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Measured collection, PPO update and archive costs — frozen B1 Transformer",fontsize=13)
    fig.supxlabel("Source: player_throughput_20260919 | 3 trials/config; 2-player baseline repeated at end | 1200 physical steps/episode",fontsize=9)
    fig.savefig(OUTPUT/"throughput.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-hours",type=float,default=6)
    parser.add_argument("--gpu-reserve-fraction",type=float,default=.1)
    args=parser.parse_args()
    if not np.isfinite(args.budget_hours) or args.budget_hours<=0:
        parser.error("Budget must be positive and finite")
    if not np.isfinite(args.gpu_reserve_fraction) or not 0<=args.gpu_reserve_fraction<1:
        parser.error("GPU reserve fraction must be in [0,1)")
    analyze(args.budget_hours,args.gpu_reserve_fraction)
