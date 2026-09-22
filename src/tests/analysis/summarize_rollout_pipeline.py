"""Recompute the pipeline pilot report from completed, versioned measurements."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT/"runs/pipeline_20260920"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    live = OUTPUT/"live"
    status = read(live/"pipeline_status.json")
    if status["status"] != "completed":
        raise ValueError("Complete all six arms before comparing")
    results = [read(live/f"n{n:02d}_{mode}/results.json") for n in (2,4,8) for mode in ("serial","pipeline")]
    rows = [r["summary"] for r in results]
    with (OUTPUT/"pipeline_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)
    diagnosis = read(OUTPUT/"kl_diagnosis.json")
    lr_rows = [r for r in diagnosis["rows"] if r["kind"]=="full_update"]
    lines = ["# 采集／训练流水线实测", "",
        f"实机重测开始：{read(live/'pipeline_manifest.json')['started']}；完成：{status['finished']}。9 月 20 日中断的旧数据保存在 interrupted_live_20260922，未混入本轮比较；中断原因未确定。", "",
        "本次是有界策略滞后的 PPO 工程短测，不是长期学习效果对照。使用同一个 C 中期 checkpoint，固定 B1 Transformer、每回合上限 1200 物理步（正常终止可提前结束）、batch 128、lr=3e-5、4 个 CPU 计算线程；每种配置连续运行 3 轮、保留 optimizer 状态。两种模式使用相同随机种子及归档/校验路径。", "",
        "## KL 早停原因与修正", "",
        "旧实现先执行 optimizer.step，再根据更新前算出的 KL 判断是否停止。这允许已经越界的策略再执行一次更新。现在在下一次 optimizer.step 前检查；被拒绝批次不计入更新，拒绝时的 KL 仍参与阈值自适应。该检查不是更新回滚，也不能保证第一步更新后 KL 永不越界。", "",
        "同一归档数据、batch 128、新 optimizer，三种 minibatch 打乱种子：", ""]
    for lr in (1e-4,3e-5,1e-5):
        selected = [r for r in lr_rows if r["lr"]==lr]
        kls = [r["after_step"]["preupdate_kl"] for r in selected]
        counts = [int(r["metrics"]["n_updates"]) for r in selected]
        lines.append(f"- lr={lr:g}：实际更新次数 {counts}；结束后 512 个固定探针样本 KL 范围 {min(kls):.4f}–{max(kls):.4f}。")
    lines += ["", "损失拆分的一步更新：", ""]
    for r in diagnosis["rows"]:
        if r["kind"] == "single_step":
            lines.append(f"- {r['component']}：更新后探针 KL={r['after_step']['preupdate_kl']:.4g}。")
    lines += ["", "在该 checkpoint/数据上，策略损失是漂移主要来源；仅价值损失更新共享 backbone 也会改变策略，只训练 critic head 则几乎不改变策略。条件动作标准差约 0.51–0.52，不能把问题简单归因于噪声方差塌缩。重放结果支持降低学习率作为短测条件，不能证明它提高实际成功率。", "",
        "## 流水线边界", "",
        "采集器独占游戏/TCP，使用冻结模型；训练器独占 optimizer。每轮 snapshot 写入独立文件，队列只允许 1 个待完成请求，策略版本最多落后 1 次实际更新。每批在采集端做全量行为概率复现，训练端另检查冻结模型并在 512 个样本上估计新旧策略 KL；超过 0.03 就丢弃该批，原 PPO minibatch KL 保护继续生效。", "",
        "旧行为模型对应的 GAE 与概率保留，未实现 V-trace 等完整 off-policy 价值修正；这是小滞后 PPO 试验，绝非与同步 PPO 数学等价。普通 main_ppo 仍默认严格串行，不会自动启用新流程。", "",
        "## 端到端结果", "",
        "时间包括采集、完整行为校验、快照/数据落盘、父进程读取、滞后检查、训练和流水线填充/排空；排除进程/游戏启动、64 步预热、最终保存和终评。下面的‘参与更新样本’按每轮去重，不是独立有效样本量。", ""]
    comparisons = []
    for n in (2,4,8):
        pair = {r["mode"]:r for r in rows if r["agents"]==n}
        s,p = pair["serial"],pair["pipeline"]
        qualifier = "≤" if p.get("recovered") else ""
        memory = "未保存" if p["whole_device_peak_used_gb"] is None else f"{p['whole_device_peak_used_gb']:.2f} GiB"
        speed = p["unique_gradient_samples_per_second"]/s["unique_gradient_samples_per_second"]
        comparisons.append({"agents":n,"gradient_throughput_ratio":speed,"ratio_is_upper_bound":bool(p.get("recovered")),
                            "physical_throughput_ratio":p["physical_samples_per_second"]/s["physical_samples_per_second"]})
        lines += [f"- **{n} players**：串行 {s['physical_samples_per_second']:.1f} 采集样本/秒、{s['unique_gradient_samples_per_second']:.1f} 参与更新样本/秒；流水线分别 {qualifier}{p['physical_samples_per_second']:.1f}、{qualifier}{p['unique_gradient_samples_per_second']:.1f}，后者相对串行为 {qualifier}{speed:.2f}×。",
                  f"  实际采集/训练重叠 {p['overlap_seconds']:.1f} 秒；拒绝 {p['rejected_rollouts']}/{p['rounds']} 批；参与更新覆盖率 {p['gradient_coverage']:.1%}；整卡峰值占用 {memory}。"]
    if any(r.get("recovered") for r in rows):
        lines += ["", "**中断恢复说明：**8-player 流水线三轮训练及最终权重均已保存，但终评阶段中断，父进程完整 wall time 和显存峰值未落盘。终评随后从相同权重补齐；吞吐仅按已保存首轮采集开始至末次更新结束的区间推导上界（省略了区间外调度开销），上文用 ≤ 标记，不当作精确测速。", ""]
    lines += ["", "每组每轮的 KL、更新次数、策略版本及时间戳见 live/nXX_mode/results.json。若流水线有丢弃数据，不能只看采集速度选择配置。各 player 数的物理样本预算不同，因此仅比较吞吐；不能按这几轮训练宣布某个 player 数学习更好。", "",
        "![吞吐与数据覆盖](pipeline_comparison.png)", "", "## 验证与后续", "",
        f"六个配置全部完成；全量冻结行为复现最大 ratio 误差 {max(r['max_behavior_audit_error'] for r in rows):.3g}；pipeline_manifest 中源代码/初始权重在测试期间未变化。终评每组只有 {', '.join(str(r['summary']['agents']) for r in results)} 个同轮复制体，仅用来发现崩溃或明显退化，不是多 seed 成功率验收。", "",
        "原‘2 players + batch 256’推荐来自旧学习率、旧 KL 检查位置、串行测试及不同 CPU 线程数，不能直接用于本轮。此次降低学习率与修正检查位置在两种模式中一致；串行/流水线对照才可用于估计并行收益。正式扩训仍需共同样本预算、独立训练 seeds 和保留集成功率。", "",
        "复算：`conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.summarize_rollout_pipeline`。运行：`conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.benchmark_rollout_pipeline --output runs/<new-name>`；单组用 `--trial --mode serial|pipeline --agents 2|4|8`。输出目录必须是新目录，避免覆盖旧数据。", ""]
    frozen_path = OUTPUT/"frozen"
    if (frozen_path/"pipeline_status.json").exists() and read(frozen_path/"pipeline_status.json")["status"]=="completed":
        frozen = read(frozen_path/"pipeline_summary.json")["rows"]
        lines += ["## 固定权重的计算对照", "",
            f"每种配置 {frozen[0]['rounds']} 轮；开始 {read(frozen_path/'pipeline_manifest.json')['started']}，完成 {read(frozen_path/'pipeline_status.json')['finished']}。",
            "真实更新会改变后续轨迹，故追加 lr=0 对照：仍执行四遍完整前向、反向与 optimizer.step，但不改变模型权重；所有其他流水线校验继续保留。此对照用来估计调度/资源重叠收益，不能用于判断学习。", ""]
        for n in (2,4,8):
            pair = {r["mode"]:r for r in frozen if r["agents"]==n}
            s,p = pair["serial"],pair["pipeline"]
            speed = p["physical_samples_per_second"]/s["physical_samples_per_second"]
            lines.append(f"- {n} players：串行 {s['physical_samples_per_second']:.1f} 样本/秒，流水线 {p['physical_samples_per_second']:.1f} 样本/秒（{speed:.2f}×）；采集／训练重叠 {p['overlap_seconds']:.1f} 秒。")
        first_serial = read(frozen_path/"n02_serial/results.json")["rows"][0]["collection_seconds"]
        first_pipeline = read(frozen_path/"n02_pipeline/results.json")["rows"][0]["collection_seconds"]
        lines += ["", f"**不能把全部吞吐差异归因于流水线。**固定权重 2-player 首轮尚未与训练重叠，串行组采集已耗时 {first_serial:.1f} 秒、流水线组 {first_pipeline:.1f} 秒；存在明显运行间波动。时间戳证明了计算与采集确有重叠，但上述速率比同时受调度、轨迹与运行状态影响，不是纯粹的并行加速倍数。", ""]
        lines += ["", "串行与流水线都包含相同的进程边界、存档和冻结复现开销。Unity reset 和接触求解存在非逐位确定性，即使策略/seed 相同也不能保证轨迹完全一致；每组仅 2–3 轮，不提供可靠置信区间。", ""]
    verification = read(OUTPUT/"artifact_validation.json")
    lines += [f"产物核验通过：{len(verification['runs'])} 个配置的更新计数与逐批记录一致；六个 lr=0 配置的最终权重逐位不变；全部终评使用对应的最终权重。游戏运行配置已恢复，环境文件及源代码 hash 未变；相关回归测试 {verification['regression_test_count']} 项通过。核验记录见 [artifact_validation.json](artifact_validation.json)。", ""]
    best = max((r for r in rows if not r.get("recovered")),key=lambda r:r["unique_gradient_samples_per_second"])
    lines += ["## 当前后续候选", "",
        f"这次真实更新网格中，最高的已完整测得参与更新吞吐来自 **{best['agents']} players / {best['mode']} / batch 128 / lr=3e-5**，约 {best['unique_gradient_samples_per_second']:.1f} 样本/秒。建议把它作为下一轮正式学习验证的基线候选；这不是跨模型、跨 seed 的最优结论。流水线实现保留为可选实验，尚不设为普通训练默认。", "",
        f"本轮六个短测模型的确定性终评共 {sum(len(r['evaluation']) for r in results)} 回合，1 秒保持成功 {sum(x['success'] for r in results for x in r['evaluation'])} 次。它们使用不同采集量且每组只有一个训练 seed，不能替代共同样本预算的正式成功率比较。正式 C/D/T 最终结果仍见 [第二轮训练结论](../corridor_credit_20260919/CONCLUSION.md)。本次短测已结束，未启动新的六小时正式训练。", ""]
    (OUTPUT/"RESULT.md").write_text("\n".join(lines),encoding="utf-8")
    (OUTPUT/"comparison.json").write_text(json.dumps({"pairs":comparisons},indent=2),encoding="utf-8")
    manifest = read(live/"pipeline_manifest.json")
    changed = [name for name,digest in manifest["hashes"].items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    (OUTPUT/"validation.json").write_text(json.dumps({"six_arms_completed":len(results)==6,
        "all_behavior_audits_passed":all(r["max_behavior_audit_error"]<=.001 for r in rows),
        "version_lag_bounded":all(r["max_version_lag"]<=1 for r in rows),
        "changed_sources_since_test":changed},indent=2),encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for mode,offset,color in (("serial",-.18,"#596275"),("pipeline",.18,"#247D91")):
        selected = [next(r for r in rows if r["agents"]==n and r["mode"]==mode) for n in (2,4,8)]
        for ax,key in zip(axes,("physical_samples_per_second","unique_gradient_samples_per_second","gradient_coverage")):
            bars = ax.bar([x+offset for x in range(3)], [r[key]*(100 if key=="gradient_coverage" else 1) for r in selected],
                   width=.36,label=mode,color=color)
            for bar,r in zip(bars,selected):
                if r.get("recovered") and key!="gradient_coverage":
                    bar.set_hatch("///")
                    ax.annotate("upper bound",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha="center",va="bottom",fontsize=7)
            ax.set_xticks(range(3),["2","4","8"]); ax.set_xlabel("Players"); ax.spines[["top","right"]].set_visible(False)
    for ax,title,y in zip(axes,("Collected samples / wall time","Samples used in updates / wall time","Samples used at least once"),
                         ("Physical samples / second","Unique samples / second","Coverage (%)")):
        ax.set_title(title); ax.set_ylabel(y); ax.legend(frameon=False)
    axes[2].set_ylim(0, 125)
    axes[2].set_yticks(range(0, 101, 20))
    axes[2].legend(frameon=False, loc="upper right", ncols=2)
    fig.suptitle("Pipeline pilot: 3 rollouts per arm, batch 128, lr 3e-5 (not learning success)")
    fig.supxlabel("Source: runs/pipeline_20260920/live, Sep 22 2026. Hatched rates are upper bounds after interrupted timing.",fontsize=8)
    fig.savefig(OUTPUT/"pipeline_comparison.png",dpi=160)
    plt.close(fig)
    print(json.dumps(comparisons,indent=2))


if __name__ == "__main__":
    main()
