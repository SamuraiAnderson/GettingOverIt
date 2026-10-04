"""Compact results on the same preselected windows, with frozen baseline targets."""
import csv
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
assert report["status"] == "completed"
before = {(w["trajectory"], w["start"], w["end"]): w["before"] for w in report["fixed_windows"]}
def aggregate(rows):
    n = sum(r["frames"] for r in rows)
    return {"action_rmse": math.sqrt(sum(r["action_rmse"]**2*r["frames"] for r in rows)/n),
            "fixed_advantage_actor_loss": sum(r["fixed_advantage_actor_loss"]*r["frames"] for r in rows)/n}
draws = [tuple(d[:3]) for d in report["planned_draws"]]
summary = {"status": report["status"], "checks": report["checks"],
           "actual_ppo_updates": report["actual_ppo_updates"],
           "budget_reference_updates": report["budget_reference_updates"],
           "baseline_common_16_draws": aggregate([before[d] for d in draws]),
           "cold_input_build_seconds": report["preparation_cache"]["gru_sil_cache_build_seconds"],
           "arms": {}, "key_windows": []}
csv_rows = []
for name, arm in report["arms"].items():
    after = {(w["trajectory"], w["start"], w["end"]): w for w in arm["fit_after"]}
    summary["arms"][name] = {k: arm[k] for k in ("metrics", "accepted_positive_presentations",
        "unique_accepted_positive_frames", "repeated_accepted_positive_presentations",
        "immediate_actor_loss_improved_steps", "key_window_coverage", "audited_kl_mean", "audited_kl_max_frame",
        "warm_update_seconds_including_instrumentation")}
    summary["arms"][name]["common_16_draws_after"] = aggregate([after[d] for d in draws])
    summary["arms"][name]["gru_weight_max_absolute_delta"] = max(
        v for k,v in arm["parameter_max_absolute_delta"].items() if k.startswith("gru."))
    for w in report["fixed_windows"]:
        if w["rank"] <= 2 and w["start"] in (160,192,224):
            key = (w["trajectory"],w["start"],w["end"])
            summary["key_windows"].append({"arm": name, "rank": w["rank"], "start": w["start"],
                "end": w["end"], "before": before[key], "after": after[key]})
    trace = {t["draw"]:t for t in arm["step_trace"]}
    for i,chunk in enumerate(arm["sampling"]["chunks"],1):
        t = trace.get(i,{})
        csv_rows.append({"arm":name,"draw":i,**{k:chunk.get(k) for k in ("trajectory","rank","start","end","positive_frames","accepted","kl","frame_kl_max")},
            "actor_loss_before_step":t.get("actor_loss_before_step_fixed_advantage"),
            "actor_loss_after_step":t.get("actor_loss_after_step_fixed_advantage")})
summary["limitations"] = report["limitations"]
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
with (OUT/"update_trace.csv").open("w",encoding="utf-8-sig",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=csv_rows[0].keys());writer.writeheader();writer.writerows(csv_rows)
print(json.dumps(summary,indent=2,ensure_ascii=False))
