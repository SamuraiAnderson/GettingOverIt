"""Recover the last evaluation, then finish the fixed-weight control in a bounded job."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from tests.analysis.benchmark_player_throughput import write_json, stamp

OUTPUT = ROOT/"runs/pipeline_20260920"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def recover_last_evaluation():
    import psutil
    import torch
    from training.config import TrainConfig
    from training.rollout_pipeline import RolloutPipeline
    if any(p.info["name"]=="GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running")
    live = OUTPUT/"live"
    manifest = read(live/"pipeline_manifest.json")
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest["hashes"].items())
    run = live/"n08_pipeline"
    data = read(run/"results.json")
    assert len(data["rows"])==3 and data["status"]=="running"
    write_json(run/"interrupted_results.json",data)
    write_json(live/"interrupted_status.json",read(live/"pipeline_status.json"))
    cfg = TrainConfig(**read(run/"config.json")["config"])
    runtime = Path(cfg.game_root)/"GoiData/runtime_config.json"
    original = (live/"checkpoints/runtime_original.json").read_bytes()
    if runtime.read_bytes()!=original:
        runtime.write_bytes(original)
    target = run/"recovered_evaluation"
    state = torch.load(run/"checkpoints/behavior_1000.pt",map_location="cpu",weights_only=False)
    pipeline = None
    try:
        pipeline = RolloutPipeline(cfg,target)
        pipeline.submit({"id":1000,"version":state["version"],
            "snapshot":str(run/"checkpoints/behavior_1000.pt"),"steps":1200,
            "seed":read(run/"config.json")["seed"]+1000,"deterministic":True})
        evaluation = pipeline.receive()
        write_json(target/"results.json",evaluation)
    finally:
        if pipeline is not None:
            pipeline.close()
    rows = data["rows"]
    # Parent start/finish and monitor peaks were not saved before interruption.
    # This interval omits initial dispatch and final bookkeeping => a LOWER bound
    # on elapsed wall time, hence an UPPER bound on throughput. Never invent peaks.
    interval = rows[-1]["update_end"]-rows[0]["started"]
    total = sum(r["physical_samples"] for r in rows)
    used = sum(r["unique_gradient_samples"] for r in rows)
    overlap = sum(max(0.,min(a["update_end"],b["collection_end"])-max(a["update_started"],b["started"]))
                  for a,b in zip(rows,rows[1:]))
    summary = {"mode":"pipeline","agents":8,"rounds":3,"seed":read(run/"config.json")["seed"],
        "batch_size":cfg.ppo_batch_size,"lr":cfg.lr,"physical_samples":total,
        "training_wall_seconds":interval,"physical_samples_per_second":total/interval,
        "unique_gradient_samples_per_second":used/interval,"gradient_coverage":used/total,
        "rejected_rollouts":sum(bool(r["rejected"]) for r in rows),"overlap_seconds":overlap,
        "max_version_lag":max(r["version_lag"] for r in rows),
        "max_behavior_audit_error":max(r["audit"]["preupdate_ratio_max_error"] for r in rows),
        "whole_device_peak_used_gb":None,"learner_actor_game_peak_rss_gb":None,"system_min_available_gb":None,
        "timing_kind":"elapsed_lower_bound_rates_upper_bound", "recovered":True}
    data.update(status="completed",summary=summary,evaluation=evaluation["scores"],
        recovered_at=stamp(), recovery_note="Training and final.pt completed before interruption; evaluation resumed. Exact parent wall time and whole-device peaks unavailable; throughput values are upper bounds.")
    write_json(run/"results.json",data)
    combined = read(live/"pipeline_summary.json")
    assert len(combined["rows"])==5
    combined["rows"].append(summary)
    write_json(live/"pipeline_summary.json",combined)
    write_json(live/"pipeline_status.json", {"status":"completed","finished":stamp(),
        "completed":6,"failures":[],"changed_sources":[],"recovered_job":"n08_pipeline",
        "measurement_note":"Last arm's throughput is an upper bound; full timing was lost on interruption."})
    print(json.dumps(summary),flush=True)


def main():
    child = None
    try:
        write_json(OUTPUT/"recovery_status.json",{"status":"running","stage":"evaluation","updated":stamp()})
        recover_last_evaluation()
        write_json(OUTPUT/"recovery_status.json",{"status":"running","stage":"fixed_weight_control","updated":stamp()})
        command = [sys.executable,"-m","src.tests.analysis.benchmark_rollout_pipeline","--minutes","35",
            "--rounds","2","--lr","0","--output",str(OUTPUT/"frozen")]
        child = subprocess.Popen(command,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        child.wait(timeout=36*60)
        if child.returncode:
            raise RuntimeError(f"Frozen control failed: {child.returncode}")
        for module in ("verify_pipeline_artifacts","summarize_rollout_pipeline"):
            subprocess.run([sys.executable,"-m",f"src.tests.analysis.{module}"],cwd=ROOT,check=True)
        write_json(OUTPUT/"recovery_status.json",{"status":"completed","finished":stamp()})
    except BaseException as exc:
        if child is not None and child.poll() is None:
            subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],capture_output=True)
        write_json(OUTPUT/"recovery_status.json",{"status":"failed","error":repr(exc),"updated":stamp()})
        raise


if __name__ == "__main__":
    main()
