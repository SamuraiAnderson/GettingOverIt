"""Controlled serial/bounded-lag pipeline pilot; never resumes a training campaign."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from tests.analysis.benchmark_player_throughput import CHECKPOINT, write_json, stamp


class Monitor:
    def __init__(self, actor_pid):
        self.actor_pid = actor_pid
        self.done = threading.Event()
        self.peak_gpu = self.peak_rss = 0
        self.min_free = float("inf")
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        import psutil
        import torch
        while not self.done.is_set():
            free, total = torch.cuda.mem_get_info()
            self.peak_gpu = max(self.peak_gpu, (total-free)/2**30)
            self.min_free = min(self.min_free, psutil.virtual_memory().available/2**30)
            try:
                parent, actor = psutil.Process(), psutil.Process(self.actor_pid)
                processes = [parent, actor, *actor.children(recursive=True)]
                self.peak_rss = max(self.peak_rss, sum(p.memory_info().rss for p in processes)/2**30)
            except psutil.Error:
                pass
            self.done.wait(.2)

    def close(self):
        self.done.set(); self.thread.join(timeout=3)
        return {"whole_device_peak_used_gb":self.peak_gpu, "learner_actor_game_peak_rss_gb":self.peak_rss,
                "system_min_available_gb":self.min_free}


def run_trial(args):
    import numpy as np
    import torch
    from training.config import TrainConfig
    from training.experimental_policy import create_actor_critic
    from training.decision_buffer import DecisionRolloutBuffer
    from training.ppo_trainer import PPOTrainer, StalePolicyError
    from training.rollout_pipeline import RolloutPipeline, check_version
    from tests.analysis.optimize_training_budget import sample_visits
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**saved["config"])
    cfg.num_agents, cfg.ppo_batch_size, cfg.lr = args.agents, args.batch, args.lr
    cfg.random_deploy = cfg.sil_enabled = cfg.explore_boost_enabled = False
    cfg.checkpoint_dir, cfg.log_dir = str(output/"checkpoints"), str(output/"logs")
    cfg.ppo_inference_batch_size = 0
    model = create_actor_critic(cfg).cuda().eval()
    model.load_state_dict(saved["model"])
    behavior = create_actor_critic(cfg).cuda().eval()
    trainer = PPOTrainer(cfg)
    version, rows, error, pipeline, monitor = 0, [], None, None, None
    eval_scores = []
    write_json(output/"config.json", {"mode":args.mode,"config":asdict(cfg),"seed":args.seed,
        "max_behavior_kl":args.max_behavior_kl,"checkpoint":str(CHECKPOINT),"rounds":args.rounds})

    def snapshot(identifier):
        path = output/"checkpoints"/f"behavior_{identifier:04d}.pt"
        # detach().cpu().clone(): no shared learner storage, even on a CPU learner.
        torch.save({"version":version,"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "exploration_boost":model.exploration_boost}, path)
        return str(path)

    def submit(identifier, *, steps=1200, deterministic=False):
        pipeline.submit({"id":identifier,"version":version,"snapshot":snapshot(identifier),
                         "steps":steps,"seed":args.seed+identifier,"deterministic":deterministic})

    try:
        pipeline = RolloutPipeline(cfg, output, timeout=300)
        monitor = Monitor(pipeline.process.pid)
        submit(999, steps=64)  # CUDA, geometry and IPC warmup excluded.
        pipeline.receive()
        started = time.perf_counter()
        submit(0)
        for i in range(args.rounds):
            row = pipeline.receive()
            lag = check_version(row["version"], version)
            # Launch next collection with an immutable CURRENT snapshot, before
            # consuming this batch. No future request can get >1 version behind.
            if args.mode == "pipeline" and i+1 < args.rounds:
                submit(i+1)
            buffer = DecisionRolloutBuffer.load(row["buffer"])
            state = torch.load(row["snapshot"], map_location="cpu", weights_only=False)
            behavior.load_state_dict(state["model"])
            behavior.exploration_boost = state["exploration_boost"]
            np.random.seed(args.seed+10000+i)
            update_started = time.perf_counter()
            try:
                metrics = trainer.update(model, buffer, behavior_model=behavior,
                                          max_behavior_kl=args.max_behavior_kl)
                rejected = None
                if metrics["n_updates"]:
                    version += 1
            except StalePolicyError as exc:
                metrics, rejected = {"n_updates":0.}, str(exc)
            torch.cuda.synchronize()
            update_end = time.perf_counter()
            visits, unique = sample_visits(buffer.size, args.batch, metrics["n_updates"])
            row.update({"version_lag":lag,"learner_version_after":version,"metrics":metrics,
                "rejected":rejected,"update_started":update_started,"update_end":update_end,
                "update_seconds":update_end-update_started,"gradient_visits":visits,
                "unique_gradient_samples":unique,"trace":[] if rejected else trainer.last_update_trace})
            rows.append(row)
            write_json(output/"results.json", {"status":"running","rows":rows,"error":None})
            print(json.dumps({"mode":args.mode,"agents":args.agents,"round":i,
                "samples":buffer.environment_steps,"updates":metrics["n_updates"],"lag":lag,
                "rejected":rejected,"collection_s":row["collection_seconds"],"update_s":row["update_seconds"]}), flush=True)
            if args.mode == "serial" and i+1 < args.rounds:
                submit(i+1)
        elapsed = time.perf_counter()-started
        torch.save({"model":model.state_dict(),"optimizer":trainer.optimizer.state_dict(),"config":asdict(cfg)},
                   output/"checkpoints/final.pt")
        # Tiny behavior check only: these are NOT enough trials to compare success.
        submit(1000, deterministic=True)
        evaluation = pipeline.receive()
        eval_scores = evaluation["scores"]
        resources = monitor.close(); monitor = None
        samples = sum(r["physical_samples"] for r in rows)
        used = sum(r["unique_gradient_samples"] for r in rows)
        overlap = sum(max(0., min(a["update_end"], b["collection_end"])-max(a["update_started"], b["started"]))
                      for a,b in zip(rows, rows[1:]))
        summary = {"mode":args.mode,"agents":args.agents,"rounds":args.rounds,"seed":args.seed,
            "batch_size":args.batch,"lr":args.lr,"physical_samples":samples,"training_wall_seconds":elapsed,
            "physical_samples_per_second":samples/elapsed,"unique_gradient_samples_per_second":used/elapsed,
            "gradient_coverage":used/samples,"rejected_rollouts":sum(bool(r["rejected"]) for r in rows),
            "overlap_seconds":overlap,"max_version_lag":max(r["version_lag"] for r in rows),
            "max_behavior_audit_error":max(r["audit"]["preupdate_ratio_max_error"] for r in rows),**resources}
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if monitor:
            monitor.close()
        if pipeline:
            pipeline.close()
        write_json(output/"results.json", {"status":"failed" if error else "completed", "rows":rows,
            "error":error,"summary":locals().get("summary"),"evaluation":eval_scores})


def supervise(args):
    import psutil
    from training.config import TrainConfig
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    deadline = time.monotonic()+args.minutes*60
    source_paths = [*sorted((ROOT/"src/training").glob("*.py")), Path(__file__), CHECKPOINT]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    write_json(output/"pipeline_manifest.json", {"started":stamp(),"minutes":args.minutes,"hashes":hashes,
        "note":"3 full rollouts per arm; unequal physical samples across player counts; throughput pilot, not learning selection"})
    summaries, failures, child = [], [], None
    runtime = Path(TrainConfig().game_root)/"GoiData/runtime_config.json"
    runtime_before = runtime.read_bytes() if runtime.exists() else None
    # Persist recovery state before any child can change the external game file.
    (output/"checkpoints").mkdir(exist_ok=True)
    if runtime_before is not None:
        (output/"checkpoints/runtime_original.json").write_bytes(runtime_before)
    identity = {"controller_pid":os.getpid(), "controller_create_time":psutil.Process().create_time()}
    # Reverse serial/pipeline order at 4 players to reduce a simple ordering bias.
    for agents in (2,4,8):
        for mode in (("pipeline","serial") if agents==4 else ("serial","pipeline")):
            label = f"n{agents:02d}_{mode}"
            if time.monotonic()+60 > deadline:
                failures.append({"job":label,"error":"pilot deadline"}); break
            command = [sys.executable,"-m","src.tests.analysis.benchmark_rollout_pipeline","--trial",
                       "--mode",mode,"--agents",str(agents),"--rounds",str(args.rounds),"--batch",str(args.batch),
                       "--lr",str(args.lr),"--max-behavior-kl",str(args.max_behavior_kl),"--seed",str(args.seed),
                       "--output",str(output/label)]
            write_json(output/"pipeline_status.json", {"status":"running","job":label,"updated":stamp(),**identity})
            with (output/"logs"/f"{label}.log").open("w",encoding="utf-8") as log:
                child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    child.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],capture_output=True)
                    child.wait(timeout=10)
                    if runtime_before is not None and runtime.read_bytes()!=runtime_before:
                        runtime.write_bytes(runtime_before)
                    failures.append({"job":label,"error":"pilot deadline"}); break
            result_path = output/label/"results.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            if child.returncode or result.get("status") != "completed":
                failures.append({"job":label,"error":result.get("error",f"exit {child.returncode}")})
                # Stop on infrastructure/contract failure; don't produce partial rankings.
                break
            summaries.append(result["summary"])
            write_json(output/"pipeline_summary.json", {"rows":summaries,"failures":failures})
            print(json.dumps(result["summary"]),flush=True)
        if failures:
            break
    changed = [str(p) for p in source_paths if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p.relative_to(ROOT))]]
    if runtime_before is not None and runtime.read_bytes()!=runtime_before:
        runtime.write_bytes(runtime_before)
    write_json(output/"pipeline_status.json", {"status":"failed" if failures or changed else "completed",**identity,
        "finished":stamp(),"failures":failures,"changed_sources":changed,"completed":len(summaries)})
    if failures or changed:
        raise RuntimeError(f"Incomplete pilot: {failures}, changed sources: {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",action="store_true")
    parser.add_argument("--mode",choices=("serial","pipeline"),default="serial")
    parser.add_argument("--agents",type=int,choices=(2,4,8),default=2)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--batch",type=int,default=128)
    parser.add_argument("--lr",type=float,default=3e-5)
    parser.add_argument("--max-behavior-kl",type=float,default=.03)
    parser.add_argument("--seed",type=int,default=20260922)
    parser.add_argument("--minutes",type=int,default=35)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/pipeline_20260920")
    args = parser.parse_args()
    if (not 1 <= args.minutes <= 45 or args.rounds < 1 or args.batch < 1 or args.lr < 0
            or not 0 < args.max_behavior_kl <= .03):
        parser.error("Require 1–45 minutes, positive rounds/batch, nonnegative lr and behavior KL in (0, .03]")
    if args.trial:
        run_trial(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
