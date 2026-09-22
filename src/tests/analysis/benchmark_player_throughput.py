"""Measure real-game collection and PPO costs; resume the paused campaign on exit.

Benchmark updates are discarded. No training source, reward, or campaign setting is changed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timedelta
import gc
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
CAMPAIGN = ROOT / "runs/corridor_credit_20260919"
OUTPUT = ROOT / "runs/player_throughput_20260919"
CHECKPOINT = ROOT / "runs/corridor_credit_20260919_C_s7/checkpoints/ppo_iter_0022.pt"
STOP_REASON = "Requested throughput benchmark: pause at the next completed job boundary; resume the unchanged campaign after measurement."


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class Resources:
    def __init__(self, game_pid=None):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.game = psutil.Process(game_pid) if game_pid else None
        self.rss = self.game_rss = 0
        self.available = psutil.virtual_memory().available
        self.done = threading.Event()

    def sample(self):
        try:
            self.rss = max(self.rss, self.process.memory_info().rss)
            if self.game:
                self.game_rss = max(self.game_rss, self.game.memory_info().rss)
            self.available = min(self.available, self.psutil.virtual_memory().available)
        except self.psutil.Error:
            pass

    def start(self):
        self.sample()
        def loop():
            while not self.done.wait(.2):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.sample()
        return {"python_peak_rss_gb": self.rss/2**30, "game_peak_rss_gb": self.game_rss/2**30,
                "system_min_available_gb": self.available/2**30}


def setup(agents, output):
    import numpy as np
    import torch
    from training.config import TrainConfig
    from training.experimental_policy import create_actor_critic
    from training.rollout import RolloutWorker
    from training.reward import ClimbingEfficiencyMap
    from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**saved["config"])
    cfg.num_agents = agents
    cfg.ppo_inference_batch_size = 0
    cfg.random_deploy = cfg.sil_enabled = cfg.explore_boost_enabled = False
    cfg.checkpoint_dir = str(output / "checkpoints")
    cfg.log_dir = str(output / "logs")
    torch.manual_seed(20260921)
    np.random.seed(20260921)
    model = create_actor_critic(cfg).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    model.load_state_dict(saved["model"])
    worker = RolloutWorker(cfg)
    env, player = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env)
    eff_map = ClimbingEfficiencyMap(polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
        cache_path=str(output / "checkpoints/terrain.npz"), height_prior_weight=cfg.height_prior_weight)
    worker.set_terrain_info(eff_map.traversable, eff_map._min_gx, eff_map._min_gy)
    worker.set_deploy_obstacle_geometry(polygons, player)
    return saved, cfg, model, worker, eff_map


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_trial(agents, repeats, steps, output):
    import numpy as np
    import torch
    from training.policy_audit import audit_policy
    from training.ppo_trainer import PPOTrainer
    from training.rollout import _write_num_duplicates
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    saved, cfg, model, worker, eff_map = setup(agents, output)
    rows, failure = [], None
    write_json(output / "config.json", asdict(cfg))
    try:
        started = time.perf_counter()
        worker.launch_game()
        launch_seconds = time.perf_counter() - started
        worker.collect_ppo(model, 64, eff_map)  # CUDA/geometry warmup, excluded from measurements.
        for repeat in range(repeats):
            seed = 20260921 + repeat
            model.load_state_dict(saved["model"])
            np.random.seed(seed)
            torch.manual_seed(seed)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            resources = Resources(worker._game_process.pid).start()
            synchronize()
            started = time.perf_counter()
            buffer, trajectories = worker.collect_ppo(model, steps, eff_map)
            synchronize()
            collection_seconds = time.perf_counter() - started
            assert len(trajectories) == agents, "Some agents did not participate"
            assert buffer.environment_steps == sum(len(t.actions) for t in trajectories)
            assert all(np.isfinite(t.raw_states).all() and np.isfinite(t.actions).all() for t in trajectories)
            profile = dict(worker.last_profile)
            # Audit every stored decision, outside collection/update timing.
            started = time.perf_counter()
            audit = audit_policy(model, buffer)
            audit_seconds = time.perf_counter() - started
            before = [p.detach().clone() for p in model.parameters()]
            trainer = PPOTrainer(cfg)
            np.random.seed(seed + 100)
            started = time.perf_counter()
            update = trainer.update(model, buffer)
            synchronize()
            update_seconds = time.perf_counter() - started
            delta = sum(float((p.detach()-b).square().sum()) for p, b in zip(model.parameters(), before))**.5
            assert delta > 0 and all(torch.isfinite(p).all() for p in model.parameters())
            started = time.perf_counter()
            # Same complete-input archive/checkpoint pattern as training; artifacts are disposable.
            buffer.save(output / "checkpoints" / f"rollout_{repeat:02d}.npz")
            torch.save({"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                        "config": asdict(cfg)}, output / "checkpoints" / f"updated_{repeat:02d}.pt")
            serialization_seconds = time.perf_counter() - started
            arrays = {f"agent_{i}_{name}": getattr(t, name) for i, t in enumerate(trajectories)
                      for name in ("raw_states", "actions")}
            np.savez_compressed(output / "checkpoints" / f"states_{repeat:02d}.npz", **arrays)
            actions = np.concatenate([t.actions for t in trajectories])
            def corr(x, y):
                return float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) if x.std() > 1e-8 and y.std() > 1e-8 else None
            lag1 = [corr(t.actions[:-1], t.actions[1:]) for t in trajectories if len(t.actions)>2]
            prefix = min(len(t.actions) for t in trajectories)
            pairs = [corr(trajectories[i].actions[:prefix], trajectories[j].actions[:prefix])
                     for i in range(agents) for j in range(i+1, agents)]
            pairs = [v for v in pairs if v is not None]
            row = {"agents": agents, "repeat": repeat, "seed": seed, "steps_cap": steps,
                   "physical_samples": buffer.environment_steps, "active_fraction": buffer.environment_steps/(steps*agents),
                   "episode_lengths": [len(t.actions) for t in trajectories],
                   "collection_seconds": collection_seconds, "update_seconds": update_seconds,
                   "serialization_seconds": serialization_seconds, "audit_seconds": audit_seconds,
                   "launch_seconds": launch_seconds, "profile": profile, "audit": audit,
                   "update": update, "parameter_delta_l2": delta, **resources.finish(),
                   "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0,
                   "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0,
                   "action_lag1_mean": float(np.mean([v for v in lag1 if v is not None])),
                   "cross_agent_action_correlation_mean": float(np.mean(pairs)) if pairs else None,
                   "action_saturation_fraction": float(np.mean(np.any(abs(actions)>=99.9, axis=1)))}
            row["collection_samples_per_second"] = row["physical_samples"] / collection_seconds
            row["end_to_end_samples_per_second"] = row["physical_samples"] / (collection_seconds + update_seconds + serialization_seconds)
            rows.append(row)
            write_json(output / "results.json", {"rows": rows, "failure": None})
            print(json.dumps({k: row[k] for k in ("agents", "repeat", "physical_samples", "collection_samples_per_second", "end_to_end_samples_per_second", "gpu_peak_reserved_gb")}), flush=True)
            del buffer, trajectories, trainer, before, arrays
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        worker.close_game()
        _write_num_duplicates(Path(cfg.game_root), 2)
        write_json(output / "results.json", {"rows": rows, "failure": failure})


def timed_update(saved, buffer, batch, seed, *, fixed=False, agents=2):
    import numpy as np
    import torch
    from training.config import TrainConfig
    from training.experimental_policy import create_actor_critic
    from training.ppo_trainer import PPOTrainer
    model = trainer = None
    try:
        cfg = TrainConfig(**saved["config"])
        cfg.num_agents, cfg.ppo_batch_size = agents, batch
        if fixed:
            cfg.lr, cfg.ppo_epochs = 0., 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = create_actor_critic(cfg).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.load_state_dict(saved["model"])
        trainer = PPOTrainer(cfg)
        np.random.seed(seed)
        torch.manual_seed(seed)
        synchronize()
        start = time.perf_counter()
        metrics = trainer.update(model, buffer)
        synchronize()
        seconds = time.perf_counter()-start
        assert metrics["n_updates"] > 0 and all(torch.isfinite(p).all() for p in model.parameters())
        if fixed:
            assert metrics["n_updates"] == (buffer.size+batch-1)//batch
            assert all(torch.equal(value.cpu(),saved["model"][name]) for name,value in model.state_dict().items())
        return {"seconds":seconds,"metrics":metrics,"failure":None,
                "gpu_peak_reserved_gb":torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0}
    except torch.cuda.OutOfMemoryError as exc:
        return {"failure":repr(exc),"batch_size":batch}
    finally:
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def minibatch_trial(output):
    """Equal-work backward benchmark: lr=0, one pass, same data, unchanged KL guard."""
    import numpy as np
    import torch
    from training.config import TrainConfig
    from training.experimental_policy import create_actor_critic
    from training.decision_buffer import DecisionRolloutBuffer
    from training.ppo_trainer import PPOTrainer
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    buffer = DecisionRolloutBuffer.load(output / "n02_start/checkpoints/rollout_00.npz")
    rows, failures, blocked = [], [], set()
    for repeat, order in enumerate(((64, 128, 256), (256, 128, 64))):
        for batch in order:
            if batch in blocked:
                continue
            result = timed_update(saved, buffer, batch, 20260930+repeat, fixed=True)
            if result["failure"]:
                blocked.add(batch)
                failures.append({"batch_size":batch,"stage":"fixed_work",**result})
            else:
                rows.append({"batch_size":batch,"repeat":repeat,**result,
                             "physical_samples":buffer.environment_steps,"samples_per_second":buffer.size/result["seconds"],
                             "weights_unchanged":True})
            write_json(output / "minibatches.json", {"scope":"one complete equal-data pass, lr=0, original KL guard; no learning claims", "rows":rows,"failures":failures})
    # Joint search with REAL, discarded updates on each archived rollout.
    # Report KL early stopping separately from fixed-work kernel throughput.
    joint = []
    write_json(output / "joint_updates.json", {"scope":"fresh optimizer, same frozen behavior weights and raw samples, original learning rate and KL guard; updates discarded", "rows":joint,"failures":failures})
    for group in sorted(output.glob("n*_*")):
        source_rows = json.loads((group / "results.json").read_text(encoding="utf-8"))["rows"]
        for source_row in source_rows:
            buffer = DecisionRolloutBuffer.load(group / "checkpoints" / f"rollout_{source_row['repeat']:02d}.npz")
            for batch in (128, 256):
                if batch in blocked:
                    continue
                result = timed_update(saved,buffer,batch,source_row["seed"]+100,agents=source_row["agents"])
                if result["failure"]:
                    blocked.add(batch)
                    failures.append({"batch_size":batch,"stage":group.name,"repeat":source_row["repeat"],**result})
                else:
                    joint.append({"group":group.name,"agents":source_row["agents"],"repeat":source_row["repeat"],
                                  "batch_size":batch,"physical_samples":buffer.environment_steps,
                                  "update_seconds":result.pop("seconds"),**result})
                write_json(output / "joint_updates.json", {"scope":"fresh optimizer, same frozen behavior weights and raw samples, original learning rate and KL guard; updates discarded", "rows":joint,"failures":failures})


def resume_campaign():
    marker = CAMPAIGN / "STOP"
    if not marker.exists() or marker.read_text(encoding="utf-8-sig").strip() != STOP_REASON:
        return {"resumed": False, "reason": "STOP marker absent or changed; leave user control intact"}
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(manifest["deadline"])
    if datetime.now().astimezone() >= deadline:
        return {"resumed": False, "reason": "Original training deadline reached"}
    marker.unlink()
    invocation = datetime.now().strftime("%H%M%S")
    command = [sys.executable, "-m", "src.tests.analysis.run_corridor_credit", "--deadline", manifest["deadline"]]
    with (CAMPAIGN / f"logs/controller_resume_{invocation}.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    result = {"resumed": True, "pid": child.pid, "time": stamp(), "command": command,
              "note": "Same two agents and original deadline; benchmark weights not used"}
    write_json(CAMPAIGN / "benchmark_resume.json", result)
    return result


def supervise(minutes):
    import psutil
    import torch
    state = json.loads((CAMPAIGN / "status.json").read_text(encoding="utf-8"))
    if state["status"] == "running" or (CAMPAIGN / "controller.lock").exists():
        raise RuntimeError("Wait for the current campaign to stop at its job boundary")
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("A game instance is still running")
    if OUTPUT.exists():
        raise RuntimeError("Refuse to overwrite an existing benchmark")
    OUTPUT.mkdir()
    (OUTPUT / "logs").mkdir()
    deadline = datetime.now().astimezone() + timedelta(minutes=minutes)
    hardware = {"logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
                "ram_gb": psutil.virtual_memory().total/2**30, "available_ram_gb": psutil.virtual_memory().available/2**30,
                "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else 0}
    manifest = {"started_at": stamp(), "deadline": deadline.isoformat(), "checkpoint": str(CHECKPOINT),
                "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "hardware": hardware,
                "groups": [2,4,8,16,2], "repeats": 3, "steps": 1200,
                "scope": "frozen policy collection; discard real PPO updates; repeated baseline controls time drift"}
    write_json(OUTPUT / "manifest.json", manifest)
    write_json(OUTPUT / "campaign_pause.json", state)
    child, error, completed = None, None, []
    try:
        jobs = [(f"n{n:02d}_{suffix}", ["--trial", "--agents", str(n), "--output", str(OUTPUT/f"n{n:02d}_{suffix}")])
                for n, suffix in ((2,"start"),(4,"main"),(8,"main"),(16,"main"),(2,"end"))]
        jobs.append(("minibatches", ["--minibatches", "--output", str(OUTPUT)]))
        for label, arguments in jobs:
            if datetime.now().astimezone() + timedelta(seconds=90) >= deadline:
                raise TimeoutError("Benchmark wall-clock budget exhausted")
            command = [sys.executable, "-m", "src.tests.analysis.benchmark_player_throughput", *arguments]
            write_json(OUTPUT / "status.json", {"status":"running", "job":label, "updated_at":stamp(), "deadline":deadline.isoformat()})
            with (OUTPUT / "logs" / f"{label}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                while child.poll() is None:
                    if datetime.now().astimezone() >= deadline:
                        raise TimeoutError("Benchmark deadline reached")
                    time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"Benchmark {label} failed with code {child.returncode}")
            completed.append(label)
            print(json.dumps({"completed":label, "time":stamp()}), flush=True)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if child is not None and child.poll() is None:
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
            child.wait(timeout=30)
        result = {"status":"failed" if error else "completed", "finished_at":stamp(), "completed":completed, "error":error}
        result["campaign_resume"] = resume_campaign()
        write_json(OUTPUT / "status.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--minibatches", action="store_true")
    parser.add_argument("--agents", type=int, choices=(2,4,8,16), default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.trial:
        collect_trial(args.agents, 3, 1200, args.output)
    elif args.minibatches:
        minibatch_trial(args.output)
    else:
        if not 1 <= args.minutes <= 45:
            parser.error("Benchmark budget must be 1–45 minutes")
        supervise(args.minutes)


if __name__ == "__main__":
    main()
